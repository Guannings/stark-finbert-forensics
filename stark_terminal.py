import sys
import os
import duckdb
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
    QWidget, QLineEdit, QPushButton, QLabel, QFrame, QComboBox,
    QTableWidget, QTableWidgetItem, QHeaderView,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
import pyqtgraph as pg

from headline_analyzer import (
    score_headline_live, find_similar_headlines, extract_keywords,
    compute_forward_returns, compute_verdict,
)

# ── Styling ──────────────────────────────────────────────────────────
DARK_BG = "#0b0c0e"
PANEL_BG = "#111114"
BORDER = "#2a2a2e"
NEON_GREEN = "#00FF87"
NEON_RED = "#FF4455"
NEON_CYAN = "#00E5FF"
TEXT_COLOR = "#E0E0E0"
DIM_TEXT = "#666"

PARQUET_GLOB = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "STARK_SCORED_FIXED.parquet"
)


# ── Custom Date Axis ─────────────────────────────────────────────────
class DateAxis(pg.AxisItem):
    """Converts Unix-epoch seconds on the x-axis to readable date strings."""

    def tickStrings(self, values, scale, spacing):
        strings = []
        for v in values:
            try:
                dt = datetime.utcfromtimestamp(v)
                if spacing > 86400 * 180:
                    strings.append(dt.strftime("%Y"))
                elif spacing > 86400 * 10:
                    strings.append(dt.strftime("%b %Y"))
                else:
                    strings.append(dt.strftime("%b %d"))
            except (OSError, OverflowError, ValueError):
                strings.append("")
        return strings


# ── Main Window ──────────────────────────────────────────────────────
class StarkTerminal(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("STARK INDUSTRIES // QUANT TERMINAL")
        self.resize(1400, 900)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(12)

        # ── Header row ───────────────────────────────────────────────
        header = QHBoxLayout()

        title = QLabel("STARK ANALYTICS")
        title.setStyleSheet(
            f"font-size: 24px; font-weight: bold; color: {NEON_GREEN}; letter-spacing: 2px;"
        )

        self.ticker_input = QLineEdit()
        self.ticker_input.setPlaceholderText("TICKER (e.g. NVDA)")
        self.ticker_input.setStyleSheet(
            f"background-color: #1a1a1a; color: {NEON_GREEN}; "
            f"border: 1px solid {BORDER}; padding: 10px; font-family: monospace; font-size: 14px;"
        )
        self.ticker_input.setFixedWidth(180)
        self.ticker_input.returnPressed.connect(self.load_data)

        self.window_combo = QComboBox()
        self.window_combo.addItems(["All Time", "1 Year", "6 Months", "3 Months", "1 Month"])
        self.window_combo.setStyleSheet(
            f"background-color: #1a1a1a; color: {TEXT_COLOR}; "
            f"border: 1px solid {BORDER}; padding: 8px; font-family: monospace;"
        )
        self.window_combo.setFixedWidth(130)

        run_btn = QPushButton("EXECUTE")
        run_btn.setStyleSheet(
            f"background-color: #1a1a2a; color: white; border: 1px solid {NEON_GREEN}; "
            f"padding: 10px 20px; font-weight: bold; font-family: monospace;"
        )
        run_btn.clicked.connect(self.load_data)

        self.status_label = QLabel("SYSTEM READY")
        self.status_label.setStyleSheet(f"color: {DIM_TEXT}; font-family: monospace; font-size: 12px;")

        header.addWidget(title)
        header.addStretch()
        header.addWidget(self.ticker_input)
        header.addWidget(self.window_combo)
        header.addWidget(run_btn)
        root.addLayout(header)

        # ── Headline input row ────────────────────────────────────────
        headline_row = QHBoxLayout()

        self.headline_input = QLineEdit()
        self.headline_input.setPlaceholderText("Enter headline to analyze (e.g. 'earnings beat expectations')")
        self.headline_input.setStyleSheet(
            f"background-color: #1a1a1a; color: {NEON_CYAN}; "
            f"border: 1px solid {BORDER}; padding: 10px; font-family: monospace; font-size: 14px;"
        )
        self.headline_input.returnPressed.connect(self.analyze_headline)

        analyze_btn = QPushButton("ANALYZE")
        analyze_btn.setStyleSheet(
            f"background-color: #1a1a2a; color: white; border: 1px solid {NEON_CYAN}; "
            f"padding: 10px 20px; font-weight: bold; font-family: monospace;"
        )
        analyze_btn.clicked.connect(self.analyze_headline)

        headline_row.addWidget(self.headline_input, stretch=1)
        headline_row.addWidget(analyze_btn)
        root.addLayout(headline_row)

        # ── Status bar ───────────────────────────────────────────────
        root.addWidget(self.status_label)

        # ── Metrics row ──────────────────────────────────────────────
        metrics = QHBoxLayout()
        self.metric_price = self._make_card("CURRENT PRICE", "$---")
        self.metric_sent = self._make_card("AVG SENTIMENT (30 d)", "---")
        self.metric_vol = self._make_card("NEWS ARTICLES", "---")
        self.metric_range = self._make_card("DATE RANGE", "---")
        for w in (self.metric_price, self.metric_sent, self.metric_vol, self.metric_range):
            metrics.addWidget(w)
        root.addLayout(metrics)

        # ── Chart config ─────────────────────────────────────────────
        pg.setConfigOption("background", DARK_BG)
        pg.setConfigOption("foreground", "#888")

        # Price chart
        self.price_plot = pg.PlotWidget(
            title="PRICE ACTION",
            axisItems={"bottom": DateAxis(orientation="bottom")},
        )
        self.price_plot.showGrid(x=True, y=True, alpha=0.15)
        self.price_plot.setLabel("left", "Price ($)")
        self.price_plot.getAxis("left").setWidth(70)
        root.addWidget(self.price_plot, stretch=3)

        # Sentiment chart (linked x-axis)
        self.sent_plot = pg.PlotWidget(
            title="SENTIMENT OSCILLATOR",
            axisItems={"bottom": DateAxis(orientation="bottom")},
        )
        self.sent_plot.showGrid(x=True, y=True, alpha=0.15)
        self.sent_plot.setLabel("left", "Score")
        self.sent_plot.getAxis("left").setWidth(70)
        self.sent_plot.setXLink(self.price_plot)
        root.addWidget(self.sent_plot, stretch=2)

        # ── Headline analysis results frame (hidden initially) ─────
        self.results_frame = QFrame()
        self.results_frame.setStyleSheet(
            f"background-color: {PANEL_BG}; border: 1px solid {BORDER}; border-radius: 6px;"
        )
        results_layout = QVBoxLayout(self.results_frame)
        results_layout.setContentsMargins(14, 10, 14, 10)

        # Verdict + FinBERT score labels
        results_header = QHBoxLayout()
        self.verdict_label = QLabel("VERDICT: —")
        self.verdict_label.setStyleSheet(
            f"color: {NEON_GREEN}; font-size: 16px; font-weight: bold; "
            f"font-family: monospace; border: none;"
        )
        self.finbert_label = QLabel("FinBERT: —")
        self.finbert_label.setStyleSheet(
            f"color: {NEON_CYAN}; font-size: 14px; font-family: monospace; border: none;"
        )
        results_header.addWidget(self.verdict_label)
        results_header.addStretch()
        results_header.addWidget(self.finbert_label)
        results_layout.addLayout(results_header)

        # Matches table
        self.matches_table = QTableWidget()
        self.matches_table.setColumnCount(5)
        self.matches_table.setHorizontalHeaderLabels(["Date", "Headline", "Sentiment", "1D %", "5D %"])
        self.matches_table.horizontalHeader().setStyleSheet(
            f"color: {NEON_CYAN}; font-family: monospace; font-weight: bold; border: none;"
        )
        self.matches_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.matches_table.setStyleSheet(
            f"background-color: {DARK_BG}; color: {TEXT_COLOR}; "
            f"font-family: monospace; font-size: 12px; border: none; "
            f"gridline-color: {BORDER};"
        )
        self.matches_table.verticalHeader().setVisible(False)
        self.matches_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        results_layout.addWidget(self.matches_table)

        self.results_frame.setVisible(False)
        root.addWidget(self.results_frame, stretch=2)

        # Window theme
        self.setStyleSheet(f"background-color: {DARK_BG};")

    # ── helpers ──────────────────────────────────────────────────────
    def _make_card(self, title: str, value: str) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet(
            f"background-color: {PANEL_BG}; border: 1px solid {BORDER}; border-radius: 6px;"
        )
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(14, 10, 14, 10)

        t = QLabel(title)
        t.setStyleSheet(f"color: {DIM_TEXT}; font-size: 10px; font-weight: bold; letter-spacing: 1px; border: none;")

        v = QLabel(value)
        v.setObjectName("value")
        v.setStyleSheet(
            "color: white; font-size: 17px; font-family: monospace; font-weight: bold; border: none;"
        )

        lay.addWidget(t)
        lay.addWidget(v)
        return frame

    def _set_card(self, card: QFrame, value: str, color: str | None = None):
        label = card.findChild(QLabel, "value")
        label.setText(value)
        c = color or "white"
        label.setStyleSheet(
            f"color: {c}; font-size: 17px; font-family: monospace; font-weight: bold; border: none;"
        )

    # ── headline analysis ───────────────────────────────────────────
    def analyze_headline(self):
        ticker = self.ticker_input.text().strip().upper()
        headline = self.headline_input.text().strip()
        if not ticker or not headline:
            self.status_label.setText("ENTER TICKER AND HEADLINE TO ANALYZE")
            self.status_label.setStyleSheet(f"color: {NEON_RED}; font-family: monospace; font-size: 12px;")
            return

        self.status_label.setText(f"ANALYZING HEADLINE FOR {ticker} ...")
        self.status_label.setStyleSheet(f"color: {NEON_CYAN}; font-family: monospace; font-size: 12px;")
        QApplication.processEvents()

        try:
            # FinBERT live score
            live_score = score_headline_live(headline)
            if live_score is not None:
                sc = NEON_GREEN if live_score > 0.05 else NEON_RED if live_score < -0.05 else TEXT_COLOR
                self.finbert_label.setText(f"FinBERT: {live_score:+.4f}")
                self.finbert_label.setStyleSheet(
                    f"color: {sc}; font-size: 14px; font-family: monospace; border: none;"
                )
            else:
                self.finbert_label.setText("FinBERT: unavailable")

            # Find similar headlines
            keywords = extract_keywords(headline)
            matches = find_similar_headlines(ticker, keywords, top_n=20)

            # Compute verdict
            if not matches.empty:
                returns = compute_forward_returns(ticker, matches["date"])
                verdict = compute_verdict(matches, returns)
            else:
                returns = {}
                verdict = compute_verdict(matches, {})

            # Update verdict label
            v = verdict["verdict"]
            conf = verdict["confidence"]
            vc = NEON_GREEN if v == "BULLISH" else NEON_RED if v == "BEARISH" else "#FFD700"
            icon = "\u25B2" if v == "BULLISH" else "\u25BC" if v == "BEARISH" else "\u25CF"
            self.verdict_label.setText(f"{icon} {v}  |  Confidence: {conf}%")
            self.verdict_label.setStyleSheet(
                f"color: {vc}; font-size: 16px; font-weight: bold; "
                f"font-family: monospace; border: none;"
            )

            # Populate matches table
            self.matches_table.setRowCount(len(matches))
            for i, (_, row) in enumerate(matches.iterrows()):
                date_str = pd.Timestamp(row["date"]).strftime("%Y-%m-%d") if pd.notna(row["date"]) else "—"
                hl = str(row["headline"]).replace("\n", " ").strip()
                sent = row["sentiment_score"]
                d_norm = pd.Timestamp(row["date"]).normalize()
                ret = returns.get(d_norm, {})

                self.matches_table.setItem(i, 0, QTableWidgetItem(date_str))
                self.matches_table.setItem(i, 1, QTableWidgetItem(hl))

                sent_item = QTableWidgetItem(f"{sent:+.3f}")
                sent_color = NEON_GREEN if sent > 0.1 else NEON_RED if sent < -0.1 else TEXT_COLOR
                sent_item.setForeground(QColor(sent_color))
                self.matches_table.setItem(i, 2, sent_item)

                for col, key in [(3, "1d"), (4, "5d")]:
                    val = ret.get(key)
                    if val is not None:
                        item = QTableWidgetItem(f"{val:+.2f}%")
                        rc = NEON_GREEN if val > 0 else NEON_RED if val < 0 else TEXT_COLOR
                        item.setForeground(QColor(rc))
                    else:
                        item = QTableWidgetItem("—")
                        item.setForeground(QColor(DIM_TEXT))
                    self.matches_table.setItem(i, col, item)

            self.results_frame.setVisible(True)
            self.status_label.setText(
                f"HEADLINE ANALYSIS COMPLETE  ·  {len(matches)} matches for {ticker}"
            )
            self.status_label.setStyleSheet(f"color: {NEON_GREEN}; font-family: monospace; font-size: 12px;")

        except Exception as e:
            self.status_label.setText(f"ANALYSIS ERROR: {e}")
            self.status_label.setStyleSheet(f"color: {NEON_RED}; font-family: monospace; font-size: 12px;")

    # ── core logic ───────────────────────────────────────────────────
    def load_data(self):
        ticker = self.ticker_input.text().strip().upper()
        if not ticker:
            return

        self.status_label.setText(f"QUERYING CORE FOR {ticker} ...")
        self.status_label.setStyleSheet(f"color: {NEON_CYAN}; font-family: monospace; font-size: 12px;")
        QApplication.processEvents()

        try:
            # ── 1. Query sentiment from parquet ──────────────────────
            con = duckdb.connect()
            df_sent = con.execute(
                """
                SELECT date, sentiment_score
                FROM read_parquet($1)
                WHERE ticker = $2
                  AND date >= '2000-01-01'
                ORDER BY date
                """,
                [PARQUET_GLOB, ticker],
            ).fetchdf()
            con.close()

            if df_sent.empty:
                self.status_label.setText(f"NO SENTIMENT DATA FOR {ticker}")
                self.status_label.setStyleSheet(f"color: {NEON_RED}; font-family: monospace; font-size: 12px;")
                return

            total_articles = len(df_sent)

            # Normalize dates
            df_sent["date"] = pd.to_datetime(df_sent["date"], utc=True).dt.tz_localize(None).dt.normalize()

            # Daily aggregation
            daily = df_sent.groupby("date")["sentiment_score"].agg(["mean", "count"]).rename(
                columns={"mean": "sentiment", "count": "articles"}
            )

            # ── 2. Time window filter ────────────────────────────────
            window = self.window_combo.currentText()
            end_date = daily.index.max()
            if window == "1 Year":
                start = end_date - pd.DateOffset(years=1)
            elif window == "6 Months":
                start = end_date - pd.DateOffset(months=6)
            elif window == "3 Months":
                start = end_date - pd.DateOffset(months=3)
            elif window == "1 Month":
                start = end_date - pd.DateOffset(months=1)
            else:
                start = daily.index.min()
            daily = daily.loc[start:]

            # ── 3. Fetch price data from yfinance ────────────────────
            price_start = daily.index.min().strftime("%Y-%m-%d")
            prices = yf.download(ticker, start=price_start, progress=False)
            if prices.empty:
                self.status_label.setText(f"NO PRICE DATA FOR {ticker}")
                self.status_label.setStyleSheet(f"color: {NEON_RED}; font-family: monospace; font-size: 12px;")
                return

            if isinstance(prices.columns, pd.MultiIndex):
                prices = prices.xs(ticker, axis=1, level=1)

            prices.index = prices.index.tz_localize(None).normalize()

            # ── 4. Merge on date ─────────────────────────────────────
            df = prices[["Close"]].join(daily).dropna(subset=["sentiment"])
            if df.empty:
                self.status_label.setText(f"NO OVERLAPPING DATA FOR {ticker}")
                self.status_label.setStyleSheet(f"color: {NEON_RED}; font-family: monospace; font-size: 12px;")
                return

            # ── 5. Draw charts ───────────────────────────────────────
            self.price_plot.clear()
            self.sent_plot.clear()

            # X-axis: Unix seconds
            x = df.index.astype(np.int64) // 10**9

            # -- Price line
            self.price_plot.plot(
                x, df["Close"].values,
                pen=pg.mkPen(color=NEON_CYAN, width=2),
                name="Close",
            )

            # -- Sentiment bars (green / red)
            scores = df["sentiment"].values
            colors = [QColor(NEON_GREEN) if s >= 0 else QColor(NEON_RED) for s in scores]
            bar_width = 86400 * 0.8  # ~1 day
            bars = pg.BarGraphItem(x=x, height=scores, width=bar_width, brushes=colors)
            self.sent_plot.addItem(bars)

            # -- 7-day rolling sentiment line
            if len(df) >= 7:
                roll = df["sentiment"].rolling(7, min_periods=1).mean()
                self.sent_plot.plot(
                    x, roll.values,
                    pen=pg.mkPen(color="#FFD700", width=2, style=Qt.PenStyle.DashLine),
                    name="7d MA",
                )

            # Zero line on sentiment
            self.sent_plot.addLine(y=0, pen=pg.mkPen(color="#555", width=1, style=Qt.PenStyle.DotLine))

            # ── 6. Update metric cards ───────────────────────────────
            last_price = df["Close"].iloc[-1]
            avg_30 = scores[-30:].mean() if len(scores) > 0 else 0.0
            date_min = df.index.min().strftime("%Y-%m-%d")
            date_max = df.index.max().strftime("%Y-%m-%d")

            self._set_card(self.metric_price, f"${last_price:,.2f}")
            self._set_card(
                self.metric_sent,
                f"{avg_30:+.4f}",
                NEON_GREEN if avg_30 >= 0 else NEON_RED,
            )
            self._set_card(self.metric_vol, f"{total_articles:,} articles")
            self._set_card(self.metric_range, f"{date_min}  →  {date_max}")

            self.status_label.setText(
                f"LOADED {len(df)} trading days for {ticker}  ·  {total_articles:,} articles scored"
            )
            self.status_label.setStyleSheet(f"color: {NEON_GREEN}; font-family: monospace; font-size: 12px;")

        except Exception as e:
            self.status_label.setText(f"ERROR: {e}")
            self.status_label.setStyleSheet(f"color: {NEON_RED}; font-family: monospace; font-size: 12px;")


# ── Entry point ──────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StarkTerminal()
    window.show()
    sys.exit(app.exec())
