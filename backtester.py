#!/usr/bin/env python3
"""
STARK Strategy Backtester — sentiment-momentum backtest with visual forensics.

Daily sentiment is derived on the fly from the same headline dataset the
analyzer queries (no separate CSV needed). Prices come from yfinance.
Signals execute with a one-day lag to avoid look-ahead bias, and the equity
curve accounts for transaction costs and volatility-targeted position sizing.
"""

import os

import numpy as np
import pandas as pd
import duckdb
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from headline_analyzer import INDEX_PATH, PARQUET_PATH, score_headline_live

# --- CONFIGURATION ---
INITIAL_CAPITAL = 1_000_000
TARGET_VOLATILITY = 0.40      # annualized vol target for position sizing
MAX_LEVERAGE = 1.0            # single-ticker backtest: cap at fully invested
TRANSACTION_COST = 0.0015     # 15 bps per side, applied on position changes
SENTIMENT_THRESHOLD = 0.5
SENTIMENT_FFILL_DAYS = 5      # carry sentiment forward over quiet news days


# ── Data loading ──────────────────────────────────────────────────────────

def load_daily_sentiment(ticker: str) -> pd.DataFrame:
    """Aggregate per-headline sentiment into a daily score for one ticker."""
    src = INDEX_PATH if os.path.exists(INDEX_PATH) else PARQUET_PATH
    if not os.path.exists(src):
        raise FileNotFoundError(
            f"No headline data found. Expected {INDEX_PATH} or {PARQUET_PATH}."
        )
    con = duckdb.connect()
    try:
        df = con.execute(
            """
            SELECT CAST(date AS DATE) AS date,
                   AVG(sentiment_score) AS daily_score,
                   COUNT(*) AS articles
            FROM read_parquet($1)
            WHERE ticker = $2
            GROUP BY 1
            ORDER BY 1
            """,
            [src, ticker.upper()],
        ).fetchdf()
    finally:
        con.close()
    return df


def build_frame(ticker: str) -> pd.DataFrame | None:
    """Join daily sentiment with prices and compute strategy indicators."""
    sent = load_daily_sentiment(ticker)
    if sent.empty:
        return None

    sent["date"] = pd.to_datetime(sent["date"])
    sent = sent.set_index("date")

    price_start = (sent.index.min() - pd.Timedelta(days=100)).strftime("%Y-%m-%d")
    prices = yf.download(ticker, start=price_start, progress=False, auto_adjust=True)
    if prices.empty:
        return None
    if isinstance(prices.columns, pd.MultiIndex):
        prices = prices.xs(ticker.upper(), axis=1, level=1)
    prices.index = prices.index.tz_localize(None).normalize()

    df = prices[["Close"]].rename(columns={"Close": "close"})
    df = df.join(sent)
    df["daily_score"] = df["daily_score"].ffill(limit=SENTIMENT_FFILL_DAYS)

    df["returns"] = df["close"].pct_change()
    df["volatility_20d"] = df["returns"].rolling(20).std() * np.sqrt(252)
    df["smooth_sentiment"] = df["daily_score"].rolling(3, min_periods=1).mean()
    df["SMA_50"] = df["close"].rolling(50).mean()

    # 1 = long, 0 = cash (decided on day t, executed on day t+1)
    df["signal"] = np.where(
        (df["smooth_sentiment"] > SENTIMENT_THRESHOLD) & (df["close"] > df["SMA_50"]),
        1, 0,
    )
    return df


# ── Backtest engine ───────────────────────────────────────────────────────

def run_backtest(df: pd.DataFrame) -> pd.DataFrame:
    """Compute positions, costs, and equity curves. Signals lag one day."""
    df = df.copy()

    vol_weight = (TARGET_VOLATILITY / df["volatility_20d"]) \
        .replace([np.inf, -np.inf], 0).fillna(0).clip(upper=MAX_LEVERAGE)

    # Position held during day t comes from day t-1's signal — no look-ahead
    df["position"] = (df["signal"] * vol_weight).shift(1).fillna(0)
    df["turnover"] = df["position"].diff().abs().fillna(df["position"].abs())
    df["strategy_returns"] = (
        df["position"] * df["returns"] - df["turnover"] * TRANSACTION_COST
    )

    df["equity"] = INITIAL_CAPITAL * (1 + df["strategy_returns"].fillna(0)).cumprod()
    df["buyhold_equity"] = INITIAL_CAPITAL * (1 + df["returns"].fillna(0)).cumprod()
    return df


def _curve_stats(returns: pd.Series, equity: pd.Series) -> dict:
    returns = returns.dropna()
    n_days = len(returns)
    if n_days == 0:
        return {}
    total = equity.iloc[-1] / INITIAL_CAPITAL - 1
    years = n_days / 252
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 and total > -1 else float("nan")
    vol = returns.std() * np.sqrt(252)
    sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else float("nan")
    drawdown = equity / equity.cummax() - 1
    return {
        "total_return": total,
        "cagr": cagr,
        "volatility": vol,
        "sharpe": sharpe,
        "max_drawdown": drawdown.min(),
    }


def compute_metrics(df: pd.DataFrame) -> dict:
    strat = _curve_stats(df["strategy_returns"], df["equity"])
    bench = _curve_stats(df["returns"], df["buyhold_equity"])

    entries = ((df["signal"] == 1) & (df["signal"].shift(1) == 0)).sum()
    exposure = (df["position"] > 0).mean()
    active = df.loc[df["position"] > 0, "strategy_returns"].dropna()
    win_rate = (active > 0).mean() if len(active) else float("nan")

    return {
        "strategy": strat,
        "buyhold": bench,
        "n_trades": int(entries),
        "exposure": exposure,
        "daily_win_rate": win_rate,
    }


def print_metrics(ticker: str, m: dict):
    def pct(x):
        return f"{x * 100:+.2f}%" if x == x else "—"  # NaN-safe

    s, b = m["strategy"], m["buyhold"]
    print(f"\n  ── BACKTEST RESULTS: {ticker} ──")
    print(f"  {'':22}{'STRATEGY':>12}{'BUY & HOLD':>14}")
    print(f"  {'Total Return':22}{pct(s.get('total_return', float('nan'))):>12}{pct(b.get('total_return', float('nan'))):>14}")
    print(f"  {'CAGR':22}{pct(s.get('cagr', float('nan'))):>12}{pct(b.get('cagr', float('nan'))):>14}")
    print(f"  {'Annualized Vol':22}{pct(s.get('volatility', float('nan'))):>12}{pct(b.get('volatility', float('nan'))):>14}")
    print(f"  {'Sharpe':22}{s.get('sharpe', float('nan')):>12.2f}{b.get('sharpe', float('nan')):>14.2f}")
    print(f"  {'Max Drawdown':22}{pct(s.get('max_drawdown', float('nan'))):>12}{pct(b.get('max_drawdown', float('nan'))):>14}")
    print(f"  {'Trades':22}{m['n_trades']:>12}")
    print(f"  {'Time in Market':22}{pct(m['exposure'])[1:]:>12}")
    if m["daily_win_rate"] == m["daily_win_rate"]:
        print(f"  {'Daily Win Rate':22}{pct(m['daily_win_rate'])[1:]:>12}")
    print()


# ── FinBERT Headline Overlay ──────────────────────────────────────────────

def analyze_headline_overlay(ticker, ticker_df, headline):
    """Score headline with FinBERT and find signal zones where sentiment aligns.

    Forward returns are measured from zone ENTRY days only, so overlapping
    days inside a zone don't inflate the sample.
    Returns (finbert_score, signal_zones_df, summary_dict) or (None, None, None).
    """
    finbert_score = score_headline_live(headline)
    if finbert_score is None:
        print("  FinBERT scoring failed.")
        return None, None, None

    subset = ticker_df.copy()
    subset["headline_signal"] = np.where(
        (subset["smooth_sentiment"] >= finbert_score) & (subset["close"] > subset["SMA_50"]),
        1, 0,
    )
    signal_zones = subset[subset["headline_signal"] == 1]

    # Zone entries: first day of each contiguous signal run
    entries = subset[
        (subset["headline_signal"] == 1)
        & (subset["headline_signal"].shift(1, fill_value=0) == 0)
    ]

    fwd_returns = []
    for idx in entries.index:
        loc = subset.index.get_loc(idx)
        for offset in [1, 5, 10]:
            target = loc + offset
            if target < len(subset):
                ret = (subset["close"].iloc[target] - subset["close"].iloc[loc]) \
                    / subset["close"].iloc[loc] * 100
                fwd_returns.append({"offset": offset, "return": ret})

    summary = {
        "finbert_score": finbert_score,
        "num_signals": len(entries),
        "avg_1d": None,
        "avg_5d": None,
        "avg_10d": None,
    }
    for offset, label in [(1, "avg_1d"), (5, "avg_5d"), (10, "avg_10d")]:
        vals = [r["return"] for r in fwd_returns if r["offset"] == offset]
        if vals:
            summary[label] = sum(vals) / len(vals)

    return finbert_score, signal_zones, summary


def print_signal_summary(summary):
    print(f"\n  ── FINBERT HEADLINE OVERLAY ──")
    print(f"  FinBERT Score:    {summary['finbert_score']:+.4f}")
    print(f"  Signal Entries:  {summary['num_signals']}")

    for label, key in [("Avg 1D Return", "avg_1d"), ("Avg 5D Return", "avg_5d"), ("Avg 10D Return", "avg_10d")]:
        val = summary[key]
        if val is not None:
            print(f"  {label}:  {val:+.2f}%")
        else:
            print(f"  {label}:  —")
    print()


# ── Plotting ──────────────────────────────────────────────────────────────

def plot_ticker_forensics(ticker, df, headline_overlay=None):
    subset = df.copy()

    finbert_score, signal_zones, summary = None, None, None
    if headline_overlay:
        finbert_score, signal_zones, summary = analyze_headline_overlay(
            ticker, subset, headline_overlay
        )
        if summary:
            print_signal_summary(summary)

    # Trades at execution day: position moves off/onto zero
    pos_active = (subset["position"] > 0).astype(int)
    trade_action = pos_active.diff().fillna(pos_active.iloc[0] if len(pos_active) else 0)
    buys = subset[trade_action == 1]
    sells = subset[trade_action == -1]

    fig, (ax1, ax_eq, ax2) = plt.subplots(
        3, 1, figsize=(14, 12),
        gridspec_kw={"height_ratios": [3, 2, 1]}, sharex=True,
    )

    # --- TOP PANEL: PRICE & TRADES ---
    ax1.plot(subset.index, subset["close"], label="Price", color="black", alpha=0.6, linewidth=1)
    ax1.plot(subset.index, subset["SMA_50"], label="50-Day Trend", color="blue", linestyle="--", alpha=0.4)

    # Shade signal zones from headline overlay
    if signal_zones is not None and not signal_zones.empty:
        in_zone = False
        zone_start = None
        first_zone = True
        for ts, row in subset.iterrows():
            if ts in signal_zones.index:
                if not in_zone:
                    zone_start = ts
                    in_zone = True
            else:
                if in_zone:
                    ax1.axvspan(zone_start, ts, alpha=0.15, color="#00E5FF",
                                label="FinBERT Signal Zone" if first_zone else "_")
                    first_zone = False
                    in_zone = False
        if in_zone:
            ax1.axvspan(zone_start, subset.index[-1], alpha=0.15, color="#00E5FF",
                        label="FinBERT Signal Zone" if first_zone else "_")

    cmap = plt.cm.RdYlGn
    norm = Normalize(vmin=-1, vmax=1)

    if not buys.empty:
        ax1.scatter(
            buys.index, buys["close"],
            c=buys["smooth_sentiment"].fillna(0), cmap=cmap, norm=norm,
            marker="^", s=150, edgecolors="black", zorder=5, label="Buy (next-day fill)",
        )
    if not sells.empty:
        ax1.scatter(
            sells.index, sells["close"],
            c=sells["smooth_sentiment"].fillna(0), cmap=cmap, norm=norm,
            marker="v", s=150, edgecolors="black", zorder=5, label="Sell/Exit",
        )

    ax1.set_title(f"{ticker}: Trade Execution vs. Sentiment Score", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Stock Price ($)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    cbar_ax = fig.add_axes([0.92, 0.55, 0.02, 0.3])
    cb = plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
    cb.set_label("Sentiment Score (-1 to +1)")

    # --- MIDDLE PANEL: EQUITY CURVE ---
    ax_eq.plot(subset.index, subset["equity"], label="Strategy", color="#008855", linewidth=1.5)
    ax_eq.plot(subset.index, subset["buyhold_equity"], label="Buy & Hold", color="gray", alpha=0.7, linewidth=1)
    ax_eq.axhline(y=INITIAL_CAPITAL, color="gray", linewidth=0.5, linestyle=":")
    ax_eq.set_yscale("log")
    ax_eq.set_ylabel("Equity ($, log)")
    ax_eq.legend(loc="upper left")
    ax_eq.grid(True, alpha=0.3)

    # --- BOTTOM PANEL: SENTIMENT INDICATOR ---
    ax2.plot(subset.index, subset["smooth_sentiment"], color="purple", label="3-Day Sentiment", linewidth=1.5)
    ax2.axhline(y=SENTIMENT_THRESHOLD, color="green", linestyle=":", label=f"Buy Threshold ({SENTIMENT_THRESHOLD})")
    ax2.axhline(y=0, color="gray", linewidth=0.5)

    if finbert_score is not None:
        ax2.axhline(y=finbert_score, color="#00E5FF", linewidth=2, linestyle="--",
                    label=f"FinBERT Score ({finbert_score:+.3f})")

    smooth = subset["smooth_sentiment"].fillna(0)
    ax2.fill_between(subset.index, smooth, 0, where=(smooth > 0), color="green", alpha=0.1)
    ax2.fill_between(subset.index, smooth, 0, where=(smooth < 0), color="red", alpha=0.1)

    ax2.set_ylabel("Sentiment")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    plt.show()


# ── Main loop ─────────────────────────────────────────────────────────────

def backtest_ticker(ticker: str, headline_overlay=None, show_plot=True):
    """Full pipeline for one ticker. Returns (df, metrics) or (None, None)."""
    print(f"Loading sentiment + prices for {ticker}...")
    df = build_frame(ticker)
    if df is None or df["daily_score"].notna().sum() == 0:
        print(f"No data found for {ticker}")
        return None, None

    df = run_backtest(df)
    metrics = compute_metrics(df)
    print_metrics(ticker, metrics)

    if show_plot:
        plot_ticker_forensics(ticker, df, headline_overlay=headline_overlay)
    return df, metrics


def main():
    print("--- STARK SENTIMENT-MOMENTUM BACKTESTER ---")
    print("Signals execute next-day; costs and vol-targeted sizing included.")
    print("Type a ticker to backtest it (e.g., 'NVDA', 'TSLA', 'AAPL').")
    print("Type 'exit' to quit.")

    while True:
        user_input = input("\nEnter Ticker: ").upper().strip()
        if user_input in ["EXIT", "QUIT", ""]:
            break
        try:
            df, metrics = backtest_ticker(user_input)
            if df is None:
                continue

            headline = input("Headline overlay (or enter to skip): ").strip()
            if headline:
                plot_ticker_forensics(user_input, df, headline_overlay=headline)
        except Exception as e:
            print(f"Error backtesting {user_input}: {e}")


if __name__ == "__main__":
    main()
