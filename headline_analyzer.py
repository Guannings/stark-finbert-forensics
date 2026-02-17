#!/usr/bin/env python3
"""
Headline Sentiment Analyzer — Dataset-Powered
Type a news headline + ticker → get a verdict grounded in 85M+ historical headlines.
"""

import sys
import os
import re
import argparse
from datetime import datetime, timedelta

import numpy as np
import duckdb
import pandas as pd
import yfinance as yf
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

# ── Styling ──────────────────────────────────────────────────────────────────
NEON_GREEN = "#00FF87"
NEON_RED = "#FF4455"
NEON_CYAN = "#00E5FF"
NEON_YELLOW = "#FFD700"
DIM_TEXT = "#666666"
TEXT_COLOR = "#E0E0E0"

console = Console()

PARQUET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "STARK_SCORED_FIXED.parquet")
INDEX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "headline_index.parquet")

# ── FinBERT lazy singleton ───────────────────────────────────────────────────
_finbert_cache = {}


def get_finbert():
    """Lazy-load FinBERT model (singleton). Imports torch/transformers on first call."""
    if "model" not in _finbert_cache:
        import torch
        from transformers import BertTokenizer, BertForSequenceClassification

        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
        model = BertForSequenceClassification.from_pretrained("ProsusAI/finbert")
        model.to(device)
        model.eval()
        _finbert_cache["tokenizer"] = tokenizer
        _finbert_cache["model"] = model
        _finbert_cache["device"] = device

    return _finbert_cache["tokenizer"], _finbert_cache["model"], _finbert_cache["device"]


def score_headline_live(headline: str) -> float | None:
    """Score a single headline with FinBERT. Returns float in [-1, 1] or None on error."""
    try:
        import torch

        tokenizer, model, device = get_finbert()
        inputs = tokenizer(
            [headline], return_tensors="pt", padding=True, truncation=True, max_length=64
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            score = (probs[0, 0] - probs[0, 1]).item()
        return score
    except Exception:
        return None


STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need", "must",
    "it", "its", "this", "that", "these", "those", "i", "we", "you", "he",
    "she", "they", "me", "him", "her", "us", "them", "my", "your", "his",
    "our", "their", "what", "which", "who", "whom", "where", "when", "how",
    "not", "no", "nor", "as", "if", "then", "than", "too", "very", "just",
    "about", "above", "after", "again", "all", "also", "am", "any", "because",
    "before", "between", "both", "each", "few", "more", "most", "other",
    "over", "own", "same", "so", "some", "such", "up", "down", "out", "off",
    "into", "through", "during", "here", "there", "only", "once", "s", "t",
    "says", "said", "new", "vs", "per", "via",
}


def extract_keywords(headline: str) -> list[str]:
    """Lowercase, strip non-alpha, remove stopwords, dedupe preserving order."""
    tokens = re.findall(r"[a-z0-9]+", headline.lower())
    seen = set()
    keywords = []
    for t in tokens:
        if t not in STOPWORDS and t not in seen and len(t) > 1:
            seen.add(t)
            keywords.append(t)
    return keywords


def parse_time_window(window: str | None, since: str | None, until: str | None) -> tuple[datetime | None, datetime | None]:
    """Convert --window / --since / --until into (date_from, date_to) datetimes."""
    date_from, date_to = None, None

    if since:
        date_from = datetime.strptime(since, "%Y-%m-%d")
    if until:
        date_to = datetime.strptime(until, "%Y-%m-%d")

    if window:
        now = datetime.now()
        mapping = {"1w": 7, "1m": 30, "3m": 90, "6m": 180, "1y": 365}
        days = mapping.get(window.lower())
        if days:
            date_from = now - timedelta(days=days)
            if date_to is None:
                date_to = now

    return date_from, date_to


def _query_indexed(ticker: str, keywords: list[str], top_n: int,
                   date_from: datetime | None = None, date_to: datetime | None = None) -> pd.DataFrame:
    """Query deduped parquet index (sorted by ticker for fast row-group pruning)."""
    con = duckdb.connect()
    kw_list = ", ".join(f"'{k}'" for k in keywords)

    date_clauses = ""
    params = [INDEX_PATH, ticker.upper(), top_n]
    if date_from is not None:
        date_clauses += f" AND date >= ${len(params) + 1}"
        params.append(date_from.strftime("%Y-%m-%d"))
    if date_to is not None:
        date_clauses += f" AND date <= ${len(params) + 1}"
        params.append(date_to.strftime("%Y-%m-%d"))

    query = f"""
    WITH ticker_rows AS (
        SELECT
            headline,
            date,
            sentiment_score,
            regexp_split_to_array(lower(headline), '[^a-z0-9]+') AS words
        FROM read_parquet($1)
        WHERE ticker = $2{date_clauses}
    ),
    scored AS (
        SELECT
            headline,
            date,
            sentiment_score,
            len(list_intersect(words, [{kw_list}])) AS shared_count,
            CASE
                WHEN len(list_distinct(list_concat(words, [{kw_list}]))) = 0 THEN 0
                ELSE len(list_intersect(words, [{kw_list}]))::DOUBLE
                     / len(list_distinct(list_concat(words, [{kw_list}])))::DOUBLE
            END AS jaccard
        FROM ticker_rows
        WHERE len(list_intersect(words, [{kw_list}])) >= 2
    )
    SELECT headline, date, sentiment_score, shared_count, jaccard
    FROM scored
    ORDER BY shared_count DESC, jaccard DESC
    LIMIT $3
    """

    try:
        df = con.execute(query, params).fetchdf()
    except Exception as e:
        console.print(f"[{NEON_RED}]DuckDB error: {e}[/]")
        return pd.DataFrame()
    finally:
        con.close()

    return df


def _query_parquet(ticker: str, keywords: list[str], top_n: int,
                   date_from: datetime | None = None, date_to: datetime | None = None) -> pd.DataFrame:
    """Fallback: query raw parquet (slow, high memory)."""
    con = duckdb.connect()
    con.execute("SET memory_limit='12GB'")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET threads=4")

    kw_list = ", ".join(f"'{k}'" for k in keywords)

    date_clauses = ""
    params = [PARQUET_PATH, ticker.upper(), top_n]
    if date_from is not None:
        date_clauses += f" AND date >= ${len(params) + 1}"
        params.append(date_from.strftime("%Y-%m-%d"))
    if date_to is not None:
        date_clauses += f" AND date <= ${len(params) + 1}"
        params.append(date_to.strftime("%Y-%m-%d"))

    query = f"""
    WITH input_words AS (
        SELECT [{kw_list}] AS kw
    ),
    ticker_headlines AS (
        SELECT
            date,
            headline,
            sentiment_score,
            regexp_split_to_array(lower(headline), '[^a-z0-9]+') AS words
        FROM read_parquet($1)
        WHERE ticker = $2{date_clauses}
    ),
    deduped AS (
        SELECT
            headline,
            MIN(date) AS first_date,
            AVG(sentiment_score) AS avg_score,
            words
        FROM ticker_headlines
        GROUP BY headline, words
    ),
    scored AS (
        SELECT
            d.headline,
            d.first_date AS date,
            d.avg_score AS sentiment_score,
            len(list_intersect(d.words, i.kw)) AS shared_count,
            CASE
                WHEN len(list_distinct(list_concat(d.words, i.kw))) = 0 THEN 0
                ELSE len(list_intersect(d.words, i.kw))::DOUBLE
                     / len(list_distinct(list_concat(d.words, i.kw)))::DOUBLE
            END AS jaccard
        FROM deduped d, input_words i
        WHERE len(list_intersect(d.words, i.kw)) >= 2
    )
    SELECT headline, date, sentiment_score, shared_count, jaccard
    FROM scored
    ORDER BY shared_count DESC, jaccard DESC
    LIMIT $3
    """

    try:
        df = con.execute(query, params).fetchdf()
    except Exception as e:
        console.print(f"[{NEON_RED}]DuckDB error: {e}[/]")
        return pd.DataFrame()
    finally:
        con.close()

    return df


def find_similar_headlines(ticker: str, keywords: list[str], top_n: int = 20,
                           date_from: datetime | None = None, date_to: datetime | None = None) -> pd.DataFrame:
    """DuckDB word-overlap query with jaccard tiebreaker. Returns top matches."""
    if not keywords:
        return pd.DataFrame()

    if os.path.exists(INDEX_PATH):
        df = _query_indexed(ticker, keywords, top_n, date_from, date_to)
    else:
        console.print(f"[{NEON_YELLOW}]Index not found — falling back to raw parquet (slow). "
                       f"Run 'python build_index.py' to build it.[/]")
        df = _query_parquet(ticker, keywords, top_n, date_from, date_to)

    if not df.empty:
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()

    return df


def compute_forward_returns(ticker: str, dates: pd.Series) -> dict:
    """Single yfinance bulk download, then index-based forward return lookups."""
    if dates.empty:
        return {}

    min_date = dates.min() - timedelta(days=5)
    max_date = dates.max() + timedelta(days=20)
    today = datetime.now()
    if max_date > today:
        max_date = today

    try:
        prices = yf.download(
            ticker.upper(),
            start=min_date.strftime("%Y-%m-%d"),
            end=max_date.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
        )
    except Exception:
        return {}

    if prices.empty:
        return {}

    if isinstance(prices.columns, pd.MultiIndex):
        prices = prices.xs(ticker.upper(), axis=1, level=1)

    prices.index = prices.index.tz_localize(None).normalize()
    close = prices["Close"].sort_index()

    returns = {}
    for d in dates:
        d_norm = pd.Timestamp(d).normalize()
        # Find nearest trading day on or after the headline date
        future_dates = close.index[close.index >= d_norm]
        if len(future_dates) == 0:
            continue
        base_idx = close.index.get_loc(future_dates[0])
        base_price = close.iloc[base_idx]

        row = {}
        for label, offset in [("1d", 1), ("5d", 5), ("10d", 10)]:
            target_idx = base_idx + offset
            if target_idx < len(close):
                row[label] = ((close.iloc[target_idx] - base_price) / base_price) * 100
            else:
                row[label] = None
        returns[d_norm] = row

    return returns


def compute_verdict(matches: pd.DataFrame, returns: dict,
                    recency_half_life_days: int = 180) -> dict:
    """Sentiment distribution + price outcomes → verdict dict."""
    result = {
        "verdict": "NEUTRAL",
        "confidence": 0,
        "avg_sentiment": 0.0,
        "pct_bullish": 0.0,
        "pct_bearish": 0.0,
        "avg_1d": None,
        "avg_5d": None,
        "avg_10d": None,
        "recency_weighted": False,
    }

    if matches.empty:
        return result

    scores = matches["sentiment_score"]

    # Recency weighting via exponential decay
    if "date" in matches.columns and matches["date"].notna().any():
        now = pd.Timestamp.now().normalize()
        days_ago = (now - pd.to_datetime(matches["date"])).dt.total_seconds() / 86400
        days_ago = days_ago.clip(lower=0).values
        decay = np.log(2) / recency_half_life_days
        weights = np.exp(-decay * days_ago)
        result["avg_sentiment"] = float(np.average(scores.values, weights=weights))
        result["recency_weighted"] = True
    else:
        result["avg_sentiment"] = scores.mean()
    result["pct_bullish"] = (scores > 0.1).mean() * 100
    result["pct_bearish"] = (scores < -0.1).mean() * 100

    # Sentiment signal: scale from -1..+1 to -100..+100
    sent_signal = max(-1, min(1, result["avg_sentiment"])) * 100

    # Price signal from forward returns
    price_signal = 0.0
    price_count = 0
    for period in ["1d", "5d", "10d"]:
        vals = [r[period] for r in returns.values() if r.get(period) is not None]
        if vals:
            avg = sum(vals) / len(vals)
            result[f"avg_{period}"] = avg
            price_signal += avg
            price_count += 1

    if price_count > 0:
        price_signal = max(-100, min(100, (price_signal / price_count) * 20))

    # Weighted composite: 40% sentiment, 60% price action
    if price_count > 0:
        composite = 0.4 * sent_signal + 0.6 * price_signal
    else:
        composite = sent_signal

    abs_composite = abs(composite)
    result["confidence"] = int(min(99, max(10, 50 + abs_composite * 0.5)))

    if composite > 10:
        result["verdict"] = "BULLISH"
    elif composite < -10:
        result["verdict"] = "BEARISH"
    else:
        result["verdict"] = "NEUTRAL"

    return result


def render_output(ticker: str, headline: str, keywords: list[str],
                  matches: pd.DataFrame, returns: dict, verdict: dict,
                  live_score: float | None = None):
    """Rich panels + tables with neon styling."""
    console.print()

    # ── Verdict banner ────────────────────────────────────────────────────
    v = verdict["verdict"]
    conf = verdict["confidence"]
    if v == "BULLISH":
        color = NEON_GREEN
        icon = "▲"
    elif v == "BEARISH":
        color = NEON_RED
        icon = "▼"
    else:
        color = NEON_YELLOW
        icon = "●"

    recency_tag = "  [recency-weighted]" if verdict.get("recency_weighted") else ""

    verdict_text = Text()
    verdict_text.append(f"  {icon} HISTORICALLY {v}", style=f"bold {color}")
    verdict_text.append(f"   Confidence: {conf}%", style=TEXT_COLOR)
    if recency_tag:
        verdict_text.append(recency_tag, style=DIM_TEXT)

    console.print(Panel(
        verdict_text,
        title=f"[bold {NEON_CYAN}]{ticker.upper()}[/] VERDICT",
        border_style=color,
        padding=(0, 2),
    ))

    # ── Keywords used ─────────────────────────────────────────────────────
    kw_str = ", ".join(keywords) if keywords else "(none)"
    console.print(f"  [{DIM_TEXT}]Keywords: {kw_str}[/]")
    console.print(f"  [{DIM_TEXT}]Matches found: {len(matches)}[/]")
    console.print()

    # ── FinBERT live score panel ───────────────────────────────────────────
    if live_score is not None:
        hist_avg = verdict["avg_sentiment"]
        delta = live_score - hist_avg

        ls_color = NEON_GREEN if live_score > 0.05 else NEON_RED if live_score < -0.05 else TEXT_COLOR
        ha_color = NEON_GREEN if hist_avg > 0.05 else NEON_RED if hist_avg < -0.05 else TEXT_COLOR
        d_color = NEON_GREEN if delta > 0 else NEON_RED if delta < 0 else TEXT_COLOR

        fb_table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style=f"bold {NEON_CYAN}")
        fb_table.add_column("Input Score", justify="center")
        fb_table.add_column("Historical Avg", justify="center")
        fb_table.add_column("Delta", justify="center")
        fb_table.add_row(
            f"[{ls_color}]{live_score:+.4f}[/]",
            f"[{ha_color}]{hist_avg:+.4f}[/]",
            f"[{d_color}]{delta:+.4f}[/]",
        )

        console.print(Panel(fb_table, title=f"[bold {NEON_CYAN}]FINBERT LIVE SCORE[/]",
                            border_style=NEON_CYAN, padding=(0, 1)))

    # ── Metrics table ─────────────────────────────────────────────────────
    metrics = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style=f"bold {NEON_CYAN}")
    metrics.add_column("Avg Sentiment", justify="center")
    metrics.add_column("% Bullish", justify="center")
    metrics.add_column("% Bearish", justify="center")
    metrics.add_column("Avg 1D", justify="center")
    metrics.add_column("Avg 5D", justify="center")
    metrics.add_column("Avg 10D", justify="center")

    def fmt_score(val):
        if val is None:
            return f"[{DIM_TEXT}]—[/]"
        color = NEON_GREEN if val > 0 else NEON_RED if val < 0 else TEXT_COLOR
        return f"[{color}]{val:+.3f}[/]"

    def fmt_pct(val):
        if val is None:
            return f"[{DIM_TEXT}]—[/]"
        color = NEON_GREEN if val > 0 else NEON_RED if val < 0 else TEXT_COLOR
        return f"[{color}]{val:+.2f}%[/]"

    metrics.add_row(
        fmt_score(verdict["avg_sentiment"]),
        f"[{NEON_GREEN}]{verdict['pct_bullish']:.0f}%[/]",
        f"[{NEON_RED}]{verdict['pct_bearish']:.0f}%[/]",
        fmt_pct(verdict["avg_1d"]),
        fmt_pct(verdict["avg_5d"]),
        fmt_pct(verdict["avg_10d"]),
    )

    console.print(Panel(metrics, title=f"[bold {NEON_CYAN}]METRICS[/]", border_style=DIM_TEXT, padding=(0, 1)))

    # ── Top matches table ─────────────────────────────────────────────────
    if matches.empty:
        console.print(f"  [{NEON_YELLOW}]No similar headlines found for {ticker.upper()}.[/]")
        return

    # Compute available width for headline: total - fixed cols - panel borders
    term_width = console.width or 80
    # Fixed cols: # (3) + Date (10) + Sent (6) + 1D (7) + 5D (7) + 10D (7) + separators (~12) + panel padding (~6)
    hl_width = max(20, term_width - 62)

    tbl = Table(box=box.SIMPLE, show_header=True, header_style=f"bold {NEON_CYAN}",
                row_styles=[TEXT_COLOR, DIM_TEXT], expand=False)
    tbl.add_column("#", justify="right", width=3, no_wrap=True)
    tbl.add_column("Date", justify="center", width=10, no_wrap=True)
    tbl.add_column("Headline", width=hl_width, no_wrap=True, overflow="ellipsis")
    tbl.add_column("Sent", justify="center", width=6, no_wrap=True)
    tbl.add_column("1D", justify="right", width=7, no_wrap=True)
    tbl.add_column("5D", justify="right", width=7, no_wrap=True)
    tbl.add_column("10D", justify="right", width=7, no_wrap=True)

    def fmt_ret(val):
        if val is None:
            return f"[{DIM_TEXT}]—[/]"
        c = NEON_GREEN if val > 0 else NEON_RED if val < 0 else TEXT_COLOR
        return f"[{c}]{val:+.2f}%[/]"

    for i, row in matches.iterrows():
        date_val = row["date"]
        date_str = pd.Timestamp(date_val).strftime("%Y-%m-%d") if pd.notna(date_val) else "—"

        hl = str(row["headline"]).replace("\n", " ").replace("\r", " ").strip()

        sent = row["sentiment_score"]
        sent_color = NEON_GREEN if sent > 0.1 else NEON_RED if sent < -0.1 else TEXT_COLOR
        sent_str = f"[{sent_color}]{sent:+.2f}[/]"

        d_norm = pd.Timestamp(date_val).normalize()
        ret = returns.get(d_norm, {})

        tbl.add_row(
            str(i + 1),
            date_str,
            hl,
            sent_str,
            fmt_ret(ret.get("1d")),
            fmt_ret(ret.get("5d")),
            fmt_ret(ret.get("10d")),
        )

    console.print(Panel(tbl, title=f"[bold {NEON_CYAN}]TOP SIMILAR HEADLINES[/]",
                        border_style=DIM_TEXT, padding=(0, 0)))
    console.print()


def analyze(ticker: str, headline: str, top_n: int = 20,
            date_from: datetime | None = None, date_to: datetime | None = None):
    """Run full analysis pipeline for a ticker + headline."""
    ticker = ticker.strip().upper()
    headline = headline.strip()

    if len(headline) < 5:
        console.print(f"[{NEON_RED}]Headline too short. Please enter a meaningful headline.[/]")
        return

    if not os.path.exists(INDEX_PATH) and not os.path.exists(PARQUET_PATH):
        console.print(f"[{NEON_RED}]No data source found. Need either headline_index.parquet or {PARQUET_PATH}[/]")
        return

    keywords = extract_keywords(headline)
    if len(keywords) < 2:
        console.print(f"[{NEON_YELLOW}]Only {len(keywords)} keyword(s) extracted — need at least 2 for matching.[/]")
        console.print(f"[{DIM_TEXT}]Try a longer or more specific headline.[/]")
        return

    # FinBERT live scoring
    console.print(f"  [{NEON_CYAN}]FinBERT scoring...[/]", end="")
    live_score = score_headline_live(headline)
    if live_score is not None:
        sc = NEON_GREEN if live_score > 0.05 else NEON_RED if live_score < -0.05 else TEXT_COLOR
        console.print(f" [{sc}]{live_score:+.4f}[/]")
    else:
        console.print(f" [{NEON_YELLOW}]unavailable[/]")

    window_label = ""
    if date_from:
        window_label += f" from {date_from.strftime('%Y-%m-%d')}"
    if date_to:
        window_label += f" to {date_to.strftime('%Y-%m-%d')}"

    console.print(f"  [{NEON_CYAN}]Searching {ticker} headlines{window_label}...[/]", end="")
    matches = find_similar_headlines(ticker, keywords, top_n=top_n,
                                     date_from=date_from, date_to=date_to)
    console.print(f" [{NEON_GREEN}]{len(matches)} matches[/]")

    if matches.empty:
        verdict = compute_verdict(matches, {})
        render_output(ticker, headline, keywords, matches, {}, verdict, live_score)
        return

    console.print(f"  [{NEON_CYAN}]Fetching price data...[/]", end="")
    returns = compute_forward_returns(ticker, matches["date"])
    console.print(f" [{NEON_GREEN}]done[/]")

    verdict = compute_verdict(matches, returns)
    render_output(ticker, headline, keywords, matches, returns, verdict, live_score)


def analyze_multi(tickers: list[str], headline: str, top_n: int = 20,
                  date_from: datetime | None = None, date_to: datetime | None = None):
    """Run analysis across multiple tickers and render a comparison table."""
    headline = headline.strip()
    if len(headline) < 5:
        console.print(f"[{NEON_RED}]Headline too short. Please enter a meaningful headline.[/]")
        return

    keywords = extract_keywords(headline)
    if len(keywords) < 2:
        console.print(f"[{NEON_YELLOW}]Only {len(keywords)} keyword(s) extracted — need at least 2 for matching.[/]")
        return

    # Score headline once (ticker-agnostic)
    console.print(f"  [{NEON_CYAN}]FinBERT scoring...[/]", end="")
    live_score = score_headline_live(headline)
    if live_score is not None:
        sc = NEON_GREEN if live_score > 0.05 else NEON_RED if live_score < -0.05 else TEXT_COLOR
        console.print(f" [{sc}]{live_score:+.4f}[/]")
    else:
        console.print(f" [{NEON_YELLOW}]unavailable[/]")

    results = []
    for t in tickers:
        t = t.strip().upper()
        console.print(f"  [{NEON_CYAN}]Analyzing {t}...[/]", end="")
        matches = find_similar_headlines(t, keywords, top_n=top_n,
                                         date_from=date_from, date_to=date_to)
        if matches.empty:
            verdict = compute_verdict(matches, {})
            results.append({"ticker": t, "matches": matches, "returns": {}, "verdict": verdict})
            console.print(f" [{NEON_YELLOW}]0 matches[/]")
            continue

        returns = compute_forward_returns(t, matches["date"])
        verdict = compute_verdict(matches, returns)
        results.append({"ticker": t, "matches": matches, "returns": returns, "verdict": verdict})
        console.print(f" [{NEON_GREEN}]{len(matches)} matches[/]")

    # Render individual results
    for r in results:
        render_output(r["ticker"], headline, keywords, r["matches"], r["returns"], r["verdict"], live_score)

    # Render comparison summary
    render_comparison(results, live_score)


def render_comparison(results: list[dict], live_score: float | None):
    """Render a side-by-side comparison table across tickers."""
    console.print()
    tbl = Table(
        title=f"[bold {NEON_CYAN}]MULTI-TICKER COMPARISON[/]",
        box=box.HEAVY_HEAD,
        show_header=True,
        header_style=f"bold {NEON_CYAN}",
    )
    tbl.add_column("Ticker", justify="center", style=f"bold {NEON_CYAN}")
    tbl.add_column("Verdict", justify="center")
    tbl.add_column("Conf", justify="center")
    tbl.add_column("Avg Sent", justify="center")
    tbl.add_column("Matches", justify="center")
    tbl.add_column("Avg 1D", justify="right")
    tbl.add_column("Avg 5D", justify="right")
    tbl.add_column("Avg 10D", justify="right")

    def fmt_ret(val):
        if val is None:
            return f"[{DIM_TEXT}]—[/]"
        c = NEON_GREEN if val > 0 else NEON_RED if val < 0 else TEXT_COLOR
        return f"[{c}]{val:+.2f}%[/]"

    for r in results:
        v = r["verdict"]
        vc = NEON_GREEN if v["verdict"] == "BULLISH" else NEON_RED if v["verdict"] == "BEARISH" else NEON_YELLOW
        sc = NEON_GREEN if v["avg_sentiment"] > 0.05 else NEON_RED if v["avg_sentiment"] < -0.05 else TEXT_COLOR

        tbl.add_row(
            r["ticker"],
            f"[bold {vc}]{v['verdict']}[/]",
            f"{v['confidence']}%",
            f"[{sc}]{v['avg_sentiment']:+.4f}[/]",
            str(len(r["matches"])),
            fmt_ret(v["avg_1d"]),
            fmt_ret(v["avg_5d"]),
            fmt_ret(v["avg_10d"]),
        )

    if live_score is not None:
        lsc = NEON_GREEN if live_score > 0.05 else NEON_RED if live_score < -0.05 else TEXT_COLOR
        console.print(f"  [{DIM_TEXT}]FinBERT live score:[/] [{lsc}]{live_score:+.4f}[/]")

    console.print(tbl)
    console.print()


def repl():
    """Interactive REPL mode."""
    console.print(Panel(
        f"[bold {NEON_CYAN}]HEADLINE SENTIMENT ANALYZER[/]\n"
        f"[{TEXT_COLOR}]Powered by 85M+ scored headlines (2009–2026)[/]\n"
        f"[{DIM_TEXT}]Type 'quit' to exit  |  Comma-separate tickers for comparison[/]",
        border_style=NEON_CYAN,
        padding=(1, 2),
    ))

    while True:
        console.print()
        try:
            ticker_raw = console.input(f"[bold {NEON_CYAN}]Ticker(s):[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if ticker_raw.lower() in ("quit", "exit", "q"):
            break
        if not ticker_raw:
            continue

        try:
            headline = console.input(f"[bold {NEON_CYAN}]Headline:[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if headline.lower() in ("quit", "exit", "q"):
            break
        if not headline:
            continue

        # Optional time window
        try:
            window_raw = console.input(f"[{DIM_TEXT}]Window (1w/1m/3m/6m/1y or enter to skip):[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        date_from, date_to = parse_time_window(window_raw or None, None, None)

        tickers = [t.strip().upper() for t in ticker_raw.split(",") if t.strip()]

        if len(tickers) > 1:
            analyze_multi(tickers, headline, top_n=20, date_from=date_from, date_to=date_to)
        else:
            analyze(tickers[0], headline, top_n=20, date_from=date_from, date_to=date_to)


def main():
    parser = argparse.ArgumentParser(
        description="Headline Sentiment Analyzer — Dataset-Powered",
        usage="%(prog)s [TICKER] [HEADLINE]",
    )
    parser.add_argument("ticker", nargs="?", help="Stock ticker(s), comma-separated (e.g. AAPL,TSLA)")
    parser.add_argument("headline", nargs="?", help="News headline to analyze")
    parser.add_argument("-n", "--top-n", type=int, default=20, help="Number of matches (default: 20)")
    parser.add_argument("--window", type=str, default=None, help="Time window: 1w, 1m, 3m, 6m, 1y")
    parser.add_argument("--since", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--until", type=str, default=None, help="End date (YYYY-MM-DD)")

    args = parser.parse_args()

    if args.ticker and args.headline:
        date_from, date_to = parse_time_window(args.window, args.since, getattr(args, "until"))

        tickers = [t.strip().upper() for t in args.ticker.split(",") if t.strip()]
        if len(tickers) > 1:
            analyze_multi(tickers, args.headline, top_n=args.top_n,
                          date_from=date_from, date_to=date_to)
        else:
            analyze(tickers[0], args.headline, top_n=args.top_n,
                    date_from=date_from, date_to=date_to)
    else:
        repl()


if __name__ == "__main__":
    main()
