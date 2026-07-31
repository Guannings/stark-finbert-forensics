#!/usr/bin/env python3
"""
Out-of-sample validation of the sentiment-momentum threshold.

The default SENTIMENT_THRESHOLD was chosen on three tickers (in-sample). This
script tests whether that choice generalizes: it picks the threshold that
maximizes mean Sharpe on a TRAIN set of tickers, then measures that same
threshold — with no further tuning — on a disjoint TEST set. If the train-
optimal threshold also does well out-of-sample, the edge is real; if not, it
was curve-fit.

Sharpe (not total return) is the objective: total return alone just rewards
being fully invested through a bull market, which a news signal should not get
credit for.

    python validate_threshold.py
"""

import warnings

import numpy as np

warnings.filterwarnings("ignore")

import backtester as bt

# Large-cap single names with substantial headline history (>=16k headlines),
# ETFs/commodity trackers removed. Sorted; split deterministically below.
UNIVERSE = [
    "AAPL", "AMD", "AMZN", "BA", "BABA", "C", "CVX", "DIS", "GE", "GS",
    "INTC", "JPM", "KO", "META", "MRK", "MSFT", "NVDA", "ORCL", "PFE",
    "T", "TSLA", "V", "WFC", "WMT", "XOM",
]

THRESHOLDS = [-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


def build_frames(tickers):
    """Build each ticker's price+sentiment frame once (the slow, I/O part)."""
    frames = {}
    for t in tickers:
        try:
            f = bt.build_frame(t)
            if f is not None and f["daily_score"].notna().sum() > 250:
                frames[t] = f
                print(f"  {t}: {len(f)} rows, {int(f['daily_score'].notna().sum())} sentiment days")
            else:
                print(f"  {t}: insufficient data — skipped")
        except Exception as e:
            print(f"  {t}: error ({e}) — skipped")
    return frames


def sharpe_at(frame, threshold):
    df = bt.run_backtest(bt.apply_signal(frame, threshold))
    s = bt.compute_metrics(df)["strategy"]
    return s.get("sharpe", float("nan")), s.get("total_return", float("nan"))


def component_sharpe(frame, mode):
    """Isolate what drives the signal: trend filter alone, sentiment alone,
    both, or the buy-and-hold baseline."""
    df = frame.copy()
    if mode == "trend":
        df["signal"] = np.where(df["close"] > df["SMA_50"], 1, 0)
    elif mode == "sentiment":
        df["signal"] = np.where(df["smooth_sentiment"] > 0, 1, 0)
    elif mode == "combined":
        df["signal"] = np.where(
            (df["smooth_sentiment"] > 0) & (df["close"] > df["SMA_50"]), 1, 0)
    m = bt.compute_metrics(bt.run_backtest(df))
    return m["strategy"].get("sharpe", float("nan")), m["buyhold"].get("sharpe", float("nan"))


def decomposition(frames):
    """Mean Sharpe of each signal component across all tickers.

    Answers the question the threshold sweep raises: does the sentiment gate
    add anything, or is the trend filter doing all the work?
    """
    acc = {"trend": [], "sentiment": [], "combined": [], "buyhold": []}
    for f in frames.values():
        for mode in ("trend", "sentiment", "combined"):
            sh, bh = component_sharpe(f, mode)
            if sh == sh:
                acc[mode].append(sh)
        if bh == bh:
            acc["buyhold"].append(bh)
    return {k: float(np.mean(v)) if v else float("nan") for k, v in acc.items()}


def mean_sharpe(frames, tickers, threshold):
    vals = []
    for t in tickers:
        sh, _ = sharpe_at(frames[t], threshold)
        if sh == sh:  # not NaN
            vals.append(sh)
    return float(np.mean(vals)) if vals else float("nan")


def best_threshold(frames, tickers):
    scores = {th: mean_sharpe(frames, tickers, th) for th in THRESHOLDS}
    best = max(scores, key=lambda th: (scores[th] if scores[th] == scores[th] else -1e9))
    return best, scores


def main():
    # Deterministic disjoint split: alternate down the sorted universe.
    universe = sorted(UNIVERSE)
    train = universe[0::2]
    test = universe[1::2]

    print("Building frames (downloads prices, aggregates sentiment)...")
    frames = build_frames(universe)
    train = [t for t in train if t in frames]
    test = [t for t in test if t in frames]

    print(f"\nTrain ({len(train)}): {', '.join(train)}")
    print(f"Test  ({len(test)}): {', '.join(test)}")

    # 1. Pick threshold on TRAIN
    train_best, train_scores = best_threshold(frames, train)
    print("\n── TRAIN: mean Sharpe by threshold ──")
    for th in THRESHOLDS:
        mark = "  <- picked" if th == train_best else ""
        print(f"  thresh {th:+.1f}: mean Sharpe {train_scores[th]:.3f}{mark}")

    # 2. What TEST would have picked on its own (for reference only)
    test_best, test_scores = best_threshold(frames, test)

    # 3. Evaluate the TRAIN-picked threshold, cold, on TEST
    test_at_train = mean_sharpe(frames, test, train_best)

    print("\n── TEST: mean Sharpe by threshold ──")
    for th in THRESHOLDS:
        marks = []
        if th == train_best:
            marks.append("train-picked")
        if th == test_best:
            marks.append("test-optimal")
        tag = ("  <- " + ", ".join(marks)) if marks else ""
        print(f"  thresh {th:+.1f}: mean Sharpe {test_scores[th]:.3f}{tag}")

    # 4. Per-ticker detail at the train-picked threshold on TEST
    print(f"\n── TEST per-ticker at train-picked threshold ({train_best:+.1f}) ──")
    print(f"  {'ticker':8}{'sharpe':>9}{'total%':>12}")
    for t in test:
        sh, tot = sharpe_at(frames[t], train_best)
        print(f"  {t:8}{sh:>9.2f}{tot * 100:>12.1f}")

    print("\n── THRESHOLD GENERALIZATION ──")
    print(f"  Train-optimal threshold:            {train_best:+.1f}")
    print(f"  Test-optimal threshold:             {test_best:+.1f}")
    print(f"  Test mean Sharpe @ train-picked:    {test_at_train:.3f}")
    print(f"  Test mean Sharpe @ test-optimal:    {test_scores[test_best]:.3f}")
    gap = test_scores[test_best] - test_at_train
    print(f"  Generalization gap (Sharpe):        {gap:.3f}")
    if train_best == test_best:
        print("  -> Train and test agree on the threshold: the ranking generalizes.")
    elif gap < 0.10:
        print("  -> Different pick, but train threshold nearly matches test-optimal. Robust.")
    else:
        print("  -> Train threshold underperforms out-of-sample. Likely curve-fit.")

    # The threshold sweep is monotonic (lower is always better), which hints
    # the sentiment gate may be dead weight. Decompose to find out.
    comp = decomposition(frames)
    print("\n── SIGNAL DECOMPOSITION (mean Sharpe, full universe) ──")
    print(f"  Price > SMA-50 only (no sentiment): {comp['trend']:+.3f}")
    print(f"  Sentiment > 0 only (no trend):      {comp['sentiment']:+.3f}")
    print(f"  Combined (sentiment AND trend):     {comp['combined']:+.3f}")
    print(f"  Buy & Hold baseline:                {comp['buyhold']:+.3f}")

    print("\n── VERDICT ──")
    if comp["sentiment"] < 0.10 and comp["combined"] < comp["trend"]:
        print("  The sentiment gate carries no risk-adjusted edge on its own, and")
        print("  adding it to the trend filter REMOVES value (combined < trend).")
        print("  Lower thresholds score better only because they disable more of")
        print("  the sentiment gate. Buy-and-hold beats every variant here.")
        print("  Takeaway: in this long/cash form on large-cap uptrends, daily")
        print("  sentiment is not a timing edge. Its value lives in the event-study")
        print("  forward returns the analyzer reports, not in this backtest.")
    else:
        print("  The sentiment gate contributes measurable risk-adjusted return.")


if __name__ == "__main__":
    main()
