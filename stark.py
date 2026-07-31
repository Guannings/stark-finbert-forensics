#!/usr/bin/env python3
"""
STARK — unified flow: headline → historical evidence → verdict → equity curve.

One command answers the whole question: what does this headline mean for the
stock, and would trading news like it have made money?

    python stark.py AAPL "Apple crushes quarterly estimates"

Stage 1 is the analyzer: FinBERT score, semantically similar historical
headlines, forward returns, and a verdict. Stage 2 turns the headline into a
strategy: go long whenever the ticker's 3-day smoothed news sentiment reaches
this headline's FinBERT level (with the 50-day trend filter), executed
next-day with transaction costs — and shows the equity curve vs buy-and-hold.
"""

import argparse

from rich.panel import Panel

import headline_analyzer as ha
import backtester as bt

console = ha.console


def run(ticker: str, headline: str, top_n: int = 20,
        date_from=None, date_to=None, method: str = "auto",
        show_chart: bool = True):
    ticker = ticker.strip().upper()

    # ── Stage 1: evidence + verdict ──────────────────────────────────────
    result = ha.analyze(ticker, headline, top_n=top_n,
                        date_from=date_from, date_to=date_to, method=method)
    if result is None:
        return None
    live_score = result["live_score"]
    if live_score is None:
        console.print(f"[{ha.NEON_YELLOW}]FinBERT unavailable — skipping signal backtest.[/]")
        return result

    # ── Stage 2: trade headlines like this one ───────────────────────────
    console.print(Panel(
        f"[bold {ha.NEON_CYAN}]SIGNAL BACKTEST[/]\n"
        f"[{ha.TEXT_COLOR}]Long {ticker} whenever 3-day news sentiment ≥ this headline's "
        f"FinBERT score ({live_score:+.3f}) and price is above the 50-day trend.\n"
        f"Next-day fills, 15 bps costs, volatility-targeted sizing.[/]",
        border_style=ha.NEON_CYAN, padding=(0, 2),
    ))

    frame = bt.build_frame(ticker)
    if frame is None or frame["daily_score"].notna().sum() == 0:
        console.print(f"[{ha.NEON_RED}]No daily sentiment history for {ticker} — cannot backtest.[/]")
        return result

    df = bt.run_backtest(bt.apply_signal(frame, live_score))
    metrics = bt.compute_metrics(df)
    bt.print_metrics(ticker, metrics)

    result["backtest"] = df
    result["backtest_metrics"] = metrics

    if show_chart:
        bt.plot_ticker_forensics(ticker, df, headline_overlay=headline, threshold=live_score)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="STARK unified flow: headline → verdict → equity curve",
        usage="%(prog)s TICKER HEADLINE [options]",
    )
    parser.add_argument("ticker", nargs="?", help="Stock ticker (e.g. AAPL)")
    parser.add_argument("headline", nargs="?", help="News headline to analyze")
    parser.add_argument("-n", "--top-n", type=int, default=20, help="Number of matches (default: 20)")
    parser.add_argument("--window", type=str, default=None, help="Time window: 1w, 1m, 3m, 6m, 1y")
    parser.add_argument("--since", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--until", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--lexical", action="store_true", help="Force keyword matching")
    parser.add_argument("--no-chart", action="store_true", help="Skip the matplotlib chart")

    args = parser.parse_args()

    if args.ticker and args.headline:
        ticker, headline = args.ticker, args.headline
    else:
        console.print(Panel(
            f"[bold {ha.NEON_CYAN}]STARK — HEADLINE TO EQUITY CURVE[/]\n"
            f"[{ha.DIM_TEXT}]Enter a ticker and a headline; get the verdict and the backtest.[/]",
            border_style=ha.NEON_CYAN, padding=(1, 2),
        ))
        try:
            ticker = console.input(f"[bold {ha.NEON_CYAN}]Ticker:[/] ").strip()
            headline = console.input(f"[bold {ha.NEON_CYAN}]Headline:[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        if not ticker or not headline:
            return

    date_from, date_to = ha.parse_time_window(args.window, args.since, args.until)
    run(ticker, headline, top_n=args.top_n,
        date_from=date_from, date_to=date_to,
        method="lexical" if args.lexical else "auto",
        show_chart=not args.no_chart)


if __name__ == "__main__":
    main()
