#!/usr/bin/env python3
"""
STARK Sentiment Analysis Suite — Launcher
Run this file to pick a tool, or run any tool directly.
"""

import sys
import os

BANNER = """
 ____  _____  _    ____  _  __
/ ___||_   _|/ \\  |  _ \\| |/ /
\\___ \\  | | / _ \\ | |_) | ' /
 ___) | | |/ ___ \\|  _ <| . \\
|____/  |_/_/   \\_|_| \\_|_|\\_\\

  Sentiment Analysis Suite
  From a headline to a verdict to an equity curve — grounded in 85M+ scored headlines.
"""

TOOLS = {
    "1": {
        "name": "Unified Flow",
        "desc": "Headline -> verdict -> equity curve of trading that signal (start here)",
        "module": "stark",
    },
    "2": {
        "name": "Headline Analyzer",
        "desc": "Score a news headline against historical data for any ticker",
        "module": "headline_analyzer",
    },
    "3": {
        "name": "Strategy Backtester",
        "desc": "Sentiment-momentum backtest: equity curve, Sharpe, drawdown vs buy & hold",
        "module": "backtester",
    },
    "4": {
        "name": "Terminal Dashboard",
        "desc": "Full PyQt6 GUI with price charts, sentiment, and headline analysis",
        "module": "stark_terminal",
    },
    "5": {
        "name": "Build Headline Index",
        "desc": "Pre-process headlines into a fast search index (run once)",
        "module": "build_index",
    },
}


def main():
    print(BANNER)
    print("  Available tools:\n")
    for key, tool in TOOLS.items():
        print(f"    [{key}]  {tool['name']}")
        print(f"         {tool['desc']}\n")

    print("    [q]  Quit\n")

    choice = input("  Select tool (1-5): ").strip()

    if choice in ("q", "quit", "exit"):
        return

    if choice not in TOOLS:
        print(f"\n  Invalid choice: {choice}")
        return

    tool = TOOLS[choice]
    print(f"\n  Launching {tool['name']}...\n")

    import importlib
    mod = importlib.import_module(tool["module"])
    mod.main()


if __name__ == "__main__":
    main()
