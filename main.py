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
"""

TOOLS = {
    "1": {
        "name": "Headline Analyzer",
        "desc": "Score a news headline against historical data for any ticker",
        "module": "headline_analyzer",
    },
    "2": {
        "name": "Strategy Backtester",
        "desc": "Visual backtesting with sentiment signals + FinBERT overlay",
        "module": "backtester",
    },
    "3": {
        "name": "Terminal Dashboard",
        "desc": "Full PyQt6 GUI with price charts, sentiment, and headline analysis",
        "module": "stark_terminal",
    },
    "4": {
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

    choice = input("  Select tool (1-4): ").strip()

    if choice in ("q", "quit", "exit"):
        return

    if choice not in TOOLS:
        print(f"\n  Invalid choice: {choice}")
        return

    tool = TOOLS[choice]
    print(f"\n  Launching {tool['name']}...\n")

    # Run the selected module as __main__
    import importlib
    mod = importlib.import_module(tool["module"])

    # Modules with main() get called; others run on import (backtester)
    if hasattr(mod, "main"):
        mod.main()


if __name__ == "__main__":
    main()
