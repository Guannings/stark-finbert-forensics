#!/usr/bin/env python3
"""
Build Headline Index — Deduplicates 85M headlines into a compact, sorted parquet.
DuckDB queries this with predicate pushdown on the ticker-sorted row groups.
"""

import os
import sys

import duckdb
from rich.console import Console
from rich.panel import Panel
from rich.progress import SpinnerColumn, TextColumn, Progress

console = Console()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARQUET_PATH = os.path.join(SCRIPT_DIR, "STARK_SCORED_FIXED.parquet")
INDEX_PATH = os.path.join(SCRIPT_DIR, "headline_index.parquet")


def build_index():
    if not os.path.exists(PARQUET_PATH):
        console.print(f"[bold red]Parquet not found: {PARQUET_PATH}[/bold red]")
        sys.exit(1)

    if os.path.exists(INDEX_PATH):
        console.print(f"[yellow]Index already exists: {INDEX_PATH}[/yellow]")
        resp = console.input("[yellow]Overwrite? (y/N): [/yellow]").strip().lower()
        if resp != "y":
            console.print("[dim]Aborted.[/dim]")
            return
        os.remove(INDEX_PATH)

    console.print(Panel("[bold cyan]BUILDING HEADLINE INDEX[/bold cyan]", border_style="cyan"))

    con = duckdb.connect()
    con.execute("SET memory_limit='8GB'")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET threads=4")

    # Single-pass: dedup + sort by ticker → ZSTD parquet
    # Sorting by ticker lets DuckDB skip irrelevant row groups via min/max stats
    console.print("[cyan]Reading parquet, deduplicating, and writing sorted index...[/cyan]")

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
        progress.add_task("[cyan]Working — this takes a few minutes...", total=None)

        con.execute("""
            COPY (
                SELECT
                    ticker,
                    headline,
                    MIN(date) AS date,
                    AVG(sentiment_score) AS sentiment_score
                FROM read_parquet($1)
                GROUP BY ticker, headline
                ORDER BY ticker
            ) TO $2 (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """, [PARQUET_PATH, INDEX_PATH])

    # Stats from the freshly written file
    console.print("[cyan]Reading stats...[/cyan]")
    stats = con.execute("""
        SELECT
            COUNT(*) AS rows,
            COUNT(DISTINCT ticker) AS tickers,
            MIN(date) AS min_date,
            MAX(date) AS max_date
        FROM read_parquet($1)
    """, [INDEX_PATH]).fetchone()

    con.close()

    idx_size_mb = os.path.getsize(INDEX_PATH) / (1024 * 1024)

    console.print(Panel(
        f"[bold green]Index built successfully[/bold green]\n\n"
        f"  Rows:    {stats[0]:>12,}\n"
        f"  Tickers: {stats[1]:>12,}\n"
        f"  Range:   {str(stats[2])[:10]} → {str(stats[3])[:10]}\n"
        f"  Size:    {idx_size_mb:>12.1f} MB\n"
        f"  File:    {INDEX_PATH}",
        title="[bold cyan]INDEX STATS[/bold cyan]",
        border_style="green",
    ))


if __name__ == "__main__":
    build_index()
