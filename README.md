# STARK Sentiment Analysis Suite

## Introduction

STARK is a quantitative sentiment analysis platform built to answer one question: **when a news headline drops, what historically happens to the stock price?**

The system combines a dataset of **85 million+ pre-scored financial headlines** (spanning 2009 to 2026) with **FinBERT**, a BERT-based NLP model fine-tuned specifically for financial text, to produce actionable sentiment verdicts grounded in real historical outcomes. Rather than relying on opinion or speculation, every verdict is backed by data: the system finds historically similar headlines, looks up what the stock actually did in the 1, 5, and 10 trading days that followed, and synthesizes a weighted signal from both the sentiment distribution and the observed price action.

This repository contains three independent tools that share a common analysis backend. You do not need to run them in sequence — each one is a standalone entry point designed for a different use case:

- **Headline Analyzer** — a command-line tool for fast, scriptable headline analysis. Supports single tickers, multi-ticker comparison, time-windowed filtering, and an interactive REPL mode.
- **Strategy Backtester** — a matplotlib-based visual backtesting tool that plots historical sentiment-driven trading signals (buy/sell markers, sentiment oscillator) and supports FinBERT headline overlays to test how a given headline's sentiment level would have performed as a signal filter.
- **Terminal Dashboard** — a full PyQt6 GUI with live price charts, a sentiment oscillator, summary metric cards, and an integrated headline analysis panel with a results table.

All three tools import their core logic from `headline_analyzer.py`, which handles FinBERT model loading, keyword-based headline search via DuckDB, forward return computation via yfinance, and verdict generation with exponential recency weighting. This means there is zero duplicated NLP or search logic across the codebase.

**If you are unsure where to start**, run `python main.py` — it launches an interactive menu that lets you pick a tool.

---

## Getting Started

### Installation

```bash
git clone https://github.com/Guannings/stark-finbert-forensics.git
cd stark-finbert-forensics
pip install -r requirements.txt
```

### Data Setup

The system requires a pre-scored headline dataset in parquet format. Due to its size (~several GB), it is not included in this repository. Place `STARK_SCORED_FIXED.parquet` in the project root directory, then build the fast search index:

```bash
python build_index.py
```

This deduplicates the raw headlines, sorts them by ticker for fast row-group pruning, and writes a compressed parquet index. It only needs to be run once.

The backtester additionally requires `TRADABLE_DATASET.csv` with the following columns: `ticker`, `date`, `close`, `daily_score`.

### Quick Start

```bash
# Launch the interactive tool selector
python main.py

# Or run any tool directly:
python headline_analyzer.py AAPL "earnings beat expectations"
python backtester.py
python stark_terminal.py
```

---

## Computational Requirements

| Resource | Minimum | Recommended |
|---|---|---|
| **Python** | 3.10+ | 3.11+ |
| **RAM** | 8 GB | 16 GB |
| **Disk** | ~6 GB (model + data) | ~10 GB |
| **GPU** | Not required | MPS (Apple Silicon) or CUDA for faster FinBERT inference |

**Notes:**
- The FinBERT model (`ProsusAI/finbert`) is approximately 440 MB and is downloaded automatically on first run via Hugging Face Transformers.
- DuckDB headline queries on the raw parquet (before building the index) can consume up to 12 GB of RAM. Building the index with `build_index.py` reduces query memory usage significantly.
- The PyQt6 terminal dashboard (`stark_terminal.py`) requires a graphical display environment and will not work in headless/SSH sessions without X-forwarding.
- On Apple Silicon Macs, FinBERT automatically uses the MPS backend for GPU-accelerated inference. On systems with NVIDIA GPUs, CUDA is used. CPU inference is the fallback and works on all platforms.
- The backtester loads the full `TRADABLE_DATASET.csv` into memory on startup. For very large datasets, ensure adequate RAM is available.

---

## Usage

### 1. Headline Analyzer (`headline_analyzer.py`)

Command-line tool for scoring headlines against historical data.

```bash
# Single ticker analysis
python headline_analyzer.py AAPL "earnings beat expectations"

# Multi-ticker comparison (comma-separated)
python headline_analyzer.py AAPL,TSLA,GOOGL "tech earnings beat expectations"

# Time-windowed search (only match headlines from the last year)
python headline_analyzer.py AAPL "revenue guidance raised" --window 1y

# Date range filter
python headline_analyzer.py TSLA "recall" --since 2023-01-01 --until 2024-01-01

# Adjust number of matches returned
python headline_analyzer.py NVDA "AI demand surge" --top-n 50

# Interactive REPL mode (no arguments)
python headline_analyzer.py
```

| Flag | Description |
|---|---|
| `-n`, `--top-n` | Number of similar headlines to return (default: 20) |
| `--window` | Time window filter: `1w`, `1m`, `3m`, `6m`, `1y` |
| `--since` | Start date for headline search (`YYYY-MM-DD`) |
| `--until` | End date for headline search (`YYYY-MM-DD`) |

In multi-ticker mode, FinBERT scores the headline once (since the score is ticker-agnostic), then runs the historical search and forward return analysis independently for each ticker, and finally renders a side-by-side comparison table.

### 2. Strategy Backtester (`backtester.py`)

Visual backtesting tool with matplotlib charts.

```bash
python backtester.py
```

1. Enter a ticker symbol (e.g., `NVDA`, `TSLA`, `AAPL`)
2. The system plots a two-panel chart: price action with buy/sell trade markers (top), and a sentiment oscillator with the buy threshold (bottom)
3. After the chart displays, you are prompted to optionally enter a headline
4. If a headline is entered, the system re-renders the chart with a FinBERT overlay: a horizontal reference line on the sentiment panel at the headline's FinBERT score, and shaded signal zones on the price panel where the historical sentiment met or exceeded that score while price was above the 50-day SMA
5. A summary prints showing how many hypothetical signal points were identified and their average 1/5/10-day forward returns

### 3. Terminal Dashboard (`stark_terminal.py`)

Full graphical dashboard built with PyQt6 and pyqtgraph.

```bash
python stark_terminal.py
```

- Enter a ticker in the top bar and click **EXECUTE** (or press Enter) to load price and sentiment data
- Use the time window dropdown to filter the chart view (All Time, 1 Year, 6 Months, 3 Months, 1 Month)
- Enter a headline in the analysis bar below the header and click **ANALYZE** to run a full headline analysis
- Results appear in a panel below the charts: verdict with confidence, FinBERT live score, and a table of matching historical headlines with sentiment scores and forward returns

### 4. Build Index (`build_index.py`)

One-time setup utility.

```bash
python build_index.py
```

Reads the raw `STARK_SCORED_FIXED.parquet`, deduplicates headlines per ticker, sorts by ticker for DuckDB row-group predicate pushdown, and writes a compressed (ZSTD) parquet index. This makes all subsequent headline queries significantly faster and less memory-intensive.

---

## Architecture

```
main.py                  <- launcher menu (pick a tool)
headline_analyzer.py     <- core analysis engine
      |            |
      v            v
backtester.py   stark_terminal.py
(matplotlib)    (PyQt6 + pyqtgraph)

build_index.py           <- one-time index builder
```

**`headline_analyzer.py`** is the shared backend. It exposes the following functions that the other tools import:

- `score_headline_live(headline)` — scores a headline with FinBERT, returns a float in [-1, +1]
- `extract_keywords(headline)` — tokenizes and filters a headline into search keywords
- `find_similar_headlines(ticker, keywords, ...)` — DuckDB-powered keyword overlap search with Jaccard similarity ranking
- `compute_forward_returns(ticker, dates)` — bulk yfinance download + 1/5/10-day forward return computation
- `compute_verdict(matches, returns)` — synthesizes sentiment distribution and price outcomes into a BULLISH/BEARISH/NEUTRAL verdict with confidence score and exponential recency weighting

**`backtester.py`** imports only `score_headline_live` — it has its own strategy logic (volatility sizing, SMA crossover, sentiment threshold signals) and uses FinBERT purely for the headline overlay feature.

**`stark_terminal.py`** imports five functions from the analyzer and uses them to power the headline analysis panel in the GUI.

---

## How It Works

### Headline Search

When you enter a headline, the system extracts keywords (lowercased, stopwords removed, deduplicated) and queries the headline index using DuckDB. The query computes word overlap between your keywords and every headline for the target ticker, filters to matches with at least 2 shared words, and ranks by shared word count with Jaccard similarity as a tiebreaker. DuckDB's predicate pushdown on the ticker-sorted parquet means only the relevant row groups are read from disk.

### FinBERT Scoring

The system uses `ProsusAI/finbert`, a BERT model fine-tuned on financial text. It outputs three probabilities (positive, neutral, negative) and the score is computed as `P(positive) - P(negative)`, yielding a value in [-1, +1]. The model is lazy-loaded as a singleton — the first call takes a few seconds to load weights, subsequent calls are near-instant.

### Verdict Generation

The verdict combines two signals:

- **Sentiment signal (40% weight):** the recency-weighted average sentiment of matched historical headlines, scaled to [-100, +100]
- **Price signal (60% weight):** the average forward returns (1D, 5D, 10D) after those historical headlines, scaled and clamped to [-100, +100]

Recency weighting applies an exponential decay with a half-life of 180 days, so recent headlines carry more influence than older ones. The composite score determines the verdict: above +10 is BULLISH, below -10 is BEARISH, and in between is NEUTRAL. Confidence scales with the magnitude of the composite signal.

### Backtester Strategy

The backtester implements a sentiment-momentum strategy: go long when the 3-day smoothed sentiment exceeds 0.5 AND the closing price is above the 50-day SMA. Position sizing is volatility-targeted (target volatility of 40%, capped at 10% of capital per position). Trade markers on the chart are color-coded by the sentiment score at the time of entry/exit using a red-yellow-green colormap.

---

## Disclaimer and Terms of Use

### 1. Educational Purpose Only

This software is provided strictly for educational and research purposes. It was built as a personal project by **PARVAUX**, a student at **National Chengchi University (NCCU)**. It is not intended to serve as a source of financial advice, and the author is not a registered financial advisor, broker, or analyst. The algorithms, models, and analytical techniques implemented herein — including FinBERT-based natural language processing, DuckDB-powered headline similarity search, Jaccard similarity scoring, exponential recency weighting, and sentiment-momentum backtesting — are demonstrations of theoretical and applied concepts in quantitative finance and natural language processing. They should not be construed as a recommendation to buy, sell, or hold any specific security, asset class, or financial instrument.

### 2. No Financial Advice

Nothing in this repository constitutes professional financial, legal, or tax advice. The verdicts, sentiment scores, confidence percentages, and forward return statistics generated by this software are the output of statistical models applied to historical data and should be treated as informational only. Investment decisions should be made based on your own independent research and consultation with a qualified financial professional. The strategies and signals modeled in this software may not be suitable for your specific financial situation, risk tolerance, or investment objectives.

### 3. Risk of Loss

All investments involve risk, including the possible loss of principal.

**a. Past Performance:** Historical sentiment scores, forward returns (1-day, 5-day, 10-day), and backtested strategy performance presented by this software are derived from historical data and are not indicative of future results. Markets are inherently unpredictable, and historical patterns may not repeat.

**b. Model Limitations:** The headline similarity search is based on keyword overlap (Jaccard similarity), which is a lexical approximation and does not capture semantic nuance, sarcasm, or context-dependent meaning. FinBERT, while fine-tuned on financial text, is a probabilistic model that can produce incorrect or misleading sentiment scores, particularly on ambiguous, novel, or domain-specific headlines.

**c. Signal Limitations:** The sentiment-momentum strategy implemented in the backtester uses fixed thresholds (sentiment > 0.5, price > 50-day SMA) that were not optimized for any specific market regime. These thresholds may fail in unprecedented macroeconomic environments, during liquidity crises, or in markets with structural changes.

**d. Data Accuracy:** Market data fetched from third-party APIs (Yahoo Finance via yfinance) may be delayed, adjusted, inaccurate, or incomplete. Headline sentiment scores in the dataset were computed via batch processing and may contain errors. The author makes no guarantee regarding the accuracy, completeness, or timeliness of any data used by this software.

**e. Recency Weighting:** The exponential recency weighting applied to verdict computation (half-life of 180 days) is an arbitrary parameterization. Different half-life values would produce different verdicts, and there is no theoretical guarantee that recent headlines are more predictive than older ones.

### 4. Hardware and Computation Liability

The author assumes no responsibility for hardware failure, system instability, excessive memory consumption, or data loss resulting from the execution of this software. DuckDB queries on large parquet files (85M+ rows) can consume significant system memory (up to 12 GB before index building). FinBERT model loading requires approximately 440 MB of disk space and GPU memory allocation. The PyQt6 terminal dashboard requires a graphical display environment. Execution of this software should only be performed on hardware meeting the minimum computational requirements specified in this document.

### 5. "AS-IS" Software Warranty

THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT. IN NO EVENT SHALL THE AUTHOR OR COPYRIGHT HOLDER BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

BY USING THIS SOFTWARE, YOU AGREE TO ASSUME ALL RISKS ASSOCIATED WITH YOUR INVESTMENT DECISIONS AND HARDWARE USAGE, RELEASING THE AUTHOR (PARVAUX) FROM ANY LIABILITY REGARDING YOUR FINANCIAL OUTCOMES OR SYSTEM INTEGRITY.

---

## Development Methodology

The core financial strategy, system architecture, and analytical approach were conceptualized and designed by the author. The idea of combining large-scale historical headline data with real-time FinBERT scoring to produce data-backed trading verdicts is original work — from the choice of Jaccard similarity for headline matching, to the recency-weighted verdict formula, to the sentiment-momentum backtesting strategy.

This project was built using an **AI-Accelerated Workflow**. The author is not a software developer by training — the domain expertise lies in quantitative finance and data analysis. Large Language Models (Gemini, Claude Opus 4.6) were utilized extensively to accelerate code implementation, generate syntax, scaffold boilerplate, and debug technical issues. This allowed development to remain focused on what matters: the quantitative logic, parameter design, signal validation, and risk management — rather than getting blocked on language-specific implementation details.

In short: the *what* and *why* came from the author; the *how* was accelerated by AI tooling.

---

## Contributors

1. **PARVAUX** — Project author. Strategy design, system architecture, quantitative logic, parameter tuning, and risk management.
2. **Claude Opus 4.6** (Anthropic) — AI-assisted code implementation, syntax generation, and debugging.
3. **Gemini** (Google) — AI-assisted code implementation and scaffolding.
