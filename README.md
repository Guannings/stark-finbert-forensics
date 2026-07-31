# STARK Sentiment Analysis Suite

## Introduction

STARK is a quantitative sentiment analysis platform built to answer one question: **when a news headline drops, what historically happens to the stock price?**

The system combines a dataset of **85 million+ pre-scored financial headlines** (spanning 2009 to 2026) with **FinBERT**, a BERT-based NLP model fine-tuned specifically for financial text, to produce actionable sentiment verdicts grounded in real historical outcomes. Rather than relying on opinion or speculation, every verdict is backed by data: the system finds historically similar headlines via **local embedding-based semantic search** (BAAI/bge-base-en-v1.5 — "Apple tops forecasts" matches "iPhone sales help Apple beat estimates" even with zero shared keywords), looks up what the stock actually did in the 1, 5, and 10 trading days that followed, and synthesizes a weighted signal from both the sentiment distribution and the observed price action.

The flagship entry point is the **Unified Flow** (`stark.py`), which answers the whole question in one command: `python stark.py AAPL "Apple crushes quarterly estimates"` runs the headline analysis (FinBERT score, semantically similar historical headlines, forward returns, verdict), then immediately backtests trading that signal — long whenever the ticker's smoothed news sentiment reaches this headline's level, with trend filter, next-day fills, and costs — and shows the equity curve against buy-and-hold.

The individual tools remain standalone entry points sharing the same analysis backend:

- **Headline Analyzer** — a command-line tool for fast, scriptable headline analysis. Supports single tickers, multi-ticker comparison, time-windowed filtering, and an interactive REPL mode.
- **Strategy Backtester** — a full sentiment-momentum backtest over the same headline dataset: equity curve vs. buy-and-hold, Sharpe, max drawdown, transaction costs, and volatility-targeted sizing, with next-day execution to avoid look-ahead bias. Includes the visual forensics chart (buy/sell markers, sentiment oscillator) and FinBERT headline overlays.
- **Terminal Dashboard** — a full PyQt6 GUI with live price charts, a sentiment oscillator, summary metric cards, and an integrated headline analysis panel with a results table.

All three tools import their core logic from `headline_analyzer.py`, which handles FinBERT model loading, headline search (semantic embeddings via `semantic_search.py`, with a keyword/DuckDB fallback), forward return computation via yfinance, and verdict generation with exponential recency weighting. This means there is zero duplicated NLP or search logic across the codebase.

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

The system requires a pre-scored headline dataset in parquet format. Due to its size (~several GB), it is not included in this repository.

The data files can live anywhere on disk. The tools look for them in this order:

1. The directory in the `STARK_DATA_DIR` environment variable
2. The directory named in a `.stark_data_dir` file in the project root (one line, gitignored)
3. The project root itself

Point one of those at the folder containing `STARK_SCORED_FIXED.parquet`, then build the fast search index:

```bash
python build_index.py
```

This deduplicates the raw headlines, sorts them by ticker for fast row-group pruning, and writes a compressed parquet index. It only needs to be run once. All tools — including the backtester, which derives its daily sentiment series from this same dataset — share this single data source.

### Quick Start

```bash
# The unified flow: headline -> verdict -> equity curve
python stark.py AAPL "earnings beat expectations"

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
- The embedding model (`BAAI/bge-base-en-v1.5`, ~440 MB) is downloaded automatically on the first semantic search. Per-ticker embedding caches use roughly 1.5 MB of disk per 1,000 headlines.
- DuckDB headline queries on the raw parquet (before building the index) can consume up to 12 GB of RAM. Building the index with `build_index.py` reduces query memory usage significantly.
- The PyQt6 terminal dashboard (`stark_terminal.py`) requires a graphical display environment and will not work in headless/SSH sessions without X-forwarding.
- On Apple Silicon Macs, FinBERT automatically uses the MPS backend for GPU-accelerated inference. On systems with NVIDIA GPUs, CUDA is used. CPU inference is the fallback and works on all platforms.
- The backtester aggregates daily sentiment per ticker directly from the headline index via DuckDB, so it needs no additional dataset.

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
| `--lexical` | Force keyword/Jaccard matching instead of semantic search |

In multi-ticker mode, FinBERT scores the headline once (since the score is ticker-agnostic), then runs the historical search and forward return analysis independently for each ticker, and finally renders a side-by-side comparison table.

### 2. Strategy Backtester (`backtester.py`)

Sentiment-momentum backtest with performance metrics and matplotlib charts.

```bash
python backtester.py
```

1. Enter a ticker symbol (e.g., `NVDA`, `TSLA`, `AAPL`)
2. The system derives the ticker's daily sentiment from the headline index, joins it with yfinance prices, runs the backtest, and prints a results table: total return, CAGR, annualized volatility, Sharpe, max drawdown, trade count, time in market, and daily win rate — side by side with buy-and-hold
3. A three-panel chart displays: price action with buy/sell trade markers (top), strategy vs. buy-and-hold equity curve on a log scale (middle), and the sentiment oscillator with the buy threshold (bottom)
4. After the chart closes, you are prompted to optionally enter a headline
5. If a headline is entered, the chart re-renders with a FinBERT overlay: a horizontal reference line on the sentiment panel at the headline's FinBERT score, and shaded signal zones on the price panel where the historical sentiment met or exceeded that score while price was above the 50-day SMA. A summary prints the average 1/5/10-day forward returns measured from each zone's entry day (entry days only, so overlapping days don't inflate the sample)

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
stark.py                 <- unified flow: analyzer + backtester in one command
headline_analyzer.py     <- core analysis engine
      |            |            \
      v            v             v
backtester.py   stark_terminal.py   semantic_search.py
(matplotlib)    (PyQt6 + pyqtgraph) (embeddings + per-ticker cache)

build_index.py           <- one-time index builder
validate_threshold.py    <- out-of-sample study of the sentiment signal
```

**`headline_analyzer.py`** is the shared backend. It exposes the following functions that the other tools import:

- `score_headline_live(headline)` — scores a headline with FinBERT, returns a float in [-1, +1]
- `extract_keywords(headline)` — tokenizes and filters a headline into search keywords
- `find_similar_headlines(ticker, keywords, ..., headline=, method=)` — semantic embedding search (via `semantic_search.py`) when a raw headline is provided, with automatic fallback to DuckDB keyword-overlap search ranked by Jaccard similarity
- `compute_forward_returns(ticker, dates)` — bulk yfinance download + 1/5/10-day forward return computation
- `compute_verdict(matches, returns)` — synthesizes sentiment distribution and price outcomes into a BULLISH/BEARISH/NEUTRAL verdict with confidence score and exponential recency weighting

**`backtester.py`** imports `score_headline_live` plus the shared data paths — its daily sentiment series is aggregated from the same headline index the analyzer searches, and it layers its own strategy logic (volatility sizing, SMA crossover, sentiment threshold signals, next-day execution, transaction costs) on top.

**`stark_terminal.py`** imports five functions from the analyzer and uses them to power the headline analysis panel in the GUI.

---

## How It Works

### Headline Search

**Semantic mode (default):** the input headline and every historical headline for the ticker are embedded with `BAAI/bge-base-en-v1.5` (running locally on MPS/CUDA/CPU via sentence-transformers), and matches are ranked by cosine similarity with a noise floor of 0.55. This captures paraphrase — "tops forecasts" matches "beats estimates" — which keyword overlap cannot. The first search for a ticker embeds its full deduped headline history (roughly a minute per 100k headlines on Apple Silicon) and caches the embeddings under `embed_cache/` in the data directory; subsequent searches load the cache instantly. Caches invalidate automatically when the headline index is rebuilt. Oversized groups (e.g. the `MARKET` bucket) are capped at the most recent 200k headlines.

**Lexical mode (fallback / `--lexical`):** the system extracts keywords (lowercased, stopwords removed, deduplicated) and queries the headline index using DuckDB. The query computes word overlap between your keywords and every headline for the target ticker, filters to matches with at least 2 shared words, and ranks by shared word count with Jaccard similarity as a tiebreaker. DuckDB's predicate pushdown on the ticker-sorted parquet means only the relevant row groups are read from disk. This mode runs automatically if sentence-transformers is not installed.

### FinBERT Scoring

The system uses `ProsusAI/finbert`, a BERT model fine-tuned on financial text. It outputs three probabilities (positive, neutral, negative) and the score is computed as `P(positive) - P(negative)`, yielding a value in [-1, +1]. The model is lazy-loaded as a singleton — the first call takes a few seconds to load weights, subsequent calls are near-instant.

### Verdict Generation

The verdict combines two signals:

- **Sentiment signal (40% weight):** the recency-weighted average sentiment of matched historical headlines, scaled to [-100, +100]
- **Price signal (60% weight):** the average forward returns (1D, 5D, 10D) after those historical headlines, scaled and clamped to [-100, +100]

Recency weighting applies an exponential decay with a half-life of 180 days, so recent headlines carry more influence than older ones — the same weights are applied to both the sentiment average and the forward-return averages. The composite score determines the verdict: above +10 is BULLISH, below -10 is BEARISH, and in between is NEUTRAL. Confidence scales with the magnitude of the composite signal and is discounted when fewer than 20 matches were found, so a weak signal on a thin sample reads as low confidence rather than defaulting to 50%+.

### Backtester Strategy

The backtester implements a sentiment-momentum strategy: go long when the 3-day smoothed sentiment clears a threshold (default 0.0, configurable) AND the closing price is above the 50-day SMA. A signal generated on day *t* is executed on day *t+1*, so the backtest never trades on information it wouldn't have had yet. Position sizing is volatility-targeted (40% annualized target, capped at fully invested), and 15 bps of transaction cost is charged on every position change. The equity curve compounds daily strategy returns from an initial $1,000,000 and is compared against buy-and-hold over the same period. Trade markers on the chart are color-coded by the sentiment score at the time of entry/exit using a red-yellow-green colormap.

### Does the sentiment signal actually work? (Validation)

`validate_threshold.py` runs an honest out-of-sample study over 25 large-cap names, and the answer is worth stating plainly: **as a long/cash market-timing signal, daily headline sentiment adds no risk-adjusted edge.**

The threshold was chosen to maximize mean Sharpe on a 13-ticker train set and then measured, untouched, on a disjoint 12-ticker test set. Both sets independently pick the *lowest* threshold tested (-0.2), with a zero generalization gap — but the absolute Sharpe there is only ~0.06. That monotonic "lower is always better" pattern is the tell: pushing the threshold down just disables the sentiment gate. Decomposing the signal confirms it (mean Sharpe across all 25 tickers):

| Signal | Mean Sharpe |
|---|---|
| Buy & Hold | **+0.52** |
| Price > 50-day SMA only (no sentiment) | +0.27 |
| Sentiment > 0 only (no trend) | +0.06 |
| Combined (sentiment AND trend) | −0.01 |

The trend filter carries what little edge exists; the sentiment gate is roughly zero on its own and *removes* value when combined (−0.01 < +0.27). Buy-and-hold beats every variant.

This isn't a failure of the project — it's the project working as intended: a data-backed system honest enough to falsify its own strategy. It also points somewhere specific. Sentiment's value shows up not in this coarse daily-timing form but in the **event-study forward returns the analyzer reports** — "after headlines like this one, the stock did X over 1/5/10 days" — which is a conditional, event-driven measurement, not a continuous market-timing gate. Extending sentiment into a long/short book, or to intraday/event horizons, is the natural next research direction.

---

## Disclaimer and Terms of Use

### 1. Educational Purpose Only

This software is provided strictly for educational and research purposes. It was built as a personal project by **PARVAUX**, a student at **National Chengchi University (NCCU)**. It is not intended to serve as a source of financial advice, and the author is not a registered financial advisor, broker, or analyst. The algorithms, models, and analytical techniques implemented herein — including FinBERT-based natural language processing, DuckDB-powered headline similarity search, Jaccard similarity scoring, exponential recency weighting, and sentiment-momentum backtesting — are demonstrations of theoretical and applied concepts in quantitative finance and natural language processing. They should not be construed as a recommendation to buy, sell, or hold any specific security, asset class, or financial instrument.

### 2. No Financial Advice

Nothing in this repository constitutes professional financial, legal, or tax advice. The verdicts, sentiment scores, confidence percentages, and forward return statistics generated by this software are the output of statistical models applied to historical data and should be treated as informational only. Investment decisions should be made based on your own independent research and consultation with a qualified financial professional. The strategies and signals modeled in this software may not be suitable for your specific financial situation, risk tolerance, or investment objectives.

### 3. Risk of Loss

All investments involve risk, including the possible loss of principal.

**a. Past Performance:** Historical sentiment scores, forward returns (1-day, 5-day, 10-day), and backtested strategy performance presented by this software are derived from historical data and are not indicative of future results. Markets are inherently unpredictable, and historical patterns may not repeat.

**b. Model Limitations:** The headline similarity search uses embedding cosine similarity (with a keyword-overlap fallback), which captures paraphrase but can still surface topically related headlines describing materially different events, and does not reliably capture sarcasm or context-dependent meaning. FinBERT, while fine-tuned on financial text, is a probabilistic model that can produce incorrect or misleading sentiment scores, particularly on ambiguous, novel, or domain-specific headlines.

**c. Signal Limitations:** The sentiment-momentum strategy implemented in the backtester uses fixed thresholds (sentiment > 0.0, price > 50-day SMA) chosen from an in-sample sweep over only three tickers and not optimized for any specific market regime. These thresholds may fail in unprecedented macroeconomic environments, during liquidity crises, or in markets with structural changes.

**d. Data Accuracy:** Market data fetched from third-party APIs (Yahoo Finance via yfinance) may be delayed, adjusted, inaccurate, or incomplete. Headline sentiment scores in the dataset were computed via batch processing and may contain errors. The author makes no guarantee regarding the accuracy, completeness, or timeliness of any data used by this software.

**e. Recency Weighting:** The exponential recency weighting applied to verdict computation (half-life of 180 days) is an arbitrary parameterization. Different half-life values would produce different verdicts, and there is no theoretical guarantee that recent headlines are more predictive than older ones.

### 4. Hardware and Computation Liability

The author assumes no responsibility for hardware failure, system instability, excessive memory consumption, or data loss resulting from the execution of this software. DuckDB queries on large parquet files (85M+ rows) can consume significant system memory (up to 12 GB before index building). FinBERT model loading requires approximately 440 MB of disk space and GPU memory allocation. The PyQt6 terminal dashboard requires a graphical display environment. Execution of this software should only be performed on hardware meeting the minimum computational requirements specified in this document.

### 5. "AS-IS" Software Warranty

**THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT. IN NO EVENT SHALL THE AUTHOR OR COPYRIGHT HOLDER BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.**

**BY USING THIS SOFTWARE, YOU AGREE TO ASSUME ALL RISKS ASSOCIATED WITH YOUR INVESTMENT DECISIONS AND HARDWARE USAGE, RELEASING THE AUTHOR (PARVAUX) FROM ANY LIABILITY REGARDING YOUR FINANCIAL OUTCOMES OR SYSTEM INTEGRITY.**

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

