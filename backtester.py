import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from headline_analyzer import score_headline_live

# --- CONFIGURATION (Same as Institutional) ---
DATA_FILE = "TRADABLE_DATASET.csv"
INITIAL_CAPITAL = 1000000
TARGET_VOLATILITY = 0.40
MAX_POSITION_SIZE = 0.10
TRANSACTION_COST_BPS = 0.0015
SENTIMENT_THRESHOLD = 0.5

print("--- INSTITUTIONAL BACKTEST WITH VISUAL FORENSICS ---")

# 1. LOAD & PREPARE
print("Loading data...")
df = pd.read_csv(DATA_FILE)
df['date'] = pd.to_datetime(df['date'])
df.sort_values(by=['ticker', 'date'], inplace=True)

# 2. CALCULATE INDICATORS (Strategy Logic)
print("Calculating logic...")
df['returns'] = df.groupby('ticker')['close'].pct_change()
df['volatility_20d'] = df.groupby('ticker')['returns'].transform(lambda x: x.rolling(20).std() * np.sqrt(252))

# Volatility Sizing
df['vol_weight'] = TARGET_VOLATILITY / df['volatility_20d']
df['vol_weight'] = df['vol_weight'].replace([np.inf, -np.inf], 0).fillna(0)
df['position_size'] = df['vol_weight'].clip(upper=MAX_POSITION_SIZE)

# Alpha Signals
df['smooth_sentiment'] = df.groupby('ticker')['daily_score'].transform(lambda x: x.rolling(3).mean())
df['SMA_50'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(50).mean())

# SIGNAL GENERATION
# 1 = Long, 0 = Cash
df['signal'] = np.where(
    (df['smooth_sentiment'] > SENTIMENT_THRESHOLD) & (df['close'] > df['SMA_50']),
    1, 0
)

# 3. IDENTIFY TRADES (For Plotting)
print("Logging trade events...")
# A trade happens when the signal changes from yesterday to today
df['prev_signal'] = df.groupby('ticker')['signal'].shift(1).fillna(0)
df['trade_action'] = df['signal'] - df['prev_signal']
# trade_action = 1 (Buy), -1 (Sell), 0 (Hold)

# Filter only the rows where trades happened
trades = df[df['trade_action'] != 0].copy()


# ── FinBERT Headline Overlay ──────────────────────────────────────────────

def analyze_headline_overlay(ticker, ticker_df, headline):
    """Score headline with FinBERT and find signal zones where sentiment aligns.

    Returns (finbert_score, signal_zones_df, summary_dict) or (None, None, None) on error.
    """
    finbert_score = score_headline_live(headline)
    if finbert_score is None:
        print("  FinBERT scoring failed.")
        return None, None, None

    subset = ticker_df.copy()
    # Signal zones: where smooth_sentiment >= finbert_score AND close > SMA_50
    subset['headline_signal'] = np.where(
        (subset['smooth_sentiment'] >= finbert_score) & (subset['close'] > subset['SMA_50']),
        1, 0
    )
    signal_zones = subset[subset['headline_signal'] == 1]

    # Compute forward returns at signal points
    fwd_returns = []
    for idx in signal_zones.index:
        loc = subset.index.get_loc(idx)
        for offset in [1, 5, 10]:
            target = loc + offset
            if target < len(subset):
                ret = (subset['close'].iloc[target] - subset['close'].iloc[loc]) / subset['close'].iloc[loc] * 100
                fwd_returns.append({'offset': offset, 'return': ret})

    summary = {
        'finbert_score': finbert_score,
        'num_signals': len(signal_zones),
        'avg_1d': None,
        'avg_5d': None,
        'avg_10d': None,
    }
    for offset, label in [(1, 'avg_1d'), (5, 'avg_5d'), (10, 'avg_10d')]:
        vals = [r['return'] for r in fwd_returns if r['offset'] == offset]
        if vals:
            summary[label] = sum(vals) / len(vals)

    return finbert_score, signal_zones, summary


def print_signal_summary(summary):
    """Print a formatted summary of FinBERT headline overlay signals."""
    print(f"\n  ── FINBERT HEADLINE OVERLAY ──")
    print(f"  FinBERT Score:    {summary['finbert_score']:+.4f}")
    print(f"  Signal Points:   {summary['num_signals']}")

    for label, key in [("Avg 1D Return", "avg_1d"), ("Avg 5D Return", "avg_5d"), ("Avg 10D Return", "avg_10d")]:
        val = summary[key]
        if val is not None:
            sign = "+" if val >= 0 else ""
            print(f"  {label}:  {sign}{val:.2f}%")
        else:
            print(f"  {label}:  —")
    print()


# 4. INTERACTIVE PLOTTING TOOL
def plot_ticker_forensics(ticker, headline_overlay=None):
    subset = df[df['ticker'] == ticker].copy()

    if subset.empty:
        print(f"No data found for {ticker}")
        return

    # Get trades for this ticker
    ticker_trades = trades[trades['ticker'] == ticker]

    # If headline overlay requested, compute it
    finbert_score, signal_zones, summary = None, None, None
    if headline_overlay:
        finbert_score, signal_zones, summary = analyze_headline_overlay(ticker, subset, headline_overlay)
        if summary:
            print_signal_summary(summary)

    # Setup the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

    # --- TOP PANEL: PRICE & TRADES ---
    ax1.plot(subset['date'], subset['close'], label='Price', color='black', alpha=0.6, linewidth=1)
    ax1.plot(subset['date'], subset['SMA_50'], label='50-Day Trend', color='blue', linestyle='--', alpha=0.4)

    # Shade signal zones from headline overlay
    if signal_zones is not None and not signal_zones.empty:
        in_zone = False
        zone_start = None
        for i, row in subset.iterrows():
            if i in signal_zones.index:
                if not in_zone:
                    zone_start = row['date']
                    in_zone = True
            else:
                if in_zone:
                    ax1.axvspan(zone_start, row['date'], alpha=0.15, color='#00E5FF', label='_' if zone_start != signal_zones.iloc[0]['date'] else 'FinBERT Signal Zone')
                    in_zone = False
        if in_zone:
            ax1.axvspan(zone_start, subset['date'].iloc[-1], alpha=0.15, color='#00E5FF')

    # Create Color Map for Sentiment
    cmap = plt.cm.RdYlGn
    norm = Normalize(vmin=-1, vmax=1)

    # Plot BUY Markers
    buys = ticker_trades[ticker_trades['trade_action'] == 1]
    if not buys.empty:
        ax1.scatter(
            buys['date'], buys['close'],
            c=buys['smooth_sentiment'], cmap=cmap, norm=norm,
            marker='^', s=150, edgecolors='black', zorder=5, label='Buy Signal'
        )

    # Plot SELL Markers
    sells = ticker_trades[ticker_trades['trade_action'] == -1]
    if not sells.empty:
        ax1.scatter(
            sells['date'], sells['close'],
            c=sells['smooth_sentiment'], cmap=cmap, norm=norm,
            marker='v', s=150, edgecolors='black', zorder=5, label='Sell/Exit'
        )

    ax1.set_title(f"{ticker}: Trade Execution vs. Sentiment Score", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Stock Price ($)")
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Add Colorbar for Sentiment
    cbar_ax = fig.add_axes([0.92, 0.4, 0.02, 0.4])
    cb = plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
    cb.set_label('Sentiment Score (-1 to +1)')

    # --- BOTTOM PANEL: SENTIMENT INDICATOR ---
    ax2.plot(subset['date'], subset['smooth_sentiment'], color='purple', label='3-Day Sentiment', linewidth=1.5)
    ax2.axhline(y=SENTIMENT_THRESHOLD, color='green', linestyle=':', label='Buy Threshold (0.5)')
    ax2.axhline(y=0, color='gray', linewidth=0.5)

    # FinBERT reference line
    if finbert_score is not None:
        ax2.axhline(y=finbert_score, color='#00E5FF', linewidth=2, linestyle='--',
                     label=f'FinBERT Score ({finbert_score:+.3f})')

    # Fill areas
    ax2.fill_between(subset['date'], subset['smooth_sentiment'], 0, where=(subset['smooth_sentiment'] > 0), color='green', alpha=0.1)
    ax2.fill_between(subset['date'], subset['smooth_sentiment'], 0, where=(subset['smooth_sentiment'] < 0), color='red', alpha=0.1)

    ax2.set_ylabel("Sentiment")
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    plt.show()


# --- MAIN LOOP ---
print("\nSystem Ready.")
print("Type a ticker to inspect its history (e.g., 'NVDA', 'TSLA', 'AAPL').")
print("Type 'exit' to quit.")

while True:
    user_input = input("\nEnter Ticker: ").upper().strip()
    if user_input in ['EXIT', 'QUIT']:
        break
    try:
        plot_ticker_forensics(user_input)

        # Prompt for optional headline overlay
        headline = input("Headline (or enter to skip): ").strip()
        if headline:
            plot_ticker_forensics(user_input, headline_overlay=headline)
    except Exception as e:
        print(f"Error plotting {user_input}: {e}")
