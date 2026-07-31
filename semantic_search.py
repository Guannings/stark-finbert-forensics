#!/usr/bin/env python3
"""
Semantic headline search — embedding-based similarity over the headline index.

Replaces keyword/Jaccard matching with cosine similarity in embedding space,
so "Apple tops forecasts" matches "Apple beats estimates" even with zero
shared keywords. Embeddings are computed locally (BAAI/bge-base-en-v1.5 on
MPS/CUDA/CPU) and cached per ticker under DATA_DIR/embed_cache/ — the first
search for a ticker embeds its full headline history (~1 minute per 100k
headlines on Apple Silicon), every search after that loads the cache.
"""

import os

import numpy as np
import pandas as pd
import duckdb

MODEL_NAME = "BAAI/bge-base-en-v1.5"
MAX_EMBED_ROWS = 200_000   # cap for huge groups like the MARKET bucket
EMBED_BATCH_SIZE = 64      # modest: MPS shares memory with other apps (LM Studio etc.)
MAX_SEQ_LENGTH = 64        # headlines are short; capping saves a lot of memory
MIN_SIMILARITY = 0.55      # cosine floor below which matches are noise

_embedder_cache = {}


def _paths():
    """Resolve data paths lazily to avoid a circular import at module load."""
    from headline_analyzer import DATA_DIR, INDEX_PATH, PARQUET_PATH
    cache_dir = os.path.join(DATA_DIR, "embed_cache")
    return cache_dir, INDEX_PATH, PARQUET_PATH


def get_embedder():
    """Lazy-load the sentence-transformer (singleton)."""
    if "model" not in _embedder_cache:
        import torch
        from sentence_transformers import SentenceTransformer

        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        model = SentenceTransformer(MODEL_NAME, device=device)
        model.max_seq_length = MAX_SEQ_LENGTH
        _embedder_cache["model"] = model
    return _embedder_cache["model"]


def _load_ticker_rows(ticker: str) -> pd.DataFrame:
    """Fetch all (deduped) headlines for a ticker, most recent first."""
    _, index_path, parquet_path = _paths()
    con = duckdb.connect()
    try:
        if os.path.exists(index_path):
            df = con.execute(
                """
                SELECT headline, date, sentiment_score
                FROM read_parquet($1)
                WHERE ticker = $2
                ORDER BY date DESC
                LIMIT $3
                """,
                [index_path, ticker.upper(), MAX_EMBED_ROWS],
            ).fetchdf()
        else:
            df = con.execute(
                """
                SELECT headline, MIN(date) AS date, AVG(sentiment_score) AS sentiment_score
                FROM read_parquet($1)
                WHERE ticker = $2
                GROUP BY headline
                ORDER BY date DESC
                LIMIT $3
                """,
                [parquet_path, ticker.upper(), MAX_EMBED_ROWS],
            ).fetchdf()
    finally:
        con.close()
    return df


def _cache_files(ticker: str):
    cache_dir, _, _ = _paths()
    base = os.path.join(cache_dir, ticker.upper())
    return base + ".npz", base + ".parquet"


def _cache_is_fresh(ticker: str) -> bool:
    """Cache is valid if it exists and is newer than the headline index."""
    npz_path, meta_path = _cache_files(ticker)
    _, index_path, parquet_path = _paths()
    if not (os.path.exists(npz_path) and os.path.exists(meta_path)):
        return False
    src = index_path if os.path.exists(index_path) else parquet_path
    if not os.path.exists(src):
        return True
    return os.path.getmtime(npz_path) >= os.path.getmtime(src)


def build_ticker_cache(ticker: str, progress: bool = True):
    """Embed a ticker's headline history and cache it. Returns (emb, meta) or None."""
    rows = _load_ticker_rows(ticker)
    if rows.empty:
        return None

    if len(rows) == MAX_EMBED_ROWS:
        print(f"  [semantic] {ticker}: capped at most recent {MAX_EMBED_ROWS:,} headlines")

    model = get_embedder()
    emb = model.encode(
        rows["headline"].astype(str).tolist(),
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=progress,
        normalize_embeddings=True,
    ).astype(np.float16)

    npz_path, meta_path = _cache_files(ticker)
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    np.savez_compressed(npz_path, emb=emb, model=np.array(MODEL_NAME))
    rows.to_parquet(meta_path, index=False)
    return emb, rows


def get_ticker_cache(ticker: str):
    """Load (embeddings, rows) for a ticker, building the cache if needed."""
    npz_path, meta_path = _cache_files(ticker)
    if _cache_is_fresh(ticker):
        data = np.load(npz_path)
        if str(data["model"]) == MODEL_NAME:
            return data["emb"], pd.read_parquet(meta_path)
    return build_ticker_cache(ticker)


def semantic_search(ticker: str, headline: str, top_n: int = 20,
                    date_from=None, date_to=None,
                    min_similarity: float = MIN_SIMILARITY) -> pd.DataFrame:
    """Rank a ticker's headlines by cosine similarity to the input headline.

    Returns a DataFrame with columns: headline, date, sentiment_score, similarity.
    Empty DataFrame if the ticker has no headlines.
    """
    cached = get_ticker_cache(ticker)
    if cached is None:
        return pd.DataFrame()
    emb, rows = cached

    model = get_embedder()
    query = model.encode([headline], normalize_embeddings=True)[0]
    sims = emb.astype(np.float32) @ query.astype(np.float32)

    df = rows.copy()
    df["similarity"] = sims
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()

    if date_from is not None:
        df = df[df["date"] >= pd.Timestamp(date_from)]
    if date_to is not None:
        df = df[df["date"] <= pd.Timestamp(date_to)]

    df = df[df["similarity"] >= min_similarity]
    df = df.sort_values("similarity", ascending=False).head(top_n)
    return df.reset_index(drop=True)
