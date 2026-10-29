import os
import sys
import logging
import requests
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

HF_PARQUET_SHARD_0 = (
    "https://huggingface.co/api/datasets/Lichess/chess-puzzles/parquet/default/train/0.parquet"
)
LICHESS_ZST_URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_file(url: str, dest_path: Path, chunk_size: int = 1024 * 1024) -> Path:
    """Download a file with progress reporting."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading from {url} to {dest_path}...")
    
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    
    downloaded = 0
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    done_pct = int(50 * downloaded / total_size)
                    mb_down = downloaded / (1024 * 1024)
                    mb_tot = total_size / (1024 * 1024)
                    sys.stdout.write(
                        f"\r[{'=' * done_pct}{' ' * (50 - done_pct)}] {mb_down:.1f}/{mb_tot:.1f} MB"
                    )
                    sys.stdout.flush()
    print()
    logger.info(f"Download complete: {dest_path} ({downloaded / (1024 * 1024):.1f} MB)")
    return dest_path


def load_raw_dataset(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Load raw puzzle dataset according to config.
    Supports HuggingFace parquet shard (fastest), local files, or Lichess .csv.zst.
    """
    raw_dir = Path(config["data"]["raw_dir"])
    ensure_dir(raw_dir)
    source = config["data"].get("source", "huggingface")
    
    # 1. Check for pre-cached raw parquet
    cached_parquet = raw_dir / "puzzles_raw.parquet"
    if cached_parquet.exists():
        logger.info(f"Loading cached raw parquet from {cached_parquet}...")
        return pd.read_parquet(cached_parquet)

    # 2. Check user's Downloads or raw_dir for full .csv.zst
    zst_candidates = [
        raw_dir / "lichess_db_puzzle.csv.zst",
        Path.home() / "Downloads" / "lichess_db_puzzle.csv.zst",
    ]
    found_zst = next((p for p in zst_candidates if p.exists()), None)

    if source == "lichess_zst" or (source == "local" and found_zst):
        target_zst = found_zst or (raw_dir / "lichess_db_puzzle.csv.zst")
        if not target_zst.exists():
            download_file(config["data"].get("zst_url", LICHESS_ZST_URL), target_zst)
        logger.info(f"Decompressing and reading {target_zst}...")
        import zstandard as zstd
        import io
        
        dctx = zstd.ZstdDecompressor()
        with open(target_zst, "rb") as fh:
            with dctx.stream_reader(fh) as reader:
                text_stream = io.TextIOWrapper(reader, encoding="utf-8")
                df = pd.read_csv(text_stream)
        logger.info(f"Loaded {len(df):,} puzzles from .csv.zst")
        df.to_parquet(cached_parquet)
        return df

    # 3. Default: Hugging Face parquet shard (over 2 million puzzles in 1 shard, fast!)
    logger.info(f"Fetching Hugging Face parquet shard from {HF_PARQUET_SHARD_0}...")
    cached_shard = raw_dir / "hf_shard_0.parquet"
    if not cached_shard.exists():
        download_file(HF_PARQUET_SHARD_0, cached_shard)
    
    logger.info(f"Reading parquet from {cached_shard}...")
    df = pd.read_parquet(cached_shard)
    logger.info(f"Loaded {len(df):,} raw puzzles from shard.")
    
    # Cache as puzzles_raw.parquet
    if not cached_parquet.exists():
        df.to_parquet(cached_parquet)
        
    return df
