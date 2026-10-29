import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

logger = logging.getLogger(__name__)

LABEL_COLS = ["label_fork", "label_pin", "label_skewer", "label_discoveredAttack"]


def split_dataset(
    df: pd.DataFrame, config: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits dataset into 70% train, 15% validation, and 15% test
    using iterative multi-label stratification (iterstrat) keyed by PuzzleId.
    Prevents data leakage and preserves multi-label class distributions.
    """
    processed_dir = Path(config["data"]["processed_dir"])
    train_path = processed_dir / "train.parquet"
    val_path = processed_dir / "val.parquet"
    test_path = processed_dir / "test.parquet"

    if train_path.exists() and val_path.exists() and test_path.exists():
        logger.info("Loading existing stratified train/val/test splits...")
        train_df = pd.read_parquet(train_path)
        val_df = pd.read_parquet(val_path)
        test_df = pd.read_parquet(test_path)
        return train_df, val_df, test_df

    seed = config["data"].get("random_state", 42)
    test_size = config["data"].get("test_size", 0.15)
    val_size = config["data"].get("val_size", 0.15)

    X = df.index.values.reshape(-1, 1)
    Y = df[LABEL_COLS].values

    # Step 1: Split into (train+val) and (test)
    msss_test = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(msss_test.split(X, Y))
    
    train_val_df = df.iloc[train_val_idx].copy().reset_index(drop=True)
    test_df = df.iloc[test_idx].copy().reset_index(drop=True)

    # Step 2: Split (train+val) into train and val
    val_ratio_of_remaining = val_size / (1.0 - test_size)
    X_rem = train_val_df.index.values.reshape(-1, 1)
    Y_rem = train_val_df[LABEL_COLS].values

    msss_val = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=val_ratio_of_remaining, random_state=seed
    )
    train_idx, val_idx = next(msss_val.split(X_rem, Y_rem))

    train_df = train_val_df.iloc[train_idx].copy().reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].copy().reset_index(drop=True)

    # Verify no PuzzleId leakage
    train_ids = set(train_df["PuzzleId"])
    val_ids = set(val_df["PuzzleId"])
    test_ids = set(test_df["PuzzleId"])
    assert len(train_ids.intersection(val_ids)) == 0, "Leakage between train and val!"
    assert len(train_ids.intersection(test_ids)) == 0, "Leakage between train and test!"
    assert len(val_ids.intersection(test_ids)) == 0, "Leakage between val and test!"

    # Log distributions
    logger.info(f"Split sizes: Train={len(train_df)} ({len(train_df)/len(df):.1%}), "
                f"Val={len(val_df)} ({len(val_df)/len(df):.1%}), "
                f"Test={len(test_df)} ({len(test_df)/len(df):.1%})")
    
    for split_name, s_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        rates = [f"{col}: {s_df[col].mean():.3f}" for col in LABEL_COLS]
        logger.info(f"  {split_name} label rates -> {', '.join(rates)}")

    # Save splits
    train_df.to_parquet(train_path)
    val_df.to_parquet(val_path)
    test_df.to_parquet(test_path)
    logger.info("Saved stratified splits to data/processed/")

    return train_df, val_df, test_df
