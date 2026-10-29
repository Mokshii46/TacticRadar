import logging
import chess
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List, Set

logger = logging.getLogger(__name__)

TARGET_MOTIFS = ["fork", "pin", "skewer", "discoveredAttack"]


def apply_fen_offset(fen: str, first_move_uci: str) -> str:
    """
    Applies the first move in Moves to the pre-puzzle FEN to get the real puzzle position.
    In the Lichess puzzle database, FEN is the position before the opponent's move.
    """
    try:
        board = chess.Board(fen)
        move = chess.Move.from_uci(first_move_uci)
        if move in board.legal_moves:
            board.push(move)
            return board.fen()
        else:
            return ""
    except Exception:
        return ""


def parse_themes(themes_obj: Any) -> Set[str]:
    """Parse space or comma-separated themes string or array into a set."""
    if isinstance(themes_obj, (list, tuple, np.ndarray, set)):
        return {str(x).strip() for x in themes_obj}
    elif isinstance(themes_obj, str):
        return set(themes_obj.strip().split())
    return set()


def extract_quiet_positions(
    df: pd.DataFrame,
    num_quiet_needed: int,
    target_motifs: List[str],
    random_state: int = 42
) -> pd.DataFrame:
    """
    Sample realistic 'quiet' positions with all-zero labels.
    Uses non-tactical puzzles (e.g. quietMove, endgame, defensiveMove)
    that contain NONE of the target motifs.
    """
    logger.info(f"Extracting {num_quiet_needed} quiet/non-tactical positions...")
    
    # Filter for puzzles with none of the target motifs
    def has_no_target(themes_set):
        return all(m not in themes_set for m in target_motifs)
        
    mask = df["themes_set"].apply(has_no_target)
    quiet_df = df[mask].copy()
    
    # Prioritize puzzles tagged with quietMove, endgame, defensiveMove
    quiet_tags = {"quietMove", "endgame", "defensiveMove", "advantage", "zugzwang"}
    quiet_df["quiet_priority"] = quiet_df["themes_set"].apply(
        lambda s: len(s.intersection(quiet_tags))
    )
    quiet_df = quiet_df.sort_values(by=["quiet_priority", "Popularity"], ascending=[False, False])
    
    if len(quiet_df) > num_quiet_needed:
        quiet_df = quiet_df.sample(n=num_quiet_needed, random_state=random_state)
    
    # Fix FEN offset for quiet positions
    valid_quiet = []
    for _, row in quiet_df.iterrows():
        moves = str(row["Moves"]).split()
        if not moves:
            continue
        real_fen = apply_fen_offset(row["FEN"], moves[0])
        if not real_fen:
            continue
            
        record = {
            "PuzzleId": row["PuzzleId"],
            "FEN": real_fen,
            "pre_fen": row["FEN"],
            "first_move": moves[0],
            "solution_moves": " ".join(moves[1:]),
            "move_count": len(moves),
            "Rating": row["Rating"],
            "Popularity": row["Popularity"],
            "NbPlays": row["NbPlays"],
            "Themes": " ".join(row["Themes"]) if isinstance(row["Themes"], (list, np.ndarray)) else str(row["Themes"]),
            "is_quiet": 1,
            "label_fork": 0,
            "label_pin": 0,
            "label_skewer": 0,
            "label_discoveredAttack": 0,
        }
        valid_quiet.append(record)
        if len(valid_quiet) >= num_quiet_needed:
            break
            
    res_df = pd.DataFrame(valid_quiet)
    logger.info(f"Successfully generated {len(res_df)} quiet positions.")
    return res_df


def preprocess_dataset(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Full preprocessing pipeline:
    1. Filter on Popularity, NbPlays, and solution length (if enabled).
    2. Identify target tactical motifs (fork, pin, skewer, discoveredAttack).
    3. Apply FEN offset fix (push Moves[0]).
    4. Balance target classes and sample quiet positions.
    5. Cache processed dataset.
    """
    processed_dir = Path(config["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_path = processed_dir / "puzzles_processed.parquet"
    
    if processed_path.exists():
        logger.info(f"Loading already processed dataset from {processed_path}...")
        return pd.read_parquet(processed_path)

    cfg_data = config["data"]
    min_pop = cfg_data.get("min_popularity", 70)
    min_plays = cfg_data.get("min_nb_plays", 50)
    target_motifs = cfg_data.get("target_themes", TARGET_MOTIFS)
    target_per_label = cfg_data.get("target_samples_per_label", 5000)
    max_total = cfg_data.get("max_total_samples", 25000)
    neg_ratio = cfg_data.get("negative_ratio", 0.20)
    restrict_short = cfg_data.get("restrict_short_puzzles", False)
    max_moves = cfg_data.get("max_solution_moves", 4)
    seed = cfg_data.get("random_state", 42)

    logger.info(f"Raw dataset shape: {df.shape}")
    
    # Basic quality filtering
    df = df[(df["Popularity"] >= min_pop) & (df["NbPlays"] >= min_plays)].copy()
    logger.info(f"After popularity (>={min_pop}) and plays (>={min_plays}) filter: {len(df):,}")

    # Parse themes
    df["themes_set"] = df["Themes"].apply(parse_themes)
    
    # Filter short puzzles if restricted
    if restrict_short:
        df["move_count"] = df["Moves"].apply(lambda m: len(str(m).split()))
        df = df[df["move_count"] <= max_moves].copy()
        logger.info(f"After restricting to <= {max_moves} moves: {len(df):,}")

    # Collect tactical candidates per motif
    motif_puzzles: Dict[str, pd.DataFrame] = {}
    for motif in target_motifs:
        mask = df["themes_set"].apply(lambda s: motif in s)
        motif_df = df[mask]
        logger.info(f"Found {len(motif_df):,} puzzles containing '{motif}'")
        if len(motif_df) > target_per_label:
            motif_puzzles[motif] = motif_df.sample(n=target_per_label, random_state=seed)
        else:
            motif_puzzles[motif] = motif_df

    # Union the tactical candidates
    tactical_df = pd.concat(list(motif_puzzles.values())).drop_duplicates(subset=["PuzzleId"])
    logger.info(f"Combined unique tactical puzzles: {len(tactical_df):,}")

    # Process tactical positions (apply FEN offset)
    logger.info("Applying FEN offset fix (playing Moves[0])...")
    processed_tactical = []
    for _, row in tactical_df.iterrows():
        moves = str(row["Moves"]).split()
        if not moves:
            continue
        real_fen = apply_fen_offset(row["FEN"], moves[0])
        if not real_fen:
            continue
            
        themes_set = row["themes_set"]
        record = {
            "PuzzleId": row["PuzzleId"],
            "FEN": real_fen,
            "pre_fen": row["FEN"],
            "first_move": moves[0],
            "solution_moves": " ".join(moves[1:]),
            "move_count": len(moves),
            "Rating": row["Rating"],
            "Popularity": row["Popularity"],
            "NbPlays": row["NbPlays"],
            "Themes": " ".join(row["Themes"]) if isinstance(row["Themes"], (list, np.ndarray)) else str(row["Themes"]),
            "is_quiet": 0,
            "label_fork": int("fork" in themes_set),
            "label_pin": int("pin" in themes_set),
            "label_skewer": int("skewer" in themes_set),
            "label_discoveredAttack": int("discoveredAttack" in themes_set),
        }
        processed_tactical.append(record)

    tactical_processed_df = pd.DataFrame(processed_tactical)
    logger.info(f"Valid tactical puzzles after FEN offset: {len(tactical_processed_df):,}")

    # Calculate negative quiet sample count
    n_tactical = len(tactical_processed_df)
    n_quiet = int(n_tactical * (neg_ratio / (1.0 - neg_ratio)))
    quiet_processed_df = extract_quiet_positions(
        df,
        num_quiet_needed=n_quiet,
        target_motifs=target_motifs,
        random_state=seed
    )

    # Combine tactical and quiet positions
    combined_df = pd.concat([tactical_processed_df, quiet_processed_df], ignore_index=True)
    combined_df = combined_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Cap if total exceeds max_total_samples
    if len(combined_df) > max_total:
        combined_df = combined_df.iloc[:max_total].copy()

    logger.info(f"Final preprocessed dataset size: {len(combined_df):,}")
    logger.info("Label breakdown:")
    for motif in target_motifs:
        count = (combined_df[f"label_{motif}"] == 1).sum()
        pct = 100.0 * count / len(combined_df)
        logger.info(f"  - {motif}: {count:,} ({pct:.1f}%)")
    quiet_count = (combined_df["is_quiet"] == 1).sum()
    logger.info(f"  - quiet (no target tactics): {quiet_count:,} ({100.0*quiet_count/len(combined_df):.1f}%)")

    # Save to parquet
    combined_df.to_parquet(processed_path)
    logger.info(f"Saved processed dataset to {processed_path}")
    return combined_df
