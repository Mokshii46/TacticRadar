import chess
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from src.features.rules import RuleBasedTacticsDetector, PIECE_VALUES, SLIDER_TYPES

PIECE_TYPES = [
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
]
PIECE_NAMES = ["pawn", "knight", "bishop", "rook", "queen", "king"]


def get_king_box(king_square: int) -> chess.SquareSet:
    """Returns the 3x3 square zone around a king square."""
    rank = chess.square_rank(king_square)
    file = chess.square_file(king_square)
    box_sqs = set()
    for r in range(max(0, rank - 1), min(8, rank + 2)):
        for f in range(max(0, file - 1), min(8, file + 2)):
            box_sqs.add(chess.square(f, r))
    return chess.SquareSet(box_sqs)


def extract_tabular_features(fen_or_board: Any) -> Dict[str, float]:
    """
    Extract comprehensive tactical and positional features from a chess position.
    Features strictly follow Goren, PAL, and modern chess ML literature.
    """
    if isinstance(fen_or_board, str):
        try:
            board = chess.Board(fen_or_board)
        except Exception:
            board = chess.Board()
    else:
        board = fen_or_board

    feats: Dict[str, float] = {}

    # 1. Game State & Turn
    side = board.turn
    feats["turn"] = 1.0 if side == chess.WHITE else 0.0
    feats["is_check"] = 1.0 if board.is_check() else 0.0
    feats["legal_moves_count"] = float(board.legal_moves.count())
    
    # Castling rights
    feats["white_castling_kingside"] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    feats["white_castling_queenside"] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    feats["black_castling_kingside"] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    feats["black_castling_queenside"] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0

    # 2. Material Counts and Balance
    white_mat = 0
    black_mat = 0
    for p_type, name in zip(PIECE_TYPES[:5], PIECE_NAMES[:5]):
        w_cnt = len(board.pieces(p_type, chess.WHITE))
        b_cnt = len(board.pieces(p_type, chess.BLACK))
        feats[f"count_white_{name}"] = float(w_cnt)
        feats[f"count_black_{name}"] = float(b_cnt)
        feats[f"diff_{name}"] = float(w_cnt - b_cnt)
        val = PIECE_VALUES[p_type]
        white_mat += w_cnt * val
        black_mat += b_cnt * val

    feats["total_material_white"] = float(white_mat)
    feats["total_material_black"] = float(black_mat)
    feats["material_balance"] = float(white_mat - black_mat)
    # Relative material balance from perspective of side to move
    feats["material_balance_pov"] = float((white_mat - black_mat) if side == chess.WHITE else (black_mat - white_mat))

    # 3. Piece Attackers, Defenders, Hanging Pieces
    white_hanging = 0
    black_hanging = 0
    white_undefended = 0
    black_undefended = 0
    white_attacked_by_lower = 0
    black_attacked_by_lower = 0
    total_defenders_white = 0
    total_defenders_black = 0
    pieces_count_white = 0
    pieces_count_black = 0

    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if not piece:
            continue

        color = piece.color
        val = PIECE_VALUES[piece.piece_type]
        defenders = board.attackers(color, sq)
        enemy_attackers = board.attackers(not color, sq)
        n_def = len(defenders)
        n_att = len(enemy_attackers)

        if color == chess.WHITE:
            pieces_count_white += 1
            total_defenders_white += n_def
            if n_def == 0:
                white_undefended += 1
            if n_att > 0 and n_def == 0 and piece.piece_type != chess.KING:
                white_hanging += 1
            # Check attacked by lower value
            for a_sq in enemy_attackers:
                a_piece = board.piece_at(a_sq)
                if a_piece and PIECE_VALUES[a_piece.piece_type] < val:
                    white_attacked_by_lower += 1
                    break
        else:
            pieces_count_black += 1
            total_defenders_black += n_def
            if n_def == 0:
                black_undefended += 1
            if n_att > 0 and n_def == 0 and piece.piece_type != chess.KING:
                black_hanging += 1
            for a_sq in enemy_attackers:
                a_piece = board.piece_at(a_sq)
                if a_piece and PIECE_VALUES[a_piece.piece_type] < val:
                    black_attacked_by_lower += 1
                    break

    feats["white_hanging_pieces"] = float(white_hanging)
    feats["black_hanging_pieces"] = float(black_hanging)
    feats["hanging_difference"] = float(white_hanging - black_hanging)
    feats["white_undefended_pieces"] = float(white_undefended)
    feats["black_undefended_pieces"] = float(black_undefended)
    feats["white_attacked_by_lower"] = float(white_attacked_by_lower)
    feats["black_attacked_by_lower"] = float(black_attacked_by_lower)
    feats["avg_defenders_white"] = float(total_defenders_white / max(1, pieces_count_white))
    feats["avg_defenders_black"] = float(total_defenders_black / max(1, pieces_count_black))

    # 4. Mobility and Attacks by Piece Type
    for p_type, name in zip(PIECE_TYPES[:5], PIECE_NAMES[:5]):
        w_attacks = sum(len(board.attacks(sq)) for sq in board.pieces(p_type, chess.WHITE))
        b_attacks = sum(len(board.attacks(sq)) for sq in board.pieces(p_type, chess.BLACK))
        feats[f"attacks_white_{name}"] = float(w_attacks)
        feats[f"attacks_black_{name}"] = float(b_attacks)

    # Opponent mobility (switch turn temporarily)
    board.push(chess.Move.null())
    feats["opponent_legal_moves"] = float(board.legal_moves.count())
    board.pop()

    # 5. Fork Features (Maximum valuable targets attacked by single piece)
    detector = RuleBasedTacticsDetector()
    all_detections = detector.detect_all(board)
    counts = all_detections["counts"]

    max_targets_by_type = {p: 0 for p in PIECE_TYPES[:5]}
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if not piece:
            continue
        p_val = PIECE_VALUES[piece.piece_type]
        if piece.piece_type not in max_targets_by_type:
            continue
        
        # Count valuable enemy targets
        enemy = not piece.color
        targets = 0
        for att_sq in board.attacks(sq):
            t_piece = board.piece_at(att_sq)
            if t_piece and t_piece.color == enemy:
                t_val = PIECE_VALUES[t_piece.piece_type]
                is_def = len(board.attackers(enemy, att_sq)) > 0
                if t_piece.piece_type == chess.KING or t_val > p_val or not is_def:
                    targets += 1
        if targets > max_targets_by_type[piece.piece_type]:
            max_targets_by_type[piece.piece_type] = targets

    for p_type, name in zip(PIECE_TYPES[:5], PIECE_NAMES[:5]):
        feats[f"max_fork_targets_{name}"] = float(max_targets_by_type[p_type])
    feats["max_fork_targets_overall"] = float(max(max_targets_by_type.values()))
    feats["rule_fork_count"] = float(counts["fork"])
    feats["rule_fork_flag"] = float(all_detections["flags"]["fork"])

    # 6. Pin and Skewer Features
    white_abs_pins = sum(1 for sq in chess.SQUARES if board.piece_at(sq) and board.piece_at(sq).color == chess.WHITE and board.is_pinned(chess.WHITE, sq))
    black_abs_pins = sum(1 for sq in chess.SQUARES if board.piece_at(sq) and board.piece_at(sq).color == chess.BLACK and board.is_pinned(chess.BLACK, sq))
    feats["white_absolute_pins"] = float(white_abs_pins)
    feats["black_absolute_pins"] = float(black_abs_pins)
    feats["total_absolute_pins"] = float(white_abs_pins + black_abs_pins)
    feats["rule_pin_count"] = float(counts["pin"])
    feats["rule_pin_flag"] = float(all_detections["flags"]["pin"])
    feats["rule_skewer_count"] = float(counts["skewer"])
    feats["rule_skewer_flag"] = float(all_detections["flags"]["skewer"])

    # Aligned pieces on lines with enemy sliders
    aligned_pins = 0
    aligned_skewers = 0
    for s_sq in chess.SQUARES:
        s_piece = board.piece_at(s_sq)
        if not s_piece or s_piece.piece_type not in SLIDER_TYPES:
            continue
        for enemy_sq in chess.SQUARES:
            e_piece = board.piece_at(enemy_sq)
            if not e_piece or e_piece.color == s_piece.color or PIECE_VALUES[e_piece.piece_type] < 3:
                continue
            between = chess.SquareSet.between(s_sq, enemy_sq)
            if len([sq for sq in between if board.piece_at(sq)]) == 1:
                aligned_pins += 1

    feats["aligned_slider_targets"] = float(aligned_pins)

    # 7. Discovered Attack Features
    feats["rule_discovered_count"] = float(counts["discoveredAttack"])
    feats["rule_discovered_flag"] = float(all_detections["flags"]["discoveredAttack"])
    disc_checks = sum(1 for d in all_detections["motifs"]["discoveredAttack"] if d.get("is_discovered_check"))
    feats["discovered_checks_count"] = float(disc_checks)

    # 8. King Safety Features
    w_king_sq = board.king(chess.WHITE)
    b_king_sq = board.king(chess.BLACK)

    if w_king_sq is not None:
        w_king_box = get_king_box(w_king_sq)
        w_attackers_near_king = sum(len(board.attackers(chess.BLACK, sq)) for sq in w_king_box)
        w_king_file = chess.square_file(w_king_sq)
        w_file_pawns = len([sq for sq in board.pieces(chess.PAWN, chess.WHITE) if chess.square_file(sq) == w_king_file])
        feats["white_attackers_near_king"] = float(w_attackers_near_king)
        feats["white_king_file_open"] = 1.0 if w_file_pawns == 0 else 0.0
    else:
        feats["white_attackers_near_king"] = 0.0
        feats["white_king_file_open"] = 0.0

    if b_king_sq is not None:
        b_king_box = get_king_box(b_king_sq)
        b_attackers_near_king = sum(len(board.attackers(chess.WHITE, sq)) for sq in b_king_box)
        b_king_file = chess.square_file(b_king_sq)
        b_file_pawns = len([sq for sq in board.pieces(chess.PAWN, chess.BLACK) if chess.square_file(sq) == b_king_file])
        feats["black_attackers_near_king"] = float(b_attackers_near_king)
        feats["black_king_file_open"] = 1.0 if b_file_pawns == 0 else 0.0
    else:
        feats["black_attackers_near_king"] = 0.0
        feats["black_king_file_open"] = 0.0

    return feats


def get_feature_names() -> List[str]:
    """Returns sorted list of all feature names by running on starting position."""
    dummy = extract_tabular_features(chess.STARTING_FEN)
    return list(dummy.keys())


def extract_single_fen(fen: str) -> Dict[str, float]:
    return extract_tabular_features(fen)


def extract_features_batch(fens: List[str], num_workers: int = 4) -> pd.DataFrame:
    """
    Extract features in parallel or vectorized batches.
    """
    if num_workers <= 1 or len(fens) < 100:
        records = [extract_tabular_features(fen) for fen in tqdm(fens, desc="Extracting features")]
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            records = list(tqdm(executor.map(extract_single_fen, fens, chunksize=100), total=len(fens), desc="Extracting features"))

    return pd.DataFrame(records)
