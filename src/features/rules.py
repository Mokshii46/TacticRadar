import chess
from typing import Dict, List, Any, Tuple, Optional

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 100,
}

SLIDER_TYPES = [chess.BISHOP, chess.ROOK, chess.QUEEN]


class RuleBasedTacticsDetector:
    """
    Heuristic rule-based tactical motif detector using python-chess.
    Provides baseline predictions and detailed coordinate mappings for board UI highlighting.
    """

    def __init__(self, board: Optional[chess.Board] = None):
        self.board = board

    def detect_forks(self, board: chess.Board, side: Optional[chess.Color] = None) -> List[Dict[str, Any]]:
        """
        Detects forks: a single piece attacking 2+ enemy targets that are higher value,
        undefended, or checking the King.
        """
        if side is None:
            sides = [board.turn, not board.turn]
        else:
            sides = [side]

        forks = []
        for attacker_color in sides:
            enemy_color = not attacker_color
            for sq in chess.SQUARES:
                piece = board.piece_at(sq)
                if not piece or piece.color != attacker_color:
                    continue

                att_val = PIECE_VALUES[piece.piece_type]
                attacks = board.attacks(sq)
                valuable_targets = []

                for target_sq in attacks:
                    target_piece = board.piece_at(target_sq)
                    if not target_piece or target_piece.color != enemy_color:
                        continue
                    
                    target_val = PIECE_VALUES[target_piece.piece_type]
                    is_defended = len(board.attackers(enemy_color, target_sq)) > 0
                    
                    # Target qualifies if King, or higher value than attacker, or undefended
                    if target_piece.piece_type == chess.KING or target_val > att_val or not is_defended:
                        valuable_targets.append(chess.square_name(target_sq))

                if len(valuable_targets) >= 2:
                    forks.append({
                        "motif": "fork",
                        "side": "white" if attacker_color == chess.WHITE else "black",
                        "attacker_square": chess.square_name(sq),
                        "attacker_piece": piece.symbol(),
                        "target_squares": valuable_targets,
                        "description": f"{piece.symbol()} on {chess.square_name(sq)} forks {', '.join(valuable_targets)}",
                    })

        return forks

    def detect_pins(self, board: chess.Board, side: Optional[chess.Color] = None) -> List[Dict[str, Any]]:
        """
        Detects pins: a slider attacking an enemy piece where moving it exposes
        a more valuable enemy piece (or King = absolute pin) along the ray.
        """
        if side is None:
            sides = [board.turn, not board.turn]
        else:
            sides = [side]

        pins = []
        for attacker_color in sides:
            enemy_color = not attacker_color
            
            # Check absolute pins first via board.is_pinned
            for sq in chess.SQUARES:
                piece = board.piece_at(sq)
                if piece and piece.color == enemy_color:
                    if board.is_pinned(enemy_color, sq):
                        king_sq = board.king(enemy_color)
                        if king_sq is not None:
                            # Find pinning piece
                            pinners = board.pin_mask(enemy_color, sq)
                            for p_sq in chess.SquareSet(pinners):
                                p_piece = board.piece_at(p_sq)
                                if p_piece and p_piece.color == attacker_color and p_piece.piece_type in SLIDER_TYPES:
                                    pins.append({
                                        "motif": "pin",
                                        "side": "white" if attacker_color == chess.WHITE else "black",
                                        "pinning_square": chess.square_name(p_sq),
                                        "pinned_square": chess.square_name(sq),
                                        "target_square": chess.square_name(king_sq),
                                        "is_absolute": True,
                                        "description": f"Absolute pin by {p_piece.symbol()} on {chess.square_name(sq)} against King",
                                    })

            # Check relative pins (e.g. Queen or Rook behind pinned piece)
            for p_sq in chess.SQUARES:
                p_piece = board.piece_at(p_sq)
                if not p_piece or p_piece.color != attacker_color or p_piece.piece_type not in SLIDER_TYPES:
                    continue

                for target_sq in chess.SQUARES:
                    target_piece = board.piece_at(target_sq)
                    if not target_piece or target_piece.color != enemy_color:
                        continue
                    if target_piece.piece_type == chess.KING:
                        continue  # handled by absolute pins
                    if PIECE_VALUES[target_piece.piece_type] <= 3:
                        continue  # target behind must be valuable (Rook, Queen)

                    # Check ray between p_sq and target_sq
                    ray = chess.SquareSet.ray(p_sq, target_sq)
                    between = chess.SquareSet.between(p_sq, target_sq)
                    if not ray or not between:
                        continue

                    # Check if exactly one piece is between them
                    between_pieces = [sq for sq in between if board.piece_at(sq) is not None]
                    if len(between_pieces) == 1:
                        mid_sq = between_pieces[0]
                        mid_piece = board.piece_at(mid_sq)
                        if mid_piece.color == enemy_color and PIECE_VALUES[mid_piece.piece_type] < PIECE_VALUES[target_piece.piece_type]:
                            pins.append({
                                "motif": "pin",
                                "side": "white" if attacker_color == chess.WHITE else "black",
                                "pinning_square": chess.square_name(p_sq),
                                "pinned_square": chess.square_name(mid_sq),
                                "target_square": chess.square_name(target_sq),
                                "is_absolute": False,
                                "description": f"Relative pin of {mid_piece.symbol()} on {chess.square_name(mid_sq)} against {target_piece.symbol()} on {chess.square_name(target_sq)}",
                            })

        # Remove duplicate pins by (pinned_square, pinning_square)
        unique_pins = []
        seen = set()
        for p in pins:
            key = (p["pinning_square"], p["pinned_square"])
            if key not in seen:
                seen.add(key)
                unique_pins.append(p)

        return unique_pins

    def detect_skewers(self, board: chess.Board, side: Optional[chess.Color] = None) -> List[Dict[str, Any]]:
        """
        Detects skewers: a slider attacks a high-value piece in front, with another piece behind it.
        When the high-value front piece moves, the piece behind is attacked.
        """
        if side is None:
            sides = [board.turn, not board.turn]
        else:
            sides = [side]

        skewers = []
        for attacker_color in sides:
            enemy_color = not attacker_color
            for p_sq in chess.SQUARES:
                p_piece = board.piece_at(p_sq)
                if not p_piece or p_piece.color != attacker_color or p_piece.piece_type not in SLIDER_TYPES:
                    continue

                for back_sq in chess.SQUARES:
                    back_piece = board.piece_at(back_sq)
                    if not back_piece or back_piece.color != enemy_color:
                        continue

                    ray = chess.SquareSet.ray(p_sq, back_sq)
                    between = chess.SquareSet.between(p_sq, back_sq)
                    if not ray or not between:
                        continue

                    between_pieces = [sq for sq in between if board.piece_at(sq) is not None]
                    if len(between_pieces) == 1:
                        front_sq = between_pieces[0]
                        front_piece = board.piece_at(front_sq)
                        if (
                            front_piece.color == enemy_color
                            and PIECE_VALUES[front_piece.piece_type] > PIECE_VALUES[back_piece.piece_type]
                            and PIECE_VALUES[front_piece.piece_type] >= 5  # Rook, Queen, King in front
                        ):
                            skewers.append({
                                "motif": "skewer",
                                "side": "white" if attacker_color == chess.WHITE else "black",
                                "skewer_square": chess.square_name(p_sq),
                                "front_square": chess.square_name(front_sq),
                                "back_square": chess.square_name(back_sq),
                                "description": f"Skewer by {p_piece.symbol()} on {front_piece.symbol()} ({chess.square_name(front_sq)}), exposing {back_piece.symbol()} ({chess.square_name(back_sq)})",
                            })

        # Deduplicate
        unique_skewers = []
        seen = set()
        for s in skewers:
            key = (s["skewer_square"], s["front_square"], s["back_square"])
            if key not in seen:
                seen.add(key)
                unique_skewers.append(s)

        return unique_skewers

    def detect_discovered_attacks(
        self, board: chess.Board, side: Optional[chess.Color] = None
    ) -> List[Dict[str, Any]]:
        """
        Detects discovered attacks: a friendly piece masks a friendly slider's line of sight
        to an enemy piece (especially King, Queen, Rook). When the friendly masking piece moves,
        a discovered attack occurs.
        """
        if side is None:
            sides = [board.turn, not board.turn]
        else:
            sides = [side]

        discovered = []
        for attacker_color in sides:
            enemy_color = not attacker_color
            for slider_sq in chess.SQUARES:
                slider_piece = board.piece_at(slider_sq)
                if not slider_piece or slider_piece.color != attacker_color or slider_piece.piece_type not in SLIDER_TYPES:
                    continue

                for target_sq in chess.SQUARES:
                    target_piece = board.piece_at(target_sq)
                    if not target_piece or target_piece.color != enemy_color:
                        continue
                    if PIECE_VALUES[target_piece.piece_type] < 3:
                        continue  # Target should be at least minor piece, rook, queen, or king

                    ray = chess.SquareSet.ray(slider_sq, target_sq)
                    between = chess.SquareSet.between(slider_sq, target_sq)
                    if not ray or not between:
                        continue

                    between_pieces = [sq for sq in between if board.piece_at(sq) is not None]
                    if len(between_pieces) == 1:
                        masking_sq = between_pieces[0]
                        masking_piece = board.piece_at(masking_sq)
                        # Masking piece MUST be friendly
                        if masking_piece.color == attacker_color:
                            is_check = (target_piece.piece_type == chess.KING)
                            discovered.append({
                                "motif": "discoveredAttack",
                                "side": "white" if attacker_color == chess.WHITE else "black",
                                "slider_square": chess.square_name(slider_sq),
                                "masking_square": chess.square_name(masking_sq),
                                "target_square": chess.square_name(target_sq),
                                "is_discovered_check": is_check,
                                "description": f"Discovered {'check' if is_check else 'attack'} on {target_piece.symbol()} by {slider_piece.symbol()} when {masking_piece.symbol()} on {chess.square_name(masking_sq)} moves",
                            })

        # Deduplicate
        unique_disc = []
        seen = set()
        for d in discovered:
            key = (d["slider_square"], d["masking_square"], d["target_square"])
            if key not in seen:
                seen.add(key)
                unique_disc.append(d)

        return unique_disc

    def detect_all(self, board: chess.Board) -> Dict[str, Any]:
        """
        Run all detectors on the position, returning motif lists and binary flags.
        """
        forks = self.detect_forks(board)
        pins = self.detect_pins(board)
        skewers = self.detect_skewers(board)
        disc = self.detect_discovered_attacks(board)

        return {
            "flags": {
                "fork": int(len(forks) > 0),
                "pin": int(len(pins) > 0),
                "skewer": int(len(skewers) > 0),
                "discoveredAttack": int(len(disc) > 0),
            },
            "counts": {
                "fork": len(forks),
                "pin": len(pins),
                "skewer": len(skewers),
                "discoveredAttack": len(disc),
            },
            "motifs": {
                "fork": forks,
                "pin": pins,
                "skewer": skewers,
                "discoveredAttack": disc,
            },
        }
