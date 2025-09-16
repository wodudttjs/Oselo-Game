# ultra_ai/UltraAdvancedAI_corner_safe.py
# Corner-safety hardened search + Corner-first policy + 2-ply corner avoidance:
#  - If a legal corner move exists this turn, TAKE IT immediately (pre-search).
#  - Filter out moves that give opponent an immediate corner (1-ply) when alternatives exist.
#  - NEW: Filter (when possible) moves that let opponent force a corner in 2 plies.

from ast import pattern
from os import remove, startfile
from shutil import move
import time
import logging
import random
from dataclasses import dataclass

from typing import Optional, Tuple, List, Dict
import hashlib
import math
from zobrist import ZOBRIST_TABLE, ZOBRIST_TURN
from config import load_config

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')

# Sentinel scores
MATE_SCORE = 100000
INF_SCORE = 10**9

from collections import defaultdict
from typing import Optional, Tuple, Dict, List

from constants import (
    adjust_position_weight,
    BLACK,
    WHITE,
    EMPTY,
    opponent,
    CORNERS,
    X_SQUARES,
    C_SQUARES,
)

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')

# ---------------- Zobrist (deterministic) ----------------
_rng = random.Random(1337)
ZOBRIST_TABLE: List[List[List[int]]] = [[[0 for _ in range(3)] for _ in range(8)] for _ in range(8)]
for i in range(8):
    for j in range(8):
        ZOBRIST_TABLE[i][j][1] = _rng.getrandbits(64) or 1  # BLACK
        ZOBRIST_TABLE[i][j][2] = _rng.getrandbits(64) or 1  # WHITE
ZOBRIST_TURN: int = _rng.getrandbits(64) or 1


def _pidx(v: int) -> int:
    return 1 if v == BLACK else 2 if v == WHITE else 0


DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))


@dataclass(slots=True)
class TTEntry:
    score: float
    move: Optional[Tuple[int, int]]
    depth: int
    node_type: str
    age: int

    pos_hash: int
    
class BitBoard:
    """Bit-based board representation for ultra-fast operations"""
    def __init__(self, board=None):
        # Allow initializing from another BitBoard or from a Board
        if isinstance(board, BitBoard):
            # Fast copy from an existing BitBoard
            self.black = int(board.black)
            self.white = int(board.white)
            self.hash_base = np.uint64(getattr(board, 'hash_base', 0))
        elif board is not None:
            # Initialize from a classic Board (with .board 2D array)
            self.black = 0
            self.white = 0
            for i in range(8):
                for j in range(8):
                    pos = i * 8 + j
                    if board.board[i][j] == BLACK:
                        self.black |= (1 << pos)
                    elif board.board[i][j] == WHITE:
                        self.white |= (1 << pos)
            self.hash_base = self._compute_hash_base_from_masks(self.black, self.white)
        else:
            self.black = 0x0000001008000000  # Initial black positions
            self.white = 0x0000000810000000  # Initial white positions
            self.hash_base = self._compute_hash_base_from_masks(self.black, self.white)
    
    def get_empty_mask(self):
        return ~(self.black | self.white) & 0xFFFFFFFFFFFFFFFF
    
    def popcount(self, mask):
        return bin(mask).count('1')



    @staticmethod
    def _iter_bits(mask):
        while mask:
            lsb = mask & -mask
            idx = (lsb.bit_length() - 1)
            yield idx
            mask ^= lsb

    @staticmethod
    def _compute_hash_base_from_masks(black_mask: int, white_mask: int) -> np.uint64:
        h = np.uint64(0)
        for idx in BitBoard._iter_bits(black_mask):
            x, y = divmod(idx, 8)
            h ^= ZOBRIST_TABLE[x][y][BLACK]
        for idx in BitBoard._iter_bits(white_mask):
            x, y = divmod(idx, 8)
            h ^= ZOBRIST_TABLE[x][y][WHITE]
        return h

    # Directional masks for bitboard move generation
    _MASK = 0xFFFFFFFFFFFFFFFF
    _FILE_A = 0x0101010101010101
    _FILE_H = 0x8080808080808080
    _NOT_A = _MASK ^ _FILE_A
    _NOT_H = _MASK ^ _FILE_H

    @staticmethod
    def _shift_n(bb): return (bb >> 8) & BitBoard._MASK
    @staticmethod
    def _shift_s(bb): return (bb << 8) & BitBoard._MASK
    @staticmethod
    def _shift_e(bb): return ((bb & BitBoard._NOT_H) << 1) & BitBoard._MASK
    @staticmethod
    def _shift_w(bb): return ((bb & BitBoard._NOT_A) >> 1) & BitBoard._MASK
    @staticmethod
    def _shift_ne(bb): return ((bb & BitBoard._NOT_H) >> 7) & BitBoard._MASK
    @staticmethod
    def _shift_nw(bb): return ((bb & BitBoard._NOT_A) >> 9) & BitBoard._MASK
    @staticmethod
    def _shift_se(bb): return ((bb & BitBoard._NOT_H) << 9) & BitBoard._MASK
    @staticmethod
    def _shift_sw(bb): return ((bb & BitBoard._NOT_A) << 7) & BitBoard._MASK

    def _valid_moves_mask(self, own, opp):
        empty = ~(own | opp) & BitBoard._MASK
        moves = 0
        for sh in (
            self._shift_n, self._shift_s, self._shift_e, self._shift_w,
            self._shift_ne, self._shift_nw, self._shift_se, self._shift_sw,
        ):
            x = sh(own) & opp
            c = x
            while c:
                c = sh(c) & opp
                x |= c
            moves |= sh(x) & empty
        return moves

    def get_valid_moves_bitboard(self, color):
        own = self.black if color == BLACK else self.white
        opp = self.white if color == BLACK else self.black
        mask = self._valid_moves_mask(own, opp)
        moves = []
        while mask:
            lsb = mask & -mask
            idx = (lsb.bit_length() - 1)
            x, y = divmod(idx, 8)
            moves.append((x, y))
            mask ^= lsb
        return moves

    def _flip_mask_for_move(self, own, opp, move_bit):
        flips = 0
        for sh in (
            self._shift_n, self._shift_s, self._shift_e, self._shift_w,
            self._shift_ne, self._shift_nw, self._shift_se, self._shift_sw,
        ):
            x = 0
            c = sh(move_bit)
            while c and (c & opp):
                x |= c
                c = sh(c)
            if c and (c & own):
                flips |= x
        return flips

    def apply_move_bitboard(self, x, y, color):
        idx = x * 8 + y
        move_bit = 1 << idx
        own = self.black if color == BLACK else self.white
        opp = self.white if color == BLACK else self.black
        flips = self._flip_mask_for_move(own, opp, move_bit)
        if flips == 0:
            return None  # illegal
        own ^= flips | move_bit
        opp ^= flips
        bb = BitBoard()
        if color == BLACK:
            bb.black, bb.white = own, opp
        else:
            bb.white, bb.black = own, opp
        # Incremental hash update
        base = np.uint64(self.hash_base)
        base ^= ZOBRIST_TABLE[x][y][color]
        opp_color = opponent(color)
        for fidx in BitBoard._iter_bits(flips):
            fx, fy = divmod(fidx, 8)
            base ^= ZOBRIST_TABLE[fx][fy][opp_color]
            base ^= ZOBRIST_TABLE[fx][fy][color]
        bb.hash_base = base
        return bb


# ---------------- Modular Evaluator and SearchEngine ----------------

class Evaluator:
    def __init__(self):
        pass

    def evaluate_bitboard(self, bb: BitBoard, ai_color: int) -> int:
        empty = 64 - bb.popcount(bb.black | bb.white)
        my = bb.black if ai_color == BLACK else bb.white
        op = bb.white if ai_color == BLACK else bb.black
        disc_diff = bb.popcount(my) - bb.popcount(op)
        my_moves = len(bb.get_valid_moves_bitboard(ai_color))
        op_moves = len(bb.get_valid_moves_bitboard(opponent(ai_color)))
        mobility = 0
        tot_moves = my_moves + op_moves
        if tot_moves:
            mobility = (my_moves - op_moves) / tot_moves
        corner_idx = [(0,0),(0,7),(7,0),(7,7)]
        corners = 0
        for (cx, cy) in corner_idx:
            ci = cx*8+cy
            mask = 1 << ci
            if my & mask:
                corners += 1
            elif op & mask:
                corners -= 1
        score = 0
        score += corners * 1000
        score += mobility * 200
        score += disc_diff * 10
        if empty > 50:
            score *= 1.2
        elif empty <= 20:
            score *= 0.8
        return int(score)


class SearchEngine:
    def __init__(self, evaluator: Evaluator, tt_size: int, time_limit: float):
        self.evaluator = evaluator
        self.tt = {}
        self.tt_age = 0
        self.max_tt_size = tt_size
        self.killer_moves = defaultdict(list)
        self.counter_moves = {}
        self.history_table = np.zeros((8, 8), dtype=np.int32)
        self.nodes_searched = 0
        self.tt_hits = 0
        self.cutoffs = 0
        self.time_limit = time_limit

    @staticmethod
    def bit_pos_hash(bb: BitBoard) -> int:
        return int(getattr(bb, 'hash_base', 0))

    @staticmethod
    def bit_hash(bb: BitBoard, side_to_move: int) -> int:
        return int(np.uint64(SearchEngine.bit_pos_hash(bb)) ^ (ZOBRIST_TURN if side_to_move == BLACK else np.uint64(0)))

    def _static_move_value_bb(self, move, bb: BitBoard):
        x, y = move
        score = 0
        if move in CORNERS:
            score += 1000
        elif move in X_SQUARES:
            score -= 200
        elif move in C_SQUARES:
            score -= 200
        elif x == 0 or x == 7 or y == 0 or y == 7:
            score += 100
        return score

    def enhanced_move_ordering_bb(self, ai_color: int, bb: BitBoard, moves, depth, prev_best=None):
        if not moves:
            return []
        move_scores = []
        for i, move in enumerate(moves):
            score = 0
            if prev_best and move == prev_best:
                score += 10000
            tt_entry = self.tt.get(SearchEngine.bit_hash(bb, ai_color))
            if tt_entry and tt_entry.move == move:
                score += 8000
            if depth in self.killer_moves and move in self.killer_moves[depth]:
                score += 4000 + (2 - self.killer_moves[depth].index(move)) * 1000
            x, y = move
            total_history = max(1, int(np.sum(self.history_table)))
            history_score = (int(self.history_table[x][y]) * 2000) // total_history
            score += history_score
            score += self._static_move_value_bb(move, bb)
            idx = x * 8 + y
            mv_bit = 1 << idx
            own = bb.black if ai_color == BLACK else bb.white
            opp = bb.white if ai_color == BLACK else bb.black
            flips = bb._flip_mask_for_move(own, opp, mv_bit)
            score += bin(flips).count('1') * 50
            move_scores.append((score, i, move))
        move_scores.sort(reverse=True)
        return [m for _, _, m in move_scores]

    def alpha_beta_bb(self, ai_color: int, bb: BitBoard, depth: int, alpha: float, beta: float, side_to_move: int, start_time: float):
        self.nodes_searched += 1
        if self.nodes_searched % 4096 == 0:
            if time.time() - start_time > self.time_limit * 0.95:
                return self.evaluator.evaluate_bitboard(bb, ai_color), None
        key = SearchEngine.bit_hash(bb, side_to_move)
        tt_entry = self.tt.get(key)
        tt_move = None
        if tt_entry and tt_entry.depth >= depth and SearchEngine.bit_pos_hash(bb) == tt_entry.pos_hash:
            self.tt_hits += 1
            if tt_entry.node_type == 'exact':
                return tt_entry.score, tt_entry.move
            elif tt_entry.node_type == 'lowerbound' and tt_entry.score >= beta:
                return tt_entry.score, tt_entry.move
            elif tt_entry.node_type == 'upperbound' and tt_entry.score <= alpha:
                return tt_entry.score, tt_entry.move
            tt_move = tt_entry.move
        moves = bb.get_valid_moves_bitboard(side_to_move)
        if depth == 0 or not moves:
            if not moves:
                opp = opponent(side_to_move)
                # If opponent also has no moves, it's terminal; no need to re-wrap bb
                if not bb.get_valid_moves_bitboard(opp):
                    bcnt = bb.popcount(bb.black)
                    wcnt = bb.popcount(bb.white)
                    diff = (bcnt - wcnt) if ai_color == BLACK else (wcnt - bcnt)
                    return MATE_SCORE * (1 if diff > 0 else -1 if diff < 0 else 0), None
                return self.alpha_beta_bb(ai_color, bb, depth, alpha, beta, opp, start_time)
            return self.evaluator.evaluate_bitboard(bb, ai_color), None
        ordered = self.enhanced_move_ordering_bb(ai_color, bb, moves, depth, tt_move)
        best_move = ordered[0] if ordered else None
        best_score = float('-inf')
        original_alpha = alpha
        opp_color = opponent(side_to_move)
        for i, move in enumerate(ordered):
            x, y = move
            next_bb = bb.apply_move_bitboard(x, y, side_to_move)
            if next_bb is None:
                continue
            if i == 0:
                score, _ = self.alpha_beta_bb(ai_color, next_bb, depth - 1, -beta, -alpha, opp_color, start_time)
                score = -score
            else:
                score, _ = self.alpha_beta_bb(ai_color, next_bb, depth - 1, -alpha - 1, -alpha, opp_color, start_time)
                score = -score
                if alpha < score < beta:
                    score, _ = self.alpha_beta_bb(ai_color, next_bb, depth - 1, -beta, -score, opp_color, start_time)
                    score = -score
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, score)
            if beta <= alpha:
                self.cutoffs += 1
                if len(self.killer_moves[depth]) >= 2:
                    self.killer_moves[depth].pop(0)
                self.killer_moves[depth].append(move)
                hx, hy = move
                self.history_table[hx][hy] += depth * depth
                break
        node_type = 'exact'
        if best_score <= original_alpha:
            node_type = 'upperbound'
        elif best_score >= beta:
            node_type = 'lowerbound'
        if not math.isfinite(best_score):
            best_score = MATE_SCORE if best_score > 0 else -MATE_SCORE
        if len(self.tt) >= self.max_tt_size:
            try:
                evict_key = min(self.tt, key=lambda k: (self.tt[k].age, self.tt[k].depth))
                self.tt.pop(evict_key, None)
            except ValueError:
                pass
        self.tt[key] = TTEntry(
            score=best_score,
            move=best_move,
            depth=depth,
            node_type=node_type,
            age=self.tt_age,
            pos_hash=SearchEngine.bit_pos_hash(bb)
        )
        return best_score, best_move

    def get_best_move(self, board: Board, ai_color: int, max_depth: int):
        start_time = time.time()
        bb = BitBoard(board)
        moves = bb.get_valid_moves_bitboard(ai_color)
        if not moves:
            return None
        if len(moves) == 1:
            return moves[0]
        best_move = None
        prev_score = 0
        aspiration = 50
        for depth in range(1, max_depth + 1):
            try:
                self.nodes_searched = 0
                self.tt_hits = 0
                self.cutoffs = 0
                self.tt_age += 1
                depth_start = time.time()
                if depth >= 4 and best_move:
                    alpha = prev_score - aspiration
                    beta = prev_score + aspiration
                    score, move = self.alpha_beta_bb(ai_color, bb, depth, alpha, beta, ai_color, start_time)
                    if score <= alpha:
                        score, move = self.alpha_beta_bb(ai_color, bb, depth, float('-inf'), beta, ai_color, start_time)
                    elif score >= beta:
                        score, move = self.alpha_beta_bb(ai_color, bb, depth, alpha, float('inf'), ai_color, start_time)
                    aspiration = min(100, aspiration + 25)
                else:
                    score, move = self.alpha_beta_bb(ai_color, bb, depth, float('-inf'), float('inf'), ai_color, start_time)
                if move:
                    best_move = move
                    prev_score = score
                depth_time = time.time() - depth_start
                logging.info(f"[Search] Depth {depth}: {depth_time:.3f}s, {self.nodes_searched} nodes, TT: {self.tt_hits}, Score: {score}, Move: {move}")
                if math.isfinite(score) and abs(score) >= MATE_SCORE:
                    logging.info("Terminal score - stopping")
                    break
                if time.time() - start_time > self.time_limit * 0.8:
                    logging.info("Time budget nearly used - stop")
                    break
            except Exception as e:
                logging.error(f"[Search] Error at depth {depth}: {e}")
                break
        return best_move

    def evaluate_bitboard(self, color):
        own = self.black if color == BLACK else self.white
        opp = self.white if color == BLACK else self.black
        # Simple disc differential
        return self.popcount(own) - self.popcount(opp)

class UltraAdvancedAI:
    """Corner-first Othello AI with 2-ply corner exposure avoidance.
       Public: get_move(board) -> (x,y) or None
    """

    def __init__(self, color: int, difficulty: str = 'hard', time_limit: float = 10.0) -> None:
        self.color = color
        self.difficulty = difficulty
        self.time_limit = float(time_limit)

        self.tt: Dict[int, TTEntry] = {}
        self.tt_age = 0
        self.max_tt_size = 2**20

        self.killer_moves = defaultdict(list)
        self.counter_moves: Dict[Tuple[int, int], Tuple[int, int]] = {}
        self.history_table: List[int] = [0] * 64

        self.nodes_searched = 0
        self.tt_hits = 0
        self.cutoffs = 0

        self.pattern_weights = self._init_patterns()

        self.opening_book: Dict[int, Tuple[int, int]] = {}
        self.endgame_cache: Dict[Tuple[int, int], Tuple[int, Optional[Tuple[int, int]]]] = {}

        self._init_pos_weights()
        if difficulty == 'easy':
            self.max_depth = 6
        elif difficulty == 'medium':
            self.max_depth = 8
        else:
            self.max_depth = 10

            self.selective_depth = 16
            
        # Pre-computed patterns for ultra-fast evaluation
        self._precompute_patterns()

        # Load configuration and instantiate modular components
        cfg = load_config()
        self.config = cfg
        # Apply config values
        self.max_tt_size = cfg.get('tt_size', self.max_tt_size)
        self.time_limit = cfg.get('time_limit', self.time_limit)
        if 'difficulties' in cfg and isinstance(cfg['difficulties'], dict):
            d = cfg['difficulties'].get(self.difficulty, {})
            self.max_depth = d.get('max_depth', self.max_depth)
            self.use_parallel = d.get('use_parallel', getattr(self, 'use_parallel', False))

        # Modular evaluator and search engine
        self.evaluator = Evaluator()
        self.search_engine = SearchEngine(self.evaluator, self.max_tt_size, self.time_limit)

    

    # -------- Bitboard helpers (hashing, ordering, evaluation) --------

    def bit_pos_hash(self, bb: BitBoard) -> int:
        return int(getattr(bb, 'hash_base', 0))

    def bit_hash(self, bb: BitBoard, side_to_move: int) -> int:
        base = np.uint64(self.bit_pos_hash(bb))
        return int(base ^ (ZOBRIST_TURN if side_to_move == BLACK else np.uint64(0)))

    def _static_move_value_bb(self, move, bb: BitBoard):
        x, y = move
        score = 0
        if move in CORNERS:
            score += 1000
        elif move in X_SQUARES:
            # X-square penalty unless adjacent corner is owned
            adjacent_corners = [(cx, cy) for cx, cy in CORNERS
                                if abs(cx - x) <= 1 and abs(cy - y) <= 1]
            # We don't track colors per-square here; approximate using neutral penalty
            score -= 200
        elif move in C_SQUARES:
            score -= 200
        elif x == 0 or x == 7 or y == 0 or y == 7:
            score += 100
        return score

    def enhanced_move_ordering_bb(self, bb: BitBoard, moves, depth, prev_best=None):
        if not moves:
            return []
        move_scores = []
        for i, move in enumerate(moves):
            score = 0
            if prev_best and move == prev_best:
                score += 10000

            # TT best move bonus
            tt_entry = self.tt.get(self.bit_hash(bb, self.color))
            if tt_entry and tt_entry.move == move:
                score += 8000

            # Killer moves
            if depth in self.killer_moves and move in self.killer_moves[depth]:
                score += 4000 + (2 - self.killer_moves[depth].index(move)) * 1000

            # History heuristic
            x, y = move
            total_history = max(1, int(np.sum(self.history_table)))
            history_score = (int(self.history_table[x][y]) * 2000) // total_history
            score += history_score

            # Static position value
            score += self._static_move_value_bb(move, bb)

            # Flip count bonus via bitboard
            idx = x * 8 + y
            mv_bit = 1 << idx
            own = bb.black if self.color == BLACK else bb.white
            opp = bb.white if self.color == BLACK else bb.black
            flips = bb._flip_mask_for_move(own, opp, mv_bit)
            score += bin(flips).count('1') * 50

            move_scores.append((score, i, move))

        move_scores.sort(reverse=True)
        return [m for _, _, m in move_scores]

    def evaluate_bitboard(self, bb: BitBoard):
        # Opening/Mid/End phase weight via empties
        empty = 64 - bb.popcount(bb.black | bb.white)
        # Features
        # Disc differential
        my = bb.black if self.color == BLACK else bb.white
        op = bb.white if self.color == BLACK else bb.black
        disc_diff = bb.popcount(my) - bb.popcount(op)
        # Mobility
        my_moves = len(bb.get_valid_moves_bitboard(self.color))
        op_moves = len(bb.get_valid_moves_bitboard(opponent(self.color)))
        mobility = 0
        tot_moves = my_moves + op_moves
        if tot_moves:
            mobility = (my_moves - op_moves) / tot_moves
        # Corner control (check the 4 corners occupancy by masks)
        corner_idx = [(0,0),(0,7),(7,0),(7,7)]
        corners = 0
        for (cx, cy) in corner_idx:
            ci = cx*8+cy
            mask = 1 << ci
            if my & mask:
                corners += 1
            elif op & mask:
                corners -= 1
        # Combine
        score = 0
        score += corners * 1000
        score += mobility * 200
        score += disc_diff * 10
        # Phase scaling
        if empty > 50:
            score *= 1.2
        elif empty <= 20:
            score *= 0.8
        return int(score)

    # -------- Bitboard Alpha-Beta (Negamax) --------

    def alpha_beta_bb(self, bb: BitBoard, depth: int, alpha: float, beta: float, side_to_move: int, start_time: float):
        self.nodes_searched += 1

        if self.nodes_searched % 4096 == 0:
            if time.time() - start_time > self.time_limit * 0.95:
                return self.evaluate_bitboard(bb), None

        # TT probe
        key = self.bit_hash(bb, side_to_move)
        tt_entry = self.tt.get(key)
        tt_move = None
        if tt_entry and tt_entry.depth >= depth and self.bit_pos_hash(bb) == tt_entry.pos_hash:
            self.tt_hits += 1
            if tt_entry.node_type == 'exact':
                return tt_entry.score, tt_entry.move
            elif tt_entry.node_type == 'lowerbound' and tt_entry.score >= beta:
                return tt_entry.score, tt_entry.move
            elif tt_entry.node_type == 'upperbound' and tt_entry.score <= alpha:
                return tt_entry.score, tt_entry.move
            tt_move = tt_entry.move

        # Generate moves
        moves = bb.get_valid_moves_bitboard(side_to_move)

        # Terminal or leaf
        if depth == 0 or not moves:
            if not moves:
                # Pass if opponent has moves; else game over
                opp = opponent(side_to_move)
                if not bb.get_valid_moves_bitboard(opp):
                    # Terminal: final score by disc diff
                    bcnt = bb.popcount(bb.black)
                    wcnt = bb.popcount(bb.white)
                    diff = (bcnt - wcnt) if self.color == BLACK else (wcnt - bcnt)
                    return MATE_SCORE * (1 if diff > 0 else -1 if diff < 0 else 0), None
                # Pass move: do not decrease depth
                return self.alpha_beta_bb(bb, depth, alpha, beta, opp, start_time)
            return self.evaluate_bitboard(bb), None

        # Ordering
        ordered = self.enhanced_move_ordering_bb(bb, moves, depth, tt_move)

        best_move = ordered[0] if ordered else None
        best_score = float('-inf')
        original_alpha = alpha
        opp_color = opponent(side_to_move)

        for i, move in enumerate(ordered):
            x, y = move
            next_bb = bb.apply_move_bitboard(x, y, side_to_move)
            if next_bb is None:
                continue

            # PVS windowing
            if i == 0:
                score, _ = self.alpha_beta_bb(next_bb, depth - 1, -beta, -alpha, opp_color, start_time)
                score = -score
            else:
                score, _ = self.alpha_beta_bb(next_bb, depth - 1, -alpha - 1, -alpha, opp_color, start_time)
                score = -score
                if alpha < score < beta:
                    score, _ = self.alpha_beta_bb(next_bb, depth - 1, -beta, -score, opp_color, start_time)
                    score = -score

            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, score)

            if beta <= alpha:
                self.cutoffs += 1
                # killer move
                if len(self.killer_moves[depth]) >= 2:
                    self.killer_moves[depth].pop(0)
                self.killer_moves[depth].append(move)
                # history
                hx, hy = move
                self.history_table[hx][hy] += depth * depth
                break

        # TT store
        node_type = 'exact'
        if best_score <= original_alpha:
            node_type = 'upperbound'
        elif best_score >= beta:
            node_type = 'lowerbound'
        if not math.isfinite(best_score):
            best_score = MATE_SCORE if best_score > 0 else -MATE_SCORE
        if len(self.tt) >= self.max_tt_size:
            try:
                evict_key = min(self.tt, key=lambda k: (self.tt[k].age, self.tt[k].depth))
                self.tt.pop(evict_key, None)
            except ValueError:
                pass
        self.tt[key] = TTEntry(
            score=best_score,
            move=best_move,
            depth=depth,
            node_type=node_type,
            age=self.tt_age,
            pos_hash=self.bit_pos_hash(bb)
        )

        return best_score, best_move

    def _initialize_patterns(self):
        """Initialize pattern-based evaluation weights"""
        patterns = {


        # Precompute X/C → corner map for instant checks
        self.X_TO_CORNER = {(1, 1): (0, 0), (1, 6): (0, 7), (6, 1): (7, 0), (6, 6): (7, 7)}
        self.C_TO_CORNER = {
            (0, 1): (0, 0), (1, 0): (0, 0),
            (0, 6): (0, 7), (1, 7): (0, 7),
            (7, 1): (7, 0), (6, 0): (7, 0),
            (7, 6): (7, 7), (6, 7): (7, 7),
        }

        self.enable_null_move = True
        self.null_move_min_depth = 3
        self.null_move_R = 2
        self.null_move_min_empties = 14
        self.enable_futility = True
        self.futility_margin = 250
        self.futility_max_depth = 2
        self.enable_lmp = True

        # 2-ply corner avoidance tunables
        self.two_ply_check_empties = 20  # apply when empties > this
        self.two_ply_opp_limit = 6      # consider up to N opponent replies
        self.two_ply_our_limit = 6      # consider up to M our replies
    

    # ---------- init helpers ----------

    def _init_patterns(self):
        return {

            'corner_control': 1000,
            'edge_stability': 300,
            'mobility_ratio': 200,
            'disc_differential': 100,
            'frontier_discs': -50,
        }

    def _init_pos_weights(self) -> None:
        for c in CORNERS:
            adjust_position_weight(c, 500)
        for x in X_SQUARES:
            adjust_position_weight(x, -160, stages=('early', 'mid'))  # 강한 패널티
        for c in C_SQUARES:
            adjust_position_weight(c, -100, stages=('mid',))

    # ---------- hashing ----------

    def _hash(self, board, side: int) -> int:
        h = 0
        grid = board.board
        for i in range(8):
            row = grid[i]
            for j in range(8):
                v = row[j]
                if v:
                    h ^= ZOBRIST_TABLE[i][j][_pidx(v)]
        if side == WHITE:
            h ^= ZOBRIST_TURN
        return h

    # ---------- eval (kept short) ----------

    def evaluate_board_neural(self, board) -> int:
        empties = board.get_empty_count()
        if empties == 0:
            b, w = board.count_stones()
            diff = (b - w) if self.color == BLACK else (w - b)
            return 10000 if diff > 0 else -10000 if diff < 0 else 0
        my_c = sum(1 for x, y in CORNERS if board.board[x][y] == self.color)
        op_c = sum(1 for x, y in CORNERS if board.board[x][y] == opponent(self.color))
        corner_control = my_c - op_c
        my_moves = len(board.get_valid_moves(self.color))
        op_moves = len(board.get_valid_moves(opponent(self.color)))
        mobility = (my_moves - op_moves) / max(1, my_moves + op_moves)
        b, w = board.count_stones()
        disc_diff = (b - w) if self.color == BLACK else (w - b)
        frontier = board.get_frontier_count(opponent(self.color)) - board.get_frontier_count(self.color)
        pw = self.pattern_weights
        score = 0

        # Corner in center
        if pattern[4] == 1:  # Our piece in corner
            score += 100
            # Adjacent pieces
            if pattern[1] == 1: score += 50  # Edge piece
            if pattern[3] == 1: score += 50  # Edge piece
        return score

    def zobrist_hash_incremental(self, board, move=None, color=None):
        """Deprecated: kept for compatibility. Use zobrist_hash instead."""
        stm = color if color is not None else self.color
        return self.zobrist_hash(board, stm)

    def quiescence_search(self, board, alpha, beta, depth=0, max_depth=4):
        """Quiescence search to avoid horizon effect"""
        if depth >= max_depth:
            return self.evaluate_board_neural(board)
        
        stand_pat = self.evaluate_board_neural(board)
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat
        
        # Generate "quiet" moves (captures, corner moves)
        moves = board.get_valid_moves(self.color)
        quiet_moves = [m for m in moves if self._is_quiet_move(board, m)]
        
        for move in quiet_moves[:5]:  # Limit quiescence moves
            new_board = board.apply_move(*move, self.color)
            score = -self.quiescence_search(new_board, -beta, -alpha, depth + 1, max_depth)
            
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
                
        return alpha

    def _is_quiet_move(self, board, move):
        """Check if move is 'quiet' (corner, edge, or high-capture)"""
=======
        score += corner_control * pw['corner_control']
        score += mobility * pw['mobility_ratio']
        score += disc_diff * pw['disc_differential']
        score += frontier * pw['frontier_discs']
        if empties > 50:
            score *= 1.2
        elif empties <= 20:
            score *= 0.85
        return int(score)

    # ---------- corner safety helpers ----------

    def _has_corner_move(self, board, side: int) -> bool:
        return any(m in CORNERS for m in board.get_valid_moves(side))

    def _is_corner_exposing_move(self, board, move: Tuple[int, int], side: int) -> bool:

        x, y = move
        grid = board.board
        if move in CORNERS:
            return False
        c = self.X_TO_CORNER.get(move)
        if c:
            cx, cy = c
            if grid[cx][cy] == EMPTY:
                return True
        c = self.C_TO_CORNER.get(move)
        if c:
            cx, cy = c
            if grid[cx][cy] == EMPTY:
                return True
        nb = board.apply_move(x, y, side)
        opp = opponent(side)
        if self._has_corner_move(nb, opp):
            return True
        return False

    def _exposes_corner_in_two(self, board, move: Tuple[int, int], side: int,
                               opp_limit: int, our_limit: int) -> bool:
        """Conservative 2-ply trap check.
        Returns True if there exists an opponent reply after our move such that
        for all of our top replies the opponent then has an immediate corner.
        """
        x, y = move
        if move in CORNERS:
            return False
        nb = board.apply_move(x, y, side)
        opp = opponent(side)
        opp_moves = nb.get_valid_moves(opp)
        # Immediate corner already
        if any(m in CORNERS for m in opp_moves):
            return True

        # Check if move captures many pieces
        new_board = board.apply_move(x, y, self.color)
        captured = len(new_board.move_history[-1][3])
        return captured >= 3

    def late_move_reduction(self, depth, move_index, moves_count):
        """Late Move Reduction - reduce depth for later moves"""
        if depth >= 3 and move_index >= 3 and moves_count > 6:
            reduction = min(2, (move_index - 2) // 3)
            return max(1, depth - reduction)
        return depth

    def null_move_pruning(self, board, depth, beta):
        """Null move pruning for forward pruning"""
        if depth >= 3:
            # Skip turn and search with reduced depth
            R = 2 if depth > 6 else 1  # Reduction factor
            # alpha_beta_enhanced returns (score, move); extract score then negate
            score, _ = self.alpha_beta_enhanced(
                board, depth - 1 - R, -beta, -beta + 1, False, time.time()
            )
            score = -score
            if score >= beta:
                return True
        return False

    def evaluate_board_neural(self, board):
        """Neural network-inspired evaluation using patterns"""
        if board.get_empty_count() == 0:
            b, w = board.count_stones()
            diff = (b - w) if self.color == BLACK else (w - b)
            return MATE_SCORE * (1 if diff > 0 else -1 if diff < 0 else 0)

        # Prioritize opponent replies that are near empty corners or flip more
        def opp_priority(mv: Tuple[int, int]) -> int:
            px, py = mv
            s = 0
            c = self.X_TO_CORNER.get(mv)
            if c:
                cx, cy = c
                if nb.board[cx][cy] == EMPTY:
                    s += 500
            c = self.C_TO_CORNER.get(mv)
            if c:
                cx, cy = c
                if nb.board[cx][cy] == EMPTY:
                    s += 200
            s += self._estimate_flips_scan(nb.board, px, py, opp)
            return s
        opp_moves = sorted(opp_moves, key=opp_priority, reverse=True)[:opp_limit]
        for om in opp_moves:
            nb2 = nb.apply_move(om[0], om[1], opp)
            our_replies = nb2.get_valid_moves(side)
            if not our_replies:
                # we pass; if they already can corner now, it's a trap
                if self._has_corner_move(nb2, opp):
                    return True
                # else continue (not decisive)
                continue
            # Try to parry with our best replies
            def our_priority(mv: Tuple[int, int]) -> int:
                rx, ry = mv
                return self._estimate_flips_scan(nb2.board, rx, ry, side)
            parried = False
            for rm in sorted(our_replies, key=our_priority, reverse=True)[:our_limit]:
                nb3 = nb2.apply_move(rm[0], rm[1], side)
                if not self._has_corner_move(nb3, opp):
                    parried = True
                    break
            if not parried:
                return True
        return False

    def _corner_exposure_penalty(self, board, move: Tuple[int, int], side: int, empties: int) -> int:
        if self._is_corner_exposing_move(board, move, side):
            return 200000 if empties > 20 else 80000
        return 0


    # ---------- move ordering ----------

    def _static_move_value(self, board, move: Tuple[int, int]) -> int:
        x, y = move
        if move in CORNERS:
            return 20000  # massive bonus
        c = self.X_TO_CORNER.get(move)
        if c:
            cx, cy = c
            if board.board[cx][cy] == EMPTY:
                return -10000
            return 300 if board.board[cx][cy] == self.color else -200
        c = self.C_TO_CORNER.get(move)
        if c:
            cx, cy = c
            if board.board[cx][cy] == EMPTY:
                return -4000
            return -100
        if x == 0 or x == 7 or y == 0 or y == 7:
            return 150
        return 0


    def _evaluate_positional_patterns(self, board):
        """Evaluate using pre-computed patterns"""
        score = 0
        # Check 3x3 patterns around corners
        for corner_x, corner_y in CORNERS:
            pattern_score = 0
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    x, y = corner_x + dx, corner_y + dy
                    if 0 <= x < 8 and 0 <= y < 8:
                        if board.board[x][y] == self.color:
                            pattern_score += 10
                        elif board.board[x][y] == opponent(self.color):
                            pattern_score -= 10
            score += pattern_score
        return score

    def zobrist_hash(self, board, side_to_move):
        """Return Zobrist hash including side-to-move.
        Board maintains an incremental piece-hash; we XOR a turn key based on side_to_move.
        """
        base = getattr(board, '_zobrist_hash', None)
        if base is None:
            h = np.uint64(0)
            for i in range(8):
                for j in range(8):
                    piece = board.board[i][j]
                    if piece != EMPTY:
                        h ^= ZOBRIST_TABLE[i][j][piece]
            board._zobrist_hash = h
            base = h
        return np.uint64(base) ^ (ZOBRIST_TURN if side_to_move == BLACK else np.uint64(0))

    def enhanced_move_ordering(self, board, moves, depth, prev_best=None, side_to_move=None):
        """Multi-stage enhanced move ordering"""

    def _estimate_flips_scan(self, grid, x: int, y: int, side: int) -> int:
        if grid[x][y] != EMPTY:
            return 0
        opp = opponent(side)
        total = 0
        for dx, dy in DIRS:
            i, j = x + dx, y + dy
            cnt = 0
            while 0 <= i < 8 and 0 <= j < 8 and grid[i][j] == opp:
                cnt += 1; i += dx; j += dy
            if cnt and 0 <= i < 8 and 0 <= j < 8 and grid[i][j] == side:
                total += cnt
        return total

    def _order_moves(self, board, moves, depth: int, side: int,
                     prev_best: Optional[Tuple[int, int]], prev_move: Optional[Tuple[int, int]], empties: int):

        if not moves:
            return []
        grid = board.board
        key = self._hash(board, side)
        tt = self.tt.get(key)
        tt_move = tt.move if tt else None
        total_hist = sum(self.history_table) or 1


        move_scores = []
        
        for i, move in enumerate(moves):
            score = 0
            
            # Previous iteration best move
            if prev_best and move == prev_best:
                score += 10000
            
            # Transposition table move
            stm = side_to_move if side_to_move is not None else self.color
            tt_entry = self.tt.get(self.zobrist_hash(board, stm))
            if tt_entry and tt_entry.move == move:
                score += 8000
            
            # Killer moves (multiple levels)
            if depth in self.killer_moves:
                if move in self.killer_moves[depth]:
                    score += 4000 + (2 - self.killer_moves[depth].index(move)) * 1000
            
            # Counter moves
            if hasattr(board, '_last_move') and board._last_move in self.counter_moves:
                if move == self.counter_moves[board._last_move]:
                    score += 3000
            
            # History heuristic (relative)
            x, y = move
            total_history = max(1, np.sum(self.history_table))
            history_score = (self.history_table[x][y] * 2000) // total_history
            score += history_score
            
            # Static move evaluation
            score += self._static_move_value(move, board)
            
            # Capture bonus
            new_board = board.apply_move(*move, self.color)
            captured = len(new_board.move_history[-1][3])
            score += captured * 50
            
            move_scores.append((score, i, move))
        
        # Sort by score (descending)
        move_scores.sort(reverse=True)
        return [move for _, _, move in move_scores]

    def _static_move_value(self, move, board):
        """Static evaluation of move value"""
        x, y = move
        score = 0
        
        if move in CORNERS:
            score += 1000
        elif move in X_SQUARES:
            # Check if corner is occupied
            adjacent_corners = [(cx, cy) for cx, cy in CORNERS 
                              if abs(cx - x) <= 1 and abs(cy - y) <= 1]
            if any(board.board[cx][cy] != EMPTY for cx, cy in adjacent_corners):
                score += 200  # X-square is good if corner occupied
            else:
                score -= 400  # Bad if corner empty
        elif move in C_SQUARES:
            score -= 200
        elif x == 0 or x == 7 or y == 0 or y == 7:  # Edge
            score += 100
        
        return score

    def alpha_beta_enhanced(self, board, depth, alpha, beta, maximizing, start_time):
        """Enhanced alpha-beta with all modern techniques"""

        scored: List[Tuple[int, Tuple[int, int]]] = []
        for mv in moves:
            x, y = mv
            s = 0
            if prev_best and mv == prev_best:
                s += 10000
            if tt_move and mv == tt_move:
                s += 8000
            killers = self.killer_moves.get(depth, [])
            if mv in killers:
                s += 4000 + (2 - killers.index(mv)) * 1000
            if prev_move is not None and prev_move in self.counter_moves and mv == self.counter_moves[prev_move]:
                s += 3000
            s += (self.history_table[(x << 3) | y] * 2000) // total_hist
            s += self._static_move_value(board, mv)
            s -= self._corner_exposure_penalty(board, mv, side, empties)
            s += self._estimate_flips_scan(grid, x, y, side) * 50
            scored.append((s, mv))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [m for _, m in scored]

    # ---------- search ----------

    def _lmr(self, depth: int, i: int, n: int) -> int:
        if depth >= 3 and i >= 3 and n > 6:
            return min(2, (i - 2) // 3)
        return 0

    def alpha_beta_enhanced(self, board, depth: int, alpha: float, beta: float, side: int,
                             start_time: float, prev_move: Optional[Tuple[int, int]] = None):

        self.nodes_searched += 1
        if (self.nodes_searched & 0x7FF) == 0 and time.time() - start_time >= self.time_limit * 0.95:
            return self.evaluate_board_neural(board), None

        key = self._hash(board, side)
        tt = self.tt.get(key)
        if tt and tt.depth >= depth:
            if tt.node_type == 'exact':
                return tt.score, tt.move
            if tt.node_type == 'lowerbound' and tt.score >= beta:
                return tt.score, tt.move
            if tt.node_type == 'upperbound' and tt.score <= alpha:
                return tt.score, tt.move

        moves = board.get_valid_moves(side)
        empties = board.get_empty_count()
        if depth == 0:
            return self.evaluate_board_neural(board), None
        if not moves:
            opp = opponent(side)
            if not board.get_valid_moves(opp):
                return self.evaluate_board_neural(board), None

        
        # Transposition table lookup (hash includes side-to-move)
        current_color = self.color if maximizing else opponent(self.color)
        board_hash = self.zobrist_hash(board, current_color)
        tt_entry = self.tt.get(board_hash)
        tt_move = None
        
        # Collision verification by matching position hash
        if tt_entry and tt_entry.depth >= depth and getattr(board, '_zobrist_hash', None) == tt_entry.pos_hash:
            self.tt_hits += 1
            if tt_entry.node_type == 'exact':
                return tt_entry.score, tt_entry.move
            elif tt_entry.node_type == 'lowerbound' and tt_entry.score >= beta:
                return tt_entry.score, tt_entry.move
            elif tt_entry.node_type == 'upperbound' and tt_entry.score <= alpha:
                return tt_entry.score, tt_entry.move
            tt_move = tt_entry.move
        
        # Move generation for the current side
        current_color = self.color if maximizing else opponent(self.color)
        moves = board.get_valid_moves(current_color)
        
        # Terminal node handling
        if depth == 0 or not moves:
            if not moves:
                opponent_moves = board.get_valid_moves(opponent(current_color))
                if not opponent_moves:
                    return self.evaluate_board_neural(board), None
                else:
                    return self.alpha_beta_enhanced(board, depth, alpha, beta, not maximizing, start_time)
            else:
                # Quiescence search for tactical stability
                if depth == 0:
                    return self.quiescence_search(board, alpha, beta), None
                return self.evaluate_board_neural(board), None
        
        # Null move pruning (only when beta is finite to avoid returning inf)
        if not maximizing and depth >= 3 and (beta < float('inf')) and self.null_move_pruning(board, depth, beta):
            return beta, None
        
        # Enhanced move ordering
        ordered_moves = self.enhanced_move_ordering(board, moves, depth, tt_move, side_to_move=current_color)
        
        best_move = ordered_moves[0] if ordered_moves else None
        best_score = float('-inf') if maximizing else float('inf')
        original_alpha = alpha
        
        # Principal Variation Search (PVS)
        for i, move in enumerate(ordered_moves):
            # Late move reduction
            search_depth = self.late_move_reduction(depth, i, len(ordered_moves))
            
            new_board = board.apply_move(*move, current_color)
            new_board._last_move = move  # For counter-move heuristic
            

            sc, _ = self.alpha_beta_enhanced(board, depth, -beta, -alpha, opp, start_time, prev_move)
            return -sc, None

        # null-move
        if self.enable_null_move and depth >= self.null_move_min_depth and empties >= self.null_move_min_empties and moves:
            opp = opponent(side)
            R = self.null_move_R if depth > 6 else 1
            sc, _ = self.alpha_beta_enhanced(board, depth - 1 - R, -beta, -beta + 1, opp, start_time, prev_move)
            sc = -sc
            if sc >= beta:
                return sc, None

        # futility
        if self.enable_futility and depth <= self.futility_max_depth:
            st = self.evaluate_board_neural(board)
            if st + self.futility_margin <= alpha:
                return st, None

        ordered = self._order_moves(board, moves, depth, side, tt.move if tt else None, prev_move, empties)

        # Filter out corner-exposing moves (1-ply), then 2-ply traps if safe options exist
        if empties > self.two_ply_check_empties:
            safe1 = [m for m in ordered if not self._is_corner_exposing_move(board, m, side)]
            if safe1:
                # try to filter 2-ply traps; if all are traps, keep safe1
                safer2 = [m for m in safe1 if not self._exposes_corner_in_two(board, m, side,
                                                                              self.two_ply_opp_limit,
                                                                              self.two_ply_our_limit)]
                if safer2:
                    ordered = safer2
                else:
                    ordered = safe1

        best_move = ordered[0]
        best_score = float('-inf')
        a0 = alpha
        opp = opponent(side)

        lmp_cut = None
        if self.enable_lmp and depth <= 2:
            lmp_cut = 6 + depth

        for i, mv in enumerate(ordered):
            if lmp_cut is not None and i > lmp_cut and alpha > a0:
                break
            red = self._lmr(depth, i, len(ordered))
            x, y = mv
            nb = board.apply_move(x, y, side)
            child_prev = mv


            if i == 0:
                sc, _ = self.alpha_beta_enhanced(nb, depth - 1 - red, -beta, -alpha, opp, start_time, child_prev)
                sc = -sc
            else:
                sc, _ = self.alpha_beta_enhanced(nb, depth - 1 - red, -alpha - 1, -alpha, opp, start_time, child_prev)
                sc = -sc
                if alpha < sc < beta:
                    sc2, _ = self.alpha_beta_enhanced(nb, depth - 1 - red, -beta, -sc, opp, start_time, child_prev)
                    sc = -sc2

            if sc > best_score:
                best_score = sc
                best_move = mv
            if sc > alpha:
                alpha = sc

            if alpha >= beta:
                kl = self.killer_moves[depth]
                if mv not in kl:
                    if len(kl) >= 2:
                        kl.pop(0)
                    kl.append(mv)
                self.history_table[(x << 3) | y] += depth * depth
                if prev_move is not None:
                    self.counter_moves[prev_move] = mv
                break

        
        # Store in transposition table with simple replacement policy
        node_type = 'exact'
        if best_score <= original_alpha:
            node_type = 'upperbound'
        elif best_score >= beta:
            node_type = 'lowerbound'

        if not math.isfinite(best_score):
            best_score = MATE_SCORE if best_score > 0 else -MATE_SCORE

        if len(self.tt) >= self.max_tt_size:
            try:
                evict_key = min(self.tt, key=lambda k: (self.tt[k].age, self.tt[k].depth))
                self.tt.pop(evict_key, None)
            except ValueError:
                pass

        self.tt[board_hash] = TTEntry(
            score=best_score,
            move=best_move,
            depth=depth,
            node_type=node_type,
            age=self.tt_age,
            pos_hash=int(getattr(board, '_zobrist_hash', 0))
        )
        


        if len(self.tt) >= self.max_tt_size:
            threshold = self.tt_age - 2
            removed = 0
            for k in list(self.tt.keys()):
                if self.tt[k].age < threshold:
                    self.tt.pop(k); removed += 1
                if removed >= max(1, self.max_tt_size // 64):
                    break

        ntype = 'exact'
        if best_score <= a0:
            ntype = 'upperbound'
        elif best_score >= beta:
            ntype = 'lowerbound'
        self.tt[key] = TTEntry(best_score, best_move, depth, ntype, self.tt_age)

        return best_score, best_move

    # ---------- corner-first policy helpers ----------

    @staticmethod
    def _pick_corner_if_available(moves: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        for m in moves:
            if m in CORNERS:
                return m
        return None

    # ---------- endgame & ID ----------

    def perfect_endgame_solver(self, board, empties: int, side: int):
        if empties > 12:
            return None
        key = (self._hash(board, side), empties)
        if key in self.endgame_cache:
            return self.endgame_cache[key]
        moves = board.get_valid_moves(side)
        opp = opponent(side)
        if empties == 0 or (not moves and not board.get_valid_moves(opp)):
            b, w = board.count_stones()
            diff = (b - w) if self.color == BLACK else (w - b)
            res = (diff, None)
            self.endgame_cache[key] = res
            return res
        if not moves:
            sc, _ = self.perfect_endgame_solver(board, empties, opp)
            res = (-sc, None)
            self.endgame_cache[key] = res
            return res
        best_sc, best_mv = -10**9, None
        for mv in moves:
            x, y = mv
            nb = board.apply_move(x, y, side)
            sc, _ = self.perfect_endgame_solver(nb, empties - 1, opp)
            sc = -sc
            if sc > best_sc:
                best_sc, best_mv = sc, mv
        res = (best_sc, best_mv)
        self.endgame_cache[key] = res
        return res

    def iterative_deepening_ultimate(self, board):
        start = time.time()
        side = self.color
        moves = board.get_valid_moves(side)
        if not moves:
            return None
        # Corner-first: if any corner is legal now, take it immediately
        corner_now = self._pick_corner_if_available(moves)
        if corner_now is not None:
            logging.info(f"[Corner-Safe AI] Corner available → taking {corner_now}")
            return corner_now
        if len(moves) == 1:
            return moves[0]

        
        # Opening book check
        board_hash = self.zobrist_hash(board, self.color)
        if board_hash in self.opening_book:
            return self.opening_book[board_hash]
        
        logging.info(f"[Ultimate AI] Starting search: max_depth={self.max_depth}, moves={len(moves)}")
        

        h = self._hash(board, side)
        if h in self.opening_book:
            return self.opening_book[h]

        logging.info(f"[Corner-Safe AI] max_depth={self.max_depth}, moves={len(moves)}")
        best_move = None

        prev_score = 0
        window = 50
        nodes_total = 0
        for depth in range(1, self.max_depth + 1):
            self.nodes_searched = self.tt_hits = self.cutoffs = 0
            self.tt_age += 1
            try:
                if depth >= 4 and best_move is not None:
                    alpha = prev_score - window
                    beta = prev_score + window
                    sc, mv = self.alpha_beta_enhanced(board, depth, alpha, beta, side, start, None)
                    if sc <= alpha:
                        sc, mv = self.alpha_beta_enhanced(board, depth, float('-inf'), beta, side, start, None)
                    elif sc >= beta:
                        sc, mv = self.alpha_beta_enhanced(board, depth, alpha, float('inf'), side, start, None)
                    window = min(100, window + 25)
                else:

                    score, move = self.alpha_beta_enhanced(board, depth, float('-inf'), float('inf'), True, start_time)
                
                if move:
                    best_move = move
                    prev_score = score
                
                depth_time = time.time() - depth_start
                logging.info(f"Depth {depth}: {depth_time:.3f}s, {self.nodes_searched} nodes, "
                           f"TT hits: {self.tt_hits}, Score: {score}, Move: {move}")
                
                # Early termination conditions
                if math.isfinite(score) and abs(score) >= MATE_SCORE:  # Mate found
                    logging.info("Mate score - early termination")
                    break
                    
                if time.time() - start_time > self.time_limit * 0.8:
                    logging.info("Time limit approaching - stopping search")

                    sc, mv = self.alpha_beta_enhanced(board, depth, float('-inf'), float('inf'), side, start, None)
                if mv is not None:
                    best_move = mv
                    prev_score = sc
                nodes_total += self.nodes_searched
                if abs(sc) >= 9000 or time.time() - start >= self.time_limit * 0.8:

                    break
            except Exception as e:
                logging.exception(f"Error at depth {depth}: {e}")
                break

        
        total_time = time.time() - start_time
        # Estimate nodes per second using last depth's count (already logged per depth)
        nps = self.nodes_searched / max(total_time, 0.001)
        
        logging.info(f"[Ultimate AI Complete] Time: {total_time:.3f}s, "
                    f"NPS: {nps:.0f}, Best: {best_move}")
        
        return best_move

    def iterative_deepening_bitboard(self, board):
        start_time = time.time()
        bb = BitBoard(board)
        moves = bb.get_valid_moves_bitboard(self.color)
        if not moves:
            return None
        if len(moves) == 1:
            return moves[0]

        # Opening book check
        key = self.bit_hash(bb, self.color)
        if key in self.opening_book:
            return self.opening_book[key]

        logging.info(f"[BB AI] Starting search: max_depth={self.max_depth}, moves={len(moves)}")
        best_move = None
        prev_score = 0
        aspiration = 50
        for depth in range(1, self.max_depth + 1):
            try:
                self.nodes_searched = 0
                self.tt_hits = 0
                self.cutoffs = 0
                self.tt_age += 1
                depth_start = time.time()

                if depth >= 4 and best_move:
                    alpha = prev_score - aspiration
                    beta = prev_score + aspiration
                    score, move = self.alpha_beta_bb(bb, depth, alpha, beta, self.color, start_time)
                    if score <= alpha:
                        score, move = self.alpha_beta_bb(bb, depth, float('-inf'), beta, self.color, start_time)
                    elif score >= beta:
                        score, move = self.alpha_beta_bb(bb, depth, alpha, float('inf'), self.color, start_time)
                    aspiration = min(100, aspiration + 25)
                else:
                    score, move = self.alpha_beta_bb(bb, depth, float('-inf'), float('inf'), self.color, start_time)

                if move:
                    best_move = move
                    prev_score = score

                depth_time = time.time() - depth_start
                logging.info(f"[BB] Depth {depth}: {depth_time:.3f}s, {self.nodes_searched} nodes, TT: {self.tt_hits}, Score: {score}, Move: {move}")

                if math.isfinite(score) and abs(score) >= MATE_SCORE:
                    logging.info("Mate-like terminal score - stopping")
                    break
                if time.time() - start_time > self.time_limit * 0.8:
                    logging.info("Time budget nearly used - stop")
                    break
            except Exception as e:
                logging.error(f"[BB] Error at depth {depth}: {e}")
                break

        return best_move

    def perfect_endgame_solver(self, board, depth_remaining):
        """Perfect play solver for endgame"""
        if depth_remaining <= 16:  # Use perfect solver
            cache_key = (str(board), self.color, depth_remaining)
            if cache_key in self.endgame_cache:
                return self.endgame_cache[cache_key]
            
            result = self._negamax_perfect(board, self.color, depth_remaining)
            self.endgame_cache[cache_key] = result
            return result
        
        return None

    def _negamax_perfect(self, board, color, depth):
        """Perfect negamax for endgame"""
        moves = board.get_valid_moves(color)
        if not moves:
            opponent_moves = board.get_valid_moves(opponent(color))
            if not opponent_moves:
                return board.count_score(color), None
            score, _ = self._negamax_perfect(board, opponent(color), depth)
            return -score, None
        
        best_score = float('-inf')
        best_move = None
        
        for move in moves:
            new_board = board.apply_move(*move, color)
            score, _ = self._negamax_perfect(new_board, opponent(color), depth - 1)
            score = -score
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_score, best_move

    def get_move(self, board):
        """Main entry point with all optimizations"""
        empty_count = board.get_empty_count()
        
        # Perfect endgame play
        endgame_result = self.perfect_endgame_solver(board, empty_count)
        if endgame_result:
            _, move = endgame_result
            logging.info(f"Perfect endgame move: {move}")
            return move
        
        # Use bitboard search via SearchEngine
        move = self.search_engine.get_best_move(board, self.color, self.max_depth)
        if move is not None:
            return move
        # Fallback

        return best_move

    def get_move(self, board) -> Optional[Tuple[int, int]]:
        # Corner-first BEFORE any endgame shortcut
        my_moves = board.get_valid_moves(self.color)
        corner_now = self._pick_corner_if_available(my_moves)
        if corner_now is not None:
            logging.info(f"[Corner-Safe AI] Corner available at root → taking {corner_now}")
            return corner_now
        empties = board.get_empty_count()
        end = self.perfect_endgame_solver(board, empties, self.color)
        if end is not None and end[1] is not None:
            logging.info(f"Perfect endgame move: {end[1]}")
            return end[1]
 main
        return self.iterative_deepening_ultimate(board)
