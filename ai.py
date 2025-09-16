from functools import lru_cache
import multiprocessing
import time
import random
import threading
from collections import defaultdict, deque
from constants import adjust_position_weight
from constants import BLACK, WHITE, EMPTY, opponent, EARLY_WEIGHTS, MID_WEIGHTS, LATE_WEIGHTS, CORNERS, X_SQUARES, C_SQUARES
from board import Board
import logging
import numpy as np
import pickle
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

@dataclass
class TTEntry:
    """Transposition Table Entry with enhanced information"""
    score: float
    move: Optional[Tuple[int, int]]
    depth: int
    node_type: str  # 'exact', 'lowerbound', 'upperbound'
    age: int
    pos_hash: int
    
class BitBoard:
    """Bit-based board representation for ultra-fast operations"""
    def __init__(self, board=None):
        if board:
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
                if not BitBoard(bb).get_valid_moves_bitboard(opp):
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
    def __init__(self, color, difficulty='hard', time_limit=10.0):
        self.color = color
        self.difficulty = difficulty
        self.time_limit = time_limit
        
        # Enhanced transposition table with replacement scheme
        self.tt = {}
        self.tt_age = 0
        self.max_tt_size = 2**20  # 1M entries
        
        # Multi-tier move ordering
        self.killer_moves = defaultdict(list)
        self.counter_moves = {}
        self.history_table = np.zeros((8, 8), dtype=np.int32)
        self.butterfly_table = np.zeros((8, 8), dtype=np.int32)
        
        # Search statistics
        self.nodes_searched = 0
        self.tt_hits = 0
        self.cutoffs = 0
        
        # Neural network simulation (pattern-based evaluation)
        self.pattern_weights = self._initialize_patterns()
        
        # Parallel search setup
        self.use_parallel = True
        self.max_workers = min(4, multiprocessing.cpu_count())
        
        # Opening book
        self.opening_book = self._load_opening_book()
        
        # Endgame tablebase simulation
        self.endgame_cache = {}
        
        # Initialize position weights with symmetry
        self._initialize_weights()
        
        # Difficulty settings with advanced features
        if difficulty == 'easy':
            self.max_depth = 6
            self.use_parallel = False
            self.selective_depth = 8
        elif difficulty == 'medium':
            self.max_depth = 8
            self.selective_depth = 12
        else:  # hard
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
                if not BitBoard(bb).get_valid_moves_bitboard(opp):
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
            'corner_control': 1000,
            'edge_stability': 300,
            'mobility_ratio': 200,
            'disc_differential': 100,
            'potential_mobility': 150,
            'frontier_discs': -50,
            'corner_proximity': -200,
            'wedge_pattern': 400,
            'diagonal_control': 250
        }
        return patterns

    def _load_opening_book(self):
        """Load pre-computed opening moves"""
        # Simulated opening book - in practice, load from file
        book = {
            # Opening position hashes -> best moves
        }
        return book

    def _initialize_weights(self):
        """Initialize all position weights with enhanced values"""
        for corner in CORNERS:
            adjust_position_weight(corner, 500)  # Increased corner value
        for x in X_SQUARES:
            adjust_position_weight(x, -120, stages=('early', 'mid'))
        for c in C_SQUARES:
            adjust_position_weight(c, -60, stages=('mid',))

    def _precompute_patterns(self):
        """Pre-compute common board patterns for instant lookup"""
        self.pattern_cache = {}
        # Pre-compute common 3x3 patterns around corners
        for i in range(512):  # 2^9 possible patterns
            pattern = []
            for j in range(9):
                pattern.append((i >> j) & 1)
            self.pattern_cache[i] = self._evaluate_pattern(pattern)

    def _evaluate_pattern(self, pattern):
        """Evaluate a 3x3 pattern"""
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
        x, y = move
        if (x, y) in CORNERS:
            return True
        if x == 0 or x == 7 or y == 0 or y == 7:
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

        score = 0
        empty_count = board.get_empty_count()
        
        # Multi-layer evaluation inspired by neural networks
        features = self._extract_features(board)
        
        # Pattern-based scoring
        score += features['corner_control'] * self.pattern_weights['corner_control']
        score += features['mobility'] * self.pattern_weights['mobility_ratio'] 
        score += features['stability'] * self.pattern_weights['edge_stability']
        score += features['disc_diff'] * self.pattern_weights['disc_differential']
        score += features['frontier'] * self.pattern_weights['frontier_discs']
        
        # Game phase adjustment
        phase_multiplier = self._get_phase_multiplier(empty_count)
        score *= phase_multiplier
        
        # Add positional bonuses
        score += self._evaluate_positional_patterns(board)
        
        return int(score)

    def _extract_features(self, board):
        """Extract features for neural-style evaluation"""
        features = {}
        
        # Corner control
        my_corners = sum(1 for x, y in CORNERS if board.board[x][y] == self.color)
        opp_corners = sum(1 for x, y in CORNERS if board.board[x][y] == opponent(self.color))
        features['corner_control'] = my_corners - opp_corners
        
        # Mobility
        my_moves = len(board.get_valid_moves(self.color))
        opp_moves = len(board.get_valid_moves(opponent(self.color)))
        features['mobility'] = (my_moves - opp_moves) / max(1, my_moves + opp_moves)
        
        # Disc differential
        b, w = board.count_stones()
        features['disc_diff'] = (b - w) if self.color == BLACK else (w - b)
        
        # Stability (simplified)
        features['stability'] = self._count_stable_discs(board)
        
        # Frontier discs
        features['frontier'] = board.get_frontier_count(opponent(self.color)) - board.get_frontier_count(self.color)
        
        return features

    def _count_stable_discs(self, board):
        """Fast stable disc counting"""
        stable_count = 0
        # Only check corners and edges for speed
        for x, y in CORNERS:
            if board.board[x][y] == self.color:
                stable_count += 3
            elif board.board[x][y] == opponent(self.color):
                stable_count -= 3
        return stable_count

    def _get_phase_multiplier(self, empty_count):
        """Get phase-based multiplier for evaluation"""
        if empty_count > 50:
            return 1.2  # Opening
        elif empty_count > 20:
            return 1.0  # Midgame
        else:
            return 0.8  # Endgame

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
        if not moves:
            return []

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
        self.nodes_searched += 1
        
        # Time management
        if self.nodes_searched % 2048 == 0:
            if time.time() - start_time > self.time_limit * 0.95:
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
            
            if i == 0:
                # Principal variation - full window
                score, _ = self.alpha_beta_enhanced(
                    new_board, search_depth - 1, -beta, -alpha, not maximizing, start_time
                )
                score = -score
            else:
                # Zero-window search
                score, _ = self.alpha_beta_enhanced(
                    new_board, search_depth - 1, -alpha - 1, -alpha, not maximizing, start_time
                )
                score = -score
                
                # Re-search if needed
                if alpha < score < beta:
                    score, _ = self.alpha_beta_enhanced(
                        new_board, search_depth - 1, -beta, -score, not maximizing, start_time
                    )
                    score = -score
            
            if maximizing:
                if score > best_score:
                    best_score = score
                    best_move = move
                alpha = max(alpha, score)
            else:
                if score < best_score:
                    best_score = score
                    best_move = move
                beta = min(beta, score)
            
            # Beta cutoff
            if beta <= alpha:
                self.cutoffs += 1
                # Update killer moves
                if len(self.killer_moves[depth]) >= 2:
                    self.killer_moves[depth].pop(0)
                self.killer_moves[depth].append(move)
                
                # Update history table
                x, y = move
                self.history_table[x][y] += depth * depth
                
                # Update counter moves
                if hasattr(board, '_last_move'):
                    self.counter_moves[board._last_move] = move
                
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
        
        return best_score, best_move

    def iterative_deepening_ultimate(self, board):
        """Ultimate iterative deepening with all enhancements"""
        start_time = time.time()
        best_move = None
        moves = board.get_valid_moves(self.color)
        
        if not moves:
            return None
        if len(moves) == 1:
            return moves[0]
        
        # Opening book check
        board_hash = self.zobrist_hash(board, self.color)
        if board_hash in self.opening_book:
            return self.opening_book[board_hash]
        
        logging.info(f"[Ultimate AI] Starting search: max_depth={self.max_depth}, moves={len(moves)}")
        
        prev_score = 0
        aspiration_window = 50
        
        for depth in range(1, self.max_depth + 1):
            try:
                self.nodes_searched = 0
                self.tt_hits = 0
                self.cutoffs = 0
                self.tt_age += 1
                
                depth_start = time.time()
                
                # Aspiration window search for deeper searches
                if depth >= 4 and best_move:
                    alpha = prev_score - aspiration_window
                    beta = prev_score + aspiration_window
                    
                    score, move = self.alpha_beta_enhanced(board, depth, alpha, beta, True, start_time)
                    
                    # Research if outside window
                    if score <= alpha:
                        score, move = self.alpha_beta_enhanced(board, depth, float('-inf'), beta, True, start_time)
                    elif score >= beta:
                        score, move = self.alpha_beta_enhanced(board, depth, alpha, float('inf'), True, start_time)
                        
                    aspiration_window = min(100, aspiration_window + 25)
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
                    break
                    
            except Exception as e:
                logging.error(f"Error at depth {depth}: {e}")
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
        return self.iterative_deepening_ultimate(board)
