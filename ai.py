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

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')

# Enhanced Zobrist hashing with incremental updates
ZOBRIST_TABLE = np.random.randint(1, 2**63, size=(8, 8, 3), dtype=np.uint64)
ZOBRIST_TURN = np.random.randint(1, 2**63, dtype=np.uint64)

@dataclass
class TTEntry:
    """Transposition Table Entry with enhanced information"""
    score: float
    move: Optional[Tuple[int, int]]
    depth: int
    node_type: str  # 'exact', 'lowerbound', 'upperbound'
    age: int
    
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
        else:
            self.black = 0x0000001008000000  # Initial black positions
            self.white = 0x0000000810000000  # Initial white positions
    
    def get_empty_mask(self):
        return ~(self.black | self.white) & 0xFFFFFFFFFFFFFFFF
    
    def popcount(self, mask):
        return bin(mask).count('1')

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
        """Incremental zobrist hashing for speed"""
        if not hasattr(board, '_zobrist_hash'):
            h = np.uint64(0)
            for i in range(8):
                for j in range(8):
                    piece = board.board[i][j]
                    if piece != EMPTY:
                        h ^= ZOBRIST_TABLE[i][j][piece]
            board._zobrist_hash = h
        
        if move and color:
            # Update hash incrementally
            x, y = move
            board._zobrist_hash ^= ZOBRIST_TABLE[x][y][color]
            
        return board._zobrist_hash

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
        captured = abs(new_board.count_stones()[0] - board.count_stones()[0] - 1)
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
            score = -self.alpha_beta_enhanced(board, depth - 1 - R, -beta, -beta + 1, False, time.time())
            if score >= beta:
                return True
        return False

    def evaluate_board_neural(self, board):
        """Neural network-inspired evaluation using patterns"""
        if board.get_empty_count() == 0:
            b, w = board.count_stones()
            diff = (b - w) if self.color == BLACK else (w - b)
            return 10000 * (1 if diff > 0 else -1 if diff < 0 else 0)

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

    def enhanced_move_ordering(self, board, moves, depth, prev_best=None):
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
            tt_entry = self.tt.get(self.zobrist_hash_incremental(board))
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
            captured = abs(new_board.count_stones()[0] - board.count_stones()[0] - 1)
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
        
        # Transposition table lookup
        board_hash = self.zobrist_hash_incremental(board)
        tt_entry = self.tt.get(board_hash)
        tt_move = None
        
        if tt_entry and tt_entry.depth >= depth:
            self.tt_hits += 1
            if tt_entry.node_type == 'exact':
                return tt_entry.score, tt_entry.move
            elif tt_entry.node_type == 'lowerbound' and tt_entry.score >= beta:
                return tt_entry.score, tt_entry.move
            elif tt_entry.node_type == 'upperbound' and tt_entry.score <= alpha:
                return tt_entry.score, tt_entry.move
            tt_move = tt_entry.move
        
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
        
        # Null move pruning
        if not maximizing and depth >= 3 and self.null_move_pruning(board, depth, beta):
            return beta, None
        
        # Enhanced move ordering
        ordered_moves = self.enhanced_move_ordering(board, moves, depth, tt_move)
        
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
        
        # Store in transposition table
        if len(self.tt) < self.max_tt_size:
            node_type = 'exact'
            if best_score <= original_alpha:
                node_type = 'upperbound'
            elif best_score >= beta:
                node_type = 'lowerbound'
            
            self.tt[board_hash] = TTEntry(
                score=best_score,
                move=best_move,
                depth=depth,
                node_type=node_type,
                age=self.tt_age
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
        board_hash = self.zobrist_hash_incremental(board)
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
                if abs(score) > 9000:  # Mate found
                    logging.info("Mate score - early termination")
                    break
                    
                if time.time() - start_time > self.time_limit * 0.8:
                    logging.info("Time limit approaching - stopping search")
                    break
                    
            except Exception as e:
                logging.error(f"Error at depth {depth}: {e}")
                break
        
        total_time = time.time() - start_time
        total_nodes = sum(self.nodes_searched for _ in range(depth))
        nps = total_nodes / max(total_time, 0.001)
        
        logging.info(f"[Ultimate AI Complete] Time: {total_time:.3f}s, "
                    f"NPS: {nps:.0f}, Best: {best_move}")
        
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
        
        # Use ultimate iterative deepening
        return self.iterative_deepening_ultimate(board)