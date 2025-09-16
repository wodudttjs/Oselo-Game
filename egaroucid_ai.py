import time
import random
import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set
import threading
import json

from constants import BLACK, WHITE, EMPTY, opponent, CORNERS, X_SQUARES, C_SQUARES
from board import Board

@dataclass
class TTEntry:
    """Transposition Table Entry"""
    depth: int
    score: int
    flag: str  # 'EXACT', 'ALPHA', 'BETA'
    best_move: Optional[Tuple[int, int]]
    age: int

@dataclass
class SearchResult:
    """Search Result Structure"""
    score: int
    best_move: Optional[Tuple[int, int]]
    depth: int
    nodes: int
    time_ms: int
    is_exact: bool
    pv: List[Tuple[int, int]]  # Principal Variation

class OpeningBook:
    """Simple Opening Book Implementation"""
    
    def __init__(self):
        # 기본 오프닝 패턴들 (간소화된 버전)
        self.book = {
            # 초기 4수 이후의 좋은 수들
            'standard_opening': {
                frozenset([(3,3,'W'), (3,4,'B'), (4,3,'B'), (4,4,'W')]): [
                    ((2,3), 0.5), ((3,2), 0.5), ((4,5), 0.3), ((5,4), 0.3)
                ],
                frozenset([(3,3,'W'), (3,4,'B'), (4,3,'B'), (4,4,'W'), (2,3,'B')]): [
                    ((3,2), 0.7), ((1,3), 0.4), ((2,2), -0.2)
                ]
            }
        }
    
    def get_move(self, board: Board) -> Optional[Tuple[int, int]]:
        """Get move from opening book"""
        board_state = self._board_to_state(board)
        
        for pattern_name, patterns in self.book.items():
            for pattern, moves in patterns.items():
                if self._matches_pattern(board_state, pattern):
                    # 가중치 기반 선택
                    best_moves = [move for move, weight in moves if weight > 0]
                    if best_moves:
                        return random.choice(best_moves)
        return None
    
    def _board_to_state(self, board: Board) -> Set:
        """Convert board to hashable state"""
        state = set()
        for i in range(8):
            for j in range(8):
                if board.board[i][j] != EMPTY:
                    color = 'B' if board.board[i][j] == BLACK else 'W'
                    state.add((i, j, color))
        return frozenset(state)
    
    def _matches_pattern(self, board_state: frozenset, pattern: frozenset) -> bool:
        """Check if board matches pattern"""
        return pattern.issubset(board_state)

class EgaroucidStyleAI:
    """Egaroucid-inspired Othello AI with advanced techniques"""
    
    def __init__(self, color, difficulty='hard', time_limit=5.0):
        self.color = color
        self.difficulty = difficulty
        self.time_limit = time_limit
        
        # Transposition Table
        self.tt = {}
        self.tt_age = 0
        self.max_tt_size = 1000000  # 1M entries
        
        # Opening Book
        self.opening_book = OpeningBook()
        
        # Search statistics
        self.nodes_searched = 0
        self.tt_hits = 0
        self.cutoffs = 0
        
        # Killer moves heuristic
        self.killer_moves = defaultdict(list)
        self.history_table = defaultdict(int)
        
        # Multi-ProbCut parameters (간소화된 버전)
        self.mpc_levels = {
            'early': {'depth_reduction': 4, 'threshold': 0.85},
            'mid': {'depth_reduction': 6, 'threshold': 0.90},
            'late': {'depth_reduction': 8, 'threshold': 0.95}
        }
        
        # Difficulty-based parameters
        if difficulty == 'easy':
            self.max_depth = 6
            self.time_limit = min(time_limit, 2.0)
            self.use_tt = False
            self.use_mpc = False
        elif difficulty == 'medium':
            self.max_depth = 10
            self.time_limit = min(time_limit, 4.0)
            self.use_tt = True
            self.use_mpc = False
        else:  # hard
            self.max_depth = 14
            self.use_tt = True
            self.use_mpc = True
    
    def get_board_hash(self, board: Board) -> str:
        """Get unique hash for board position"""
        board_str = ''.join(str(cell) for row in board.board for cell in row)
        return hashlib.md5(board_str.encode()).hexdigest()
    
    def store_tt(self, board_hash: str, depth: int, score: int, flag: str, best_move: Optional[Tuple[int, int]]):
        """Store position in transposition table"""
        if not self.use_tt:
            return
            
        # Clear old entries if table is full
        if len(self.tt) >= self.max_tt_size:
            self.clear_old_tt_entries()
        
        self.tt[board_hash] = TTEntry(depth, score, flag, best_move, self.tt_age)
    
    def probe_tt(self, board_hash: str, depth: int, alpha: int, beta: int) -> Optional[int]:
        """Probe transposition table"""
        if not self.use_tt or board_hash not in self.tt:
            return None
        
        entry = self.tt[board_hash]
        if entry.depth >= depth:
            self.tt_hits += 1
            if entry.flag == 'EXACT':
                return entry.score
            elif entry.flag == 'ALPHA' and entry.score <= alpha:
                return alpha
            elif entry.flag == 'BETA' and entry.score >= beta:
                return beta
        
        return None
    
    def clear_old_tt_entries(self):
        """Clear old transposition table entries"""
        old_entries = [key for key, entry in self.tt.items() 
                      if self.tt_age - entry.age > 4]
        for key in old_entries[:len(old_entries)//2]:  # Remove half of old entries
            del self.tt[key]
    
    def evaluate_position(self, board: Board) -> int:
        """Advanced position evaluation with game phase detection"""
        empty_count = board.get_empty_count()
        
        # Game phase detection
        if empty_count > 50:
            return self.evaluate_early_game(board)
        elif empty_count > 20:
            return self.evaluate_mid_game(board)
        else:
            return self.evaluate_end_game(board)
    
    def evaluate_early_game(self, board: Board) -> int:
        """Early game evaluation - focus on mobility and position"""
        score = 0
        
        # Mobility (가장 중요)
        my_moves = len(board.get_valid_moves(self.color))
        opp_moves = len(board.get_valid_moves(opponent(self.color)))
        if my_moves + opp_moves > 0:
            score += 100 * (my_moves - opp_moves) / (my_moves + opp_moves + 1)
        
        # Corner control
        my_corners = sum(1 for x, y in CORNERS if board.board[x][y] == self.color)
        opp_corners = sum(1 for x, y in CORNERS if board.board[x][y] == opponent(self.color))
        score += 300 * (my_corners - opp_corners)
        
        # Avoid X-squares near empty corners
        for corner_x, corner_y in CORNERS:
            if board.board[corner_x][corner_y] == EMPTY:
                for x, y in X_SQUARES:
                    if abs(x - corner_x) <= 1 and abs(y - corner_y) <= 1:
                        if board.board[x][y] == self.color:
                            score -= 150
                        elif board.board[x][y] == opponent(self.color):
                            score += 150
        
        # Disc count (낮은 가중치)
        b, w = board.count_stones()
        disc_diff = (b - w) if self.color == BLACK else (w - b)
        score += 10 * disc_diff
        
        return score
    
    def evaluate_mid_game(self, board: Board) -> int:
        """Mid game evaluation - balanced approach"""
        score = 0
        
        # Mobility
        my_moves = len(board.get_valid_moves(self.color))
        opp_moves = len(board.get_valid_moves(opponent(self.color)))
        if my_moves + opp_moves > 0:
            score += 80 * (my_moves - opp_moves) / (my_moves + opp_moves + 1)
        
        # Stability
        score += 40 * self.evaluate_stability(board)
        
        # Corner control
        my_corners = sum(1 for x, y in CORNERS if board.board[x][y] == self.color)
        opp_corners = sum(1 for x, y in CORNERS if board.board[x][y] == opponent(self.color))
        score += 200 * (my_corners - opp_corners)
        
        # Edge control
        score += 20 * self.evaluate_edges(board)
        
        # Frontier discs (fewer is better)
        my_frontier = board.get_frontier_count(self.color)
        opp_frontier = board.get_frontier_count(opponent(self.color))
        score += 15 * (opp_frontier - my_frontier)
        
        # Disc count
        b, w = board.count_stones()
        disc_diff = (b - w) if self.color == BLACK else (w - b)
        score += 20 * disc_diff
        
        return score
    
    def evaluate_end_game(self, board: Board) -> int:
        """End game evaluation - focus on disc count and stability"""
        if board.get_empty_count() == 0:
            b, w = board.count_stones()
            diff = (b - w) if self.color == BLACK else (w - b)
            return 10000 * (1 if diff > 0 else -1 if diff < 0 else 0)
        
        score = 0
        
        # Disc count (매우 중요)
        b, w = board.count_stones()
        disc_diff = (b - w) if self.color == BLACK else (w - b)
        score += 100 * disc_diff
        
        # Stability
        score += 80 * self.evaluate_stability(board)
        
        # Corner control
        my_corners = sum(1 for x, y in CORNERS if board.board[x][y] == self.color)
        opp_corners = sum(1 for x, y in CORNERS if board.board[x][y] == opponent(self.color))
        score += 150 * (my_corners - opp_corners)
        
        # Mobility (여전히 중요)
        my_moves = len(board.get_valid_moves(self.color))
        opp_moves = len(board.get_valid_moves(opponent(self.color)))
        if my_moves + opp_moves > 0:
            score += 60 * (my_moves - opp_moves) / (my_moves + opp_moves + 1)
        
        return score
    
    def evaluate_stability(self, board: Board) -> int:
        """Evaluate disc stability"""
        my_stable = 0
        opp_stable = 0
        
        for i in range(8):
            for j in range(8):
                if board.board[i][j] == self.color and board.is_stable(i, j):
                    my_stable += 1
                elif board.board[i][j] == opponent(self.color) and board.is_stable(i, j):
                    opp_stable += 1
        
        return my_stable - opp_stable
    
    def evaluate_edges(self, board: Board) -> int:
        """Evaluate edge control"""
        my_edges = 0
        opp_edges = 0
        
        # Top and bottom edges
        for j in range(8):
            if board.board[0][j] == self.color:
                my_edges += 1
            elif board.board[0][j] == opponent(self.color):
                opp_edges += 1
            if board.board[7][j] == self.color:
                my_edges += 1
            elif board.board[7][j] == opponent(self.color):
                opp_edges += 1
        
        # Left and right edges  
        for i in range(8):
            if board.board[i][0] == self.color:
                my_edges += 1
            elif board.board[i][0] == opponent(self.color):
                opp_edges += 1
            if board.board[i][7] == self.color:
                my_edges += 1
            elif board.board[i][7] == opponent(self.color):
                opp_edges += 1
        
        return my_edges - opp_edges
    
    def multi_prob_cut(self, board: Board, depth: int, alpha: int, beta: int) -> Optional[int]:
        """Simplified Multi-ProbCut implementation"""
        if not self.use_mpc or depth < 8:
            return None
        
        empty_count = board.get_empty_count()
        
        # Select MPC parameters based on game phase
        if empty_count > 45:
            mpc_params = self.mpc_levels['early']
        elif empty_count > 20:
            mpc_params = self.mpc_levels['mid']
        else:
            mpc_params = self.mpc_levels['late']
        
        reduced_depth = depth - mpc_params['depth_reduction']
        if reduced_depth <= 0:
            return None
        
        # Shallow search
        shallow_score, _ = self.negamax(board, reduced_depth, alpha, beta, 
                                       True, time.time() + 0.1)  # Quick search
        
        # Estimate full-depth score
        threshold = mpc_params['threshold']
        
        if shallow_score >= beta * threshold:
            return beta  # Beta cutoff
        elif shallow_score <= alpha * threshold:
            return alpha  # Alpha cutoff
        
        return None
    
    def order_moves(self, board: Board, moves: List[Tuple[int, int]], depth: int) -> List[Tuple[int, int]]:
        """Advanced move ordering"""
        if not moves:
            return moves
        
        move_scores = []
        board_hash = self.get_board_hash(board)
        
        # Get TT move if available
        tt_move = None
        if self.use_tt and board_hash in self.tt:
            tt_move = self.tt[board_hash].best_move
        
        for move in moves:
            x, y = move
            score = 0
            
            # TT move gets highest priority
            if move == tt_move:
                score += 10000
            
            # Killer moves
            if move in self.killer_moves.get(depth, []):
                score += 1000
            
            # History heuristic
            score += self.history_table.get(move, 0)
            
            # Static position evaluation
            if move in CORNERS:
                score += 500
            elif move in X_SQUARES:
                # Check if adjacent corner is safe
                corner_safe = True
                for corner in CORNERS:
                    if abs(corner[0] - x) <= 1 and abs(corner[1] - y) <= 1:
                        if board.board[corner[0]][corner[1]] == EMPTY:
                            corner_safe = False
                            break
                if not corner_safe:
                    score -= 300
            elif move in C_SQUARES:
                score -= 150
            elif x == 0 or x == 7 or y == 0 or y == 7:  # Edge
                score += 100
            
            # Quick mobility evaluation
            new_board = board.apply_move(x, y, self.color)
            opp_moves_after = len(new_board.get_valid_moves(opponent(self.color)))
            score -= opp_moves_after * 5  # Prefer moves that reduce opponent mobility
            
            move_scores.append((score, move))
        
        move_scores.sort(reverse=True)
        return [move for _, move in move_scores]
    
    def negamax(self, board: Board, depth: int, alpha: int, beta: int, 
                maximizing: bool, end_time: float) -> Tuple[int, Optional[Tuple[int, int]]]:
        """Enhanced Negamax with advanced pruning techniques"""
        self.nodes_searched += 1
        
        # Time check
        if time.time() > end_time:
            return self.evaluate_position(board), None
        
        board_hash = self.get_board_hash(board)
        
        # TT probe
        tt_score = self.probe_tt(board_hash, depth, alpha, beta)
        if tt_score is not None:
            return tt_score, None
        
        current_color = self.color if maximizing else opponent(self.color)
        moves = board.get_valid_moves(current_color)
        
        # Terminal conditions
        if depth == 0 or not moves:
            if not moves:
                opponent_moves = board.get_valid_moves(opponent(current_color))
                if not opponent_moves:
                    # Game over
                    return self.evaluate_position(board), None
                else:
                    # Pass turn
                    return self.negamax(board, depth, alpha, beta, not maximizing, end_time)
            else:
                return self.evaluate_position(board), None
        
        # Multi-ProbCut
        if maximizing:
            mpc_result = self.multi_prob_cut(board, depth, alpha, beta)
            if mpc_result is not None:
                return mpc_result, None
        
        # Move ordering
        ordered_moves = self.order_moves(board, moves, depth)
        best_move = None
        original_alpha = alpha
        
        if maximizing:
            max_score = float('-inf')
            for i, move in enumerate(ordered_moves):
                new_board = board.apply_move(*move, current_color)
                score, _ = self.negamax(new_board, depth - 1, alpha, beta, False, end_time)
                
                if score > max_score:
                    max_score = score
                    best_move = move
                
                alpha = max(alpha, score)
                if beta <= alpha:
                    # Beta cutoff
                    self.cutoffs += 1
                    if len(self.killer_moves[depth]) >= 2:
                        self.killer_moves[depth].pop(0)
                    self.killer_moves[depth].append(move)
                    break
            
            # Update history table
            if best_move:
                self.history_table[best_move] += depth * depth
            
            # Store in TT
            flag = 'EXACT' if original_alpha < max_score < beta else ('BETA' if max_score >= beta else 'ALPHA')
            self.store_tt(board_hash, depth, max_score, flag, best_move)
            
            return max_score, best_move
        else:
            min_score = float('inf')
            for move in ordered_moves:
                new_board = board.apply_move(*move, current_color)
                score, _ = self.negamax(new_board, depth - 1, alpha, beta, True, end_time)
                
                if score < min_score:
                    min_score = score
                    best_move = move
                
                beta = min(beta, score)
                if beta <= alpha:
                    self.cutoffs += 1
                    if len(self.killer_moves[depth]) >= 2:
                        self.killer_moves[depth].pop(0)
                    self.killer_moves[depth].append(move)
                    break
            
            if best_move:
                self.history_table[best_move] += depth * depth
            
            flag = 'EXACT' if alpha < min_score < original_alpha else ('ALPHA' if min_score <= alpha else 'BETA')
            self.store_tt(board_hash, depth, min_score, flag, best_move)
            
            return min_score, best_move
    
    def iterative_deepening(self, board: Board) -> SearchResult:
        """Iterative deepening with time management"""
        start_time = time.time()
        end_time = start_time + self.time_limit
        
        moves = board.get_valid_moves(self.color)
        if not moves:
            return SearchResult(0, None, 0, 0, 0, True, [])
        
        if len(moves) == 1:
            return SearchResult(0, moves[0], 1, 1, 1, False, [moves[0]])
        
        best_move = moves[0]
        best_score = float('-inf')
        pv = []
        
        # Iterative deepening
        for depth in range(1, self.max_depth + 1):
            try:
                score, move = self.negamax(board, depth, float('-inf'), float('inf'), 
                                         True, end_time)
                
                if move:
                    best_move = move
                    best_score = score
                
                # Time management
                elapsed = time.time() - start_time
                if elapsed > self.time_limit * 0.85:
                    break
                    
                # Complete search detection
                if depth >= board.get_empty_count():
                    break
                    
            except:
                break
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        return SearchResult(
            score=best_score,
            best_move=best_move,
            depth=depth,
            nodes=self.nodes_searched,
            time_ms=elapsed_ms,
            is_exact=(depth >= board.get_empty_count()),
            pv=[best_move] if best_move else []
        )
    
    def get_move(self, board: Board) -> Optional[Tuple[int, int]]:
        """Get best move using enhanced search"""
        self.nodes_searched = 0
        self.tt_hits = 0
        self.cutoffs = 0
        self.tt_age += 1
        
        # Try opening book first
        if board.get_empty_count() > 52:
            book_move = self.opening_book.get_move(board)
            if book_move and board.is_valid_move(*book_move, self.color):
                return book_move
        
        # Main search
        result = self.iterative_deepening(board)
        
        # Print search statistics (optional)
        if result.time_ms > 100:  # Only for longer searches
            print(f"Search: depth={result.depth}, nodes={result.nodes}, "
                  f"time={result.time_ms}ms, tt_hits={self.tt_hits}, "
                  f"cutoffs={self.cutoffs}, score={result.score}")
        
        return result.best_move