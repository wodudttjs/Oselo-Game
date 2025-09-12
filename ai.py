from functools import lru_cache
import multiprocessing
import time
import random
from collections import defaultdict
from constants import adjust_position_weight
from constants import BLACK, WHITE, EMPTY, opponent, EARLY_WEIGHTS, MID_WEIGHTS, LATE_WEIGHTS, CORNERS, X_SQUARES, C_SQUARES
from board import Board
import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
import numpy as np

# Zobrist hashing for faster board state comparison
ZOBRIST_TABLE = np.random.randint(1, 2**63, size=(8, 8, 3), dtype=np.uint64)

def zobrist_hash(board):
    """Fast zobrist hashing for board states"""
    h = np.uint64(0)
    for i in range(8):
        for j in range(8):
            color = board.board[i][j]
            if color != EMPTY:
                h ^= ZOBRIST_TABLE[i][j][color]
    return h

def adjust_position_weight(pos, value, stages=('early', 'mid', 'late')):
    x, y = pos
    if 'early' in stages:
        for dx, dy in [(x, y), (7-x, y), (x, 7-y), (7-x, 7-y)]:
            EARLY_WEIGHTS[dx][dy] = value
    if 'mid' in stages:
        for dx, dy in [(x, y), (7-x, y), (x, 7-y), (7-x, 7-y)]:
            MID_WEIGHTS[dx][dy] = value
    if 'late' in stages:
        for dx, dy in [(x, y), (7-x, y), (x, 7-y), (7-x, 7-y)]:
            LATE_WEIGHTS[dx][dy] = value

class AdvancedAI:
    def __init__(self, color, difficulty='hard', time_limit=3.0):
        self.color = color
        self.difficulty = difficulty
        self.time_limit = time_limit
        self.transposition_table = {}
        self.killer_moves = defaultdict(list)
        self.history_table = defaultdict(int)
        self.nodes_searched = 0
        
        # Pre-compute corner adjacency for faster X-square evaluation
        self.corner_adjacency = {}
        for x in range(8):
            for y in range(8):
                adjacent_corners = []
                for corner in CORNERS:
                    if abs(corner[0] - x) <= 1 and abs(corner[1] - y) <= 1:
                        adjacent_corners.append(corner)
                self.corner_adjacency[(x, y)] = adjacent_corners
        
        # Initialize position weights
        for corner in CORNERS:
            adjust_position_weight(corner, 300)
        for x in X_SQUARES:
            adjust_position_weight(x, -90)
        for c in C_SQUARES:
            adjust_position_weight(c, -40, stages=('mid',))
        
        # Optimized difficulty settings
        if difficulty == 'easy':
            self.max_depth = 4
            self.time_limit = 2.0
        elif difficulty == 'medium':
            self.max_depth = 6
            self.time_limit = 5.0
        else:  # hard
            self.max_depth = 8
            self.time_limit = 10.0
        
        # Cache for frequently accessed values
        self._position_cache = {}
        self._mobility_cache = {}
        
    def get_weights_for_stage(self, board):
        """Get evaluation weights based on game stage - cached"""
        empty_count = board.get_empty_count()
        if empty_count > 45:
            return EARLY_WEIGHTS
        elif empty_count > 20:
            return MID_WEIGHTS
        else:
            return LATE_WEIGHTS
    
    def get_game_stage_weights(self, empty_count):
        """Get weights for different evaluation components based on game stage - optimized"""
        if empty_count > 45:  # Early game
            return {
                'positional': 0.4, 'mobility': 1.0, 'stability': 0.2,
                'corner': 1.5, 'frontier': -0.3, 'parity': 0.1
            }
        elif empty_count > 20:  # Mid game
            return {
                'positional': 0.7, 'mobility': 0.8, 'stability': 0.6,
                'corner': 1.2, 'frontier': -0.2, 'parity': 0.3
            }
        else:  # End game
            return {
                'positional': 0.3, 'mobility': 0.3, 'stability': 0.8,
                'corner': 0.5, 'frontier': -0.1, 'parity': 1.0
            }

    def evaluate_board_fast(self, board):
        """Lightweight evaluation for move ordering"""
        empty_count = board.get_empty_count()
        
        # Terminal position check
        if empty_count == 0:
            b, w = board.count_stones()
            diff = (b - w) if self.color == BLACK else (w - b)
            return 10000 * (1 if diff > 0 else -1 if diff < 0 else 0)
        
        # Quick positional evaluation
        weights = self.get_weights_for_stage(board)
        score = 0
        
        for i in range(8):
            for j in range(8):
                if board.board[i][j] == self.color:
                    score += weights[i][j]
                elif board.board[i][j] == opponent(self.color):
                    score -= weights[i][j]
        
        # Quick mobility check
        my_moves = len(board.get_valid_moves(self.color))
        opp_moves = len(board.get_valid_moves(opponent(self.color)))
        mobility_bonus = 10 * (my_moves - opp_moves)
        
        return score + mobility_bonus

    @lru_cache(maxsize=50000)  # Increased cache size
    def evaluate_board(self, board):
        """Comprehensive board evaluation with caching"""
        try:
            empty_count = board.get_empty_count()
            
            # Terminal position
            if empty_count == 0:
                b, w = board.count_stones()
                diff = (b - w) if self.color == BLACK else (w - b)
                return 10000 * (1 if diff > 0 else -1 if diff < 0 else 0)

            stage_weights = self.get_game_stage_weights(empty_count)

            # Calculate all evaluation components
            scores = {
                'positional': self.evaluate_positional(board),
                'mobility': self.evaluate_mobility(board),
                'stability': self.evaluate_stability(board) if empty_count < 45 else 0,  # Skip in early game
                'corner': self.evaluate_corners(board),
                'frontier': self.evaluate_frontier(board) if empty_count > 20 else 0,  # Skip in endgame
                'parity': self.evaluate_parity(board)
            }

            total_score = sum(stage_weights[key] * scores[key] for key in scores)
            return total_score
            
        except Exception as e:
            print(f"AI evaluation error: {str(e)}")
            return 0

    def evaluate_positional(self, board):
        """Optimized positional evaluation"""
        weights = self.get_weights_for_stage(board)
        score = 0
        
        # Vectorized approach for main evaluation
        for i in range(8):
            for j in range(8):
                piece = board.board[i][j]
                if piece == self.color:
                    score += weights[i][j]
                elif piece == opponent(self.color):
                    score -= weights[i][j]
        
        # Dynamic penalty for dangerous squares - optimized
        for corner_x, corner_y in CORNERS:
            if board.board[corner_x][corner_y] == EMPTY:
                # Check pre-computed dangerous squares
                for dx, dy in [(1, 1), (0, 1), (1, 0)]:
                    x, y = corner_x + dx, corner_y + dy
                    if 0 <= x < 8 and 0 <= y < 8:
                        piece = board.board[x][y]
                        if piece == self.color:
                            score -= 25
                        elif piece == opponent(self.color):
                            score += 25
        
        return score

    def evaluate_mobility(self, board):
        """Optimized mobility evaluation"""
        my_moves = len(board.get_valid_moves(self.color))
        opp_moves = len(board.get_valid_moves(opponent(self.color)))
        
        if my_moves + opp_moves == 0:
            return 0
        
        # Relative mobility with early termination
        mobility_score = 100 * (my_moves - opp_moves) / (my_moves + opp_moves + 1)
        
        # Quick bonus calculations
        if my_moves == 0:
            mobility_score -= 50
        elif opp_moves == 0:
            mobility_score += 50
            
        return mobility_score

    def evaluate_stability(self, board):
        """Lightweight stability evaluation"""
        # Only check corner stability for performance
        my_stable = sum(1 for x, y in CORNERS 
                       if board.board[x][y] == self.color)
        opp_stable = sum(1 for x, y in CORNERS 
                        if board.board[x][y] == opponent(self.color))
        
        # Edge stability
        edge_positions = [(0, i) for i in range(8)] + [(7, i) for i in range(8)] + \
                        [(i, 0) for i in range(1, 7)] + [(i, 7) for i in range(1, 7)]
        
        my_edge = sum(1 for x, y in edge_positions 
                     if board.board[x][y] == self.color)
        opp_edge = sum(1 for x, y in edge_positions 
                      if board.board[x][y] == opponent(self.color))
        
        return 50 * (my_stable - opp_stable) + 5 * (my_edge - opp_edge)

    def evaluate_corners(self, board):
        """Fast corner evaluation"""
        my_corners = sum(1 for x, y in CORNERS if board.board[x][y] == self.color)
        opp_corners = sum(1 for x, y in CORNERS if board.board[x][y] == opponent(self.color))
        
        corner_score = 25 * (my_corners - opp_corners)
        
        # Exponential bonus
        if my_corners > 1:
            corner_score += 10 * my_corners * my_corners
        if opp_corners > 1:
            corner_score -= 10 * opp_corners * opp_corners
            
        return corner_score

    def evaluate_frontier(self, board):
        """Fast frontier evaluation"""
        my_frontier = board.get_frontier_count(self.color)
        opp_frontier = board.get_frontier_count(opponent(self.color))
        return opp_frontier - my_frontier

    def evaluate_parity(self, board):
        """Simple parity evaluation"""
        b, w = board.count_stones()
        return (b - w) if self.color == BLACK else (w - b)

    def sort_moves_fast(self, board, moves, depth):
        """Ultra-fast move sorting for better pruning"""
        if not moves:
            return []
            
        move_scores = []
        
        for move in moves:
            x, y = move
            score = 0
            
            # Killer moves - highest priority
            if depth in self.killer_moves and move in self.killer_moves[depth]:
                score += 2000
                
            # History heuristic
            score += self.history_table.get(move, 0) // 10  # Normalize
            
            # Fast strategic evaluation
            if move in CORNERS:
                score += 1000
            elif move in X_SQUARES:
                # Check if adjacent corner is occupied (pre-computed)
                adjacent_corners = self.corner_adjacency[move]
                if any(board.board[corner[0]][corner[1]] == EMPTY for corner in adjacent_corners):
                    score -= 300
            elif move in C_SQUARES:
                score -= 150
            else:
                # Edge bonus
                if x == 0 or x == 7 or y == 0 or y == 7:
                    score += 100
                # Center proximity
                center_dist = abs(x - 3.5) + abs(y - 3.5)
                score += int((7 - center_dist) * 10)
            
            move_scores.append((score, move))
            
        # Sort descending by score
        move_scores.sort(reverse=True)
        return [move for _, move in move_scores]

    def alphabeta_optimized(self, board, depth, alpha, beta, maximizing, start_time):
        """Highly optimized alpha-beta search"""
        self.nodes_searched += 1
        
        # Time check - less frequent for performance
        if self.nodes_searched % 1000 == 0 and time.time() - start_time > self.time_limit * 0.9:
            return self.evaluate_board_fast(board), None
        
        # Zobrist hash for transposition table
        board_hash = zobrist_hash(board)
        
        # Transposition table lookup
        if board_hash in self.transposition_table:
            entry = self.transposition_table[board_hash]
            if entry['depth'] >= depth:
                return entry['score'], entry['move']
        
        current_color = self.color if maximizing else opponent(self.color)
        moves = board.get_valid_moves(current_color)
        
        # Terminal conditions
        if depth == 0 or not moves:
            if not moves:
                opponent_moves = board.get_valid_moves(opponent(current_color))
                if not opponent_moves:
                    return self.evaluate_board(board), None
                else:
                    return self.alphabeta_optimized(board, depth, alpha, beta, not maximizing, start_time)
            else:
                return self.evaluate_board(board), None
        
        # Fast move ordering
        sorted_moves = self.sort_moves_fast(board, moves, depth)
        best_move = sorted_moves[0] if sorted_moves else None
        
        if maximizing:
            max_score = float('-inf')
            for i, move in enumerate(sorted_moves):
                new_board = board.apply_move(*move, current_color)
                score, _ = self.alphabeta_optimized(new_board, depth - 1, alpha, beta, False, start_time)
                
                if score > max_score:
                    max_score = score
                    best_move = move
                    
                alpha = max(alpha, score)
                if beta <= alpha:
                    # Update killer moves and history
                    if depth not in self.killer_moves:
                        self.killer_moves[depth] = []
                    if len(self.killer_moves[depth]) >= 2:
                        self.killer_moves[depth].pop(0)
                    self.killer_moves[depth].append(move)
                    self.history_table[move] += depth * depth
                    break
                    
            # Store in transposition table (limit size)
            if len(self.transposition_table) < 100000:
                self.transposition_table[board_hash] = {
                    'score': max_score, 'move': best_move, 'depth': depth
                }
            return max_score, best_move
            
        else:
            min_score = float('inf')
            for move in sorted_moves:
                new_board = board.apply_move(*move, current_color)
                score, _ = self.alphabeta_optimized(new_board, depth - 1, alpha, beta, True, start_time)
                
                if score < min_score:
                    min_score = score
                    best_move = move
                    
                beta = min(beta, score)
                if beta <= alpha:
                    if depth not in self.killer_moves:
                        self.killer_moves[depth] = []
                    if len(self.killer_moves[depth]) >= 2:
                        self.killer_moves[depth].pop(0)
                    self.killer_moves[depth].append(move)
                    self.history_table[move] += depth * depth
                    break
                    
            if len(self.transposition_table) < 100000:
                self.transposition_table[board_hash] = {
                    'score': min_score, 'move': best_move, 'depth': depth
                }
            return min_score, best_move

    def iterative_deepening_optimized(self, board):
        """Optimized iterative deepening with aspiration windows"""
        start_time = time.time()
        best_move = None
        last_score = 0
        moves = board.get_valid_moves(self.color)
        
        if not moves:
            return None
        if len(moves) == 1:
            return moves[0]
        
        logging.info(f"[최적화된 탐색] 시작: max_depth={self.max_depth}, 가능한 수={len(moves)}개")
        
        # Start with depth 3 for better move ordering
        for depth in range(3, self.max_depth + 1):
            try:
                self.nodes_searched = 0
                depth_start_time = time.time()
                
                # Aspiration window search for depths > 4
                if depth > 4 and best_move:
                    window = 50
                    alpha, beta = last_score - window, last_score + window
                    score, move = self.alphabeta_optimized(board, depth, alpha, beta, True, start_time)
                    
                    # Research with full window if needed
                    if score <= alpha or score >= beta:
                        score, move = self.alphabeta_optimized(board, depth, float('-inf'), float('inf'), True, start_time)
                else:
                    score, move = self.alphabeta_optimized(board, depth, float('-inf'), float('inf'), True, start_time)
                
                if move:
                    best_move = move
                    last_score = score
                
                depth_time = time.time() - depth_start_time
                logging.info(f"깊이 {depth} 완료: {depth_time:.2f}s, 노드: {self.nodes_searched}, 점수: {score}")
                
                # Early termination if mate found
                if abs(score) > 9000:
                    logging.info("승부 확정 - 조기 종료")
                    break
                
            except Exception as e:
                logging.info(f"깊이 {depth}에서 예외: {e}")
                break

        total_time = time.time() - start_time
        logging.info(f"[최적화된 탐색 완료] time={total_time:.2f}s, best_move={best_move}")
        return best_move

    def solve_endgame_fast(self, board, color, alpha=float('-inf'), beta=float('inf')):
        """Fast endgame solver with alpha-beta pruning"""
        valid_moves = board.get_valid_moves(color)
        if not valid_moves:
            opponent_moves = board.get_valid_moves(opponent(color))
            if not opponent_moves:
                return board.count_score(color), None
            score, _ = self.solve_endgame_fast(board, opponent(color), -beta, -alpha)
            return -score, None
            
        best_score = alpha
        best_move = None
        
        for move in valid_moves:
            new_board = board.apply_move(*move, color)
            score, _ = self.solve_endgame_fast(new_board, opponent(color), -beta, -best_score)
            score = -score
            
            if score > best_score:
                best_score = score
                best_move = move
                
            if best_score >= beta:
                break
                
        return best_score, best_move

    def get_move(self, board):
        """Main entry point - optimized move selection"""
        self.nodes_searched = 0
        empty_count = board.get_empty_count()
        
        # Perfect endgame play
        if empty_count <= 16:
            logging.info("완전 해석 모드 시작")
            _, move = self.solve_endgame_fast(board, self.color)
            return move
        
        # Quick opening moves
        if empty_count > 54:
            moves = board.get_valid_moves(self.color)
            if moves:
                # Simple heuristic for opening
                corner_moves = [m for m in moves if m in CORNERS]
                if corner_moves:
                    return corner_moves[0]
                
                # Avoid X-squares in opening
                safe_moves = [m for m in moves if m not in X_SQUARES]
                if safe_moves:
                    return safe_moves[0]
                return moves[0]
        
        # Use optimized iterative deepening
        return self.iterative_deepening_optimized(board)