import time
import random
from collections import defaultdict  # ✅ 이 줄이 필요합니다

from constants import BLACK, WHITE, EMPTY, opponent, EARLY_WEIGHTS, MID_WEIGHTS, LATE_WEIGHTS, CORNERS, X_SQUARES, C_SQUARES
from board import Board

class AdvancedAI:
    def __init__(self, color, difficulty='hard', time_limit=3.0):
        self.color = color
        self.difficulty = difficulty
        self.time_limit = time_limit
        self.transposition_table = {}
        self.killer_moves = defaultdict(list)
        self.history_table = defaultdict(int)
        self.nodes_searched = 0
        
        # Difficulty settings
        if difficulty == 'easy':
            self.max_depth = 4
            self.time_limit = 1.0
        elif difficulty == 'medium':
            self.max_depth = 6
            self.time_limit = 2.0
        else:  # hard
            self.max_depth = 8
            self.time_limit = 3.0

    def get_weights_for_stage(self, board):
        """Get evaluation weights based on game stage"""
        empty_count = board.get_empty_count()
        if empty_count > 45:
            return EARLY_WEIGHTS
        elif empty_count > 20:
            return MID_WEIGHTS
        else:
            return LATE_WEIGHTS

    def evaluate_board(self, board):
        """Comprehensive board evaluation function"""
        if board.get_empty_count() == 0:
            # Game over - return actual score difference
            b, w = board.count_stones()
            diff = (b - w) if self.color == BLACK else (w - b)
            return 10000 * (1 if diff > 0 else -1 if diff < 0 else 0)
        
        empty_count = board.get_empty_count()
        stage_weights = self.get_game_stage_weights(empty_count)
        
        scores = {
            'positional': self.evaluate_positional(board),
            'mobility': self.evaluate_mobility(board),
            'stability': self.evaluate_stability(board),
            'corner': self.evaluate_corners(board),
            'frontier': self.evaluate_frontier(board),
            'parity': self.evaluate_parity(board)
        }
        
        total_score = sum(stage_weights[key] * scores[key] for key in scores)
        return total_score

    def get_game_stage_weights(self, empty_count):
        """Get weights for different evaluation components based on game stage"""
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

    def evaluate_positional(self, board):
        """Evaluate positional advantage"""
        weights = self.get_weights_for_stage(board)
        score = 0
        
        for i in range(8):
            for j in range(8):
                if board.board[i][j] == self.color:
                    score += weights[i][j]
                elif board.board[i][j] == opponent(self.color):
                    score -= weights[i][j]
        
        # Dynamic penalty for dangerous squares near empty corners
        for corner_x, corner_y in CORNERS:
            if board.board[corner_x][corner_y] == EMPTY:
                # Penalize X-squares and C-squares near empty corners
                dangerous_squares = [(corner_x + dx, corner_y + dy) 
                                   for dx, dy in [(1, 1), (0, 1), (1, 0)] 
                                   if 0 <= corner_x + dx < 8 and 0 <= corner_y + dy < 8]
                
                for x, y in dangerous_squares:
                    if board.board[x][y] == self.color:
                        score -= 25
                    elif board.board[x][y] == opponent(self.color):
                        score += 25
        
        return score

    def evaluate_mobility(self, board):
        """Evaluate mobility (number of possible moves)"""
        my_moves = len(board.get_valid_moves(self.color))
        opp_moves = len(board.get_valid_moves(opponent(self.color)))
        
        if my_moves + opp_moves == 0:
            return 0
        
        # Relative mobility
        mobility_score = 100 * (my_moves - opp_moves) / (my_moves + opp_moves + 1)
        
        # Bonus for having moves when opponent doesn't
        if my_moves > 0 and opp_moves == 0:
            mobility_score += 50
        elif my_moves == 0 and opp_moves > 0:
            mobility_score -= 50
            
        return mobility_score

    def evaluate_stability(self, board):
        """Evaluate disc stability"""
        my_stable = 0
        opp_stable = 0
        
        for i in range(8):
            for j in range(8):
                if board.board[i][j] == self.color and board.is_stable(i, j):
                    my_stable += 1
                elif board.board[i][j] == opponent(self.color) and board.is_stable(i, j):
                    opp_stable += 1
                    
        return 25 * (my_stable - opp_stable)

    def evaluate_corners(self, board):
        """Evaluate corner control"""
        my_corners = sum(1 for x, y in CORNERS if board.board[x][y] == self.color)
        opp_corners = sum(1 for x, y in CORNERS if board.board[x][y] == opponent(self.color))
        
        corner_score = 25 * (my_corners - opp_corners)
        
        # Exponential bonus for multiple corners
        if my_corners > 1:
            corner_score += 10 * my_corners * my_corners
        if opp_corners > 1:
            corner_score -= 10 * opp_corners * opp_corners
            
        return corner_score

    def evaluate_frontier(self, board):
        """Evaluate frontier discs (fewer is better)"""
        my_frontier = board.get_frontier_count(self.color)
        opp_frontier = board.get_frontier_count(opponent(self.color))
        return opp_frontier - my_frontier

    def evaluate_parity(self, board):
        """Evaluate disc count parity"""
        b, w = board.count_stones()
        diff = (b - w) if self.color == BLACK else (w - b)
        return diff

    def sort_moves(self, board, moves, depth):
        """Sort moves for better alpha-beta pruning"""
        move_scores = []
        
        for move in moves:
            x, y = move
            score = 0
            
            # Killer moves bonus
            if move in self.killer_moves.get(depth, []):
                score += 1000
                
            # History heuristic
            score += self.history_table.get(move, 0)
            
            # Strategic position values
            if move in CORNERS:
                score += 500
            elif move in X_SQUARES:
                # Check if adjacent corner is occupied
                adjacent_corner_occupied = False
                for corner in CORNERS:
                    if abs(corner[0] - x) <= 1 and abs(corner[1] - y) <= 1:
                        if board.board[corner[0]][corner[1]] != EMPTY:
                            adjacent_corner_occupied = True
                            break
                if not adjacent_corner_occupied:
                    score -= 200
            elif move in C_SQUARES:
                score -= 100
            elif x == 0 or x == 7 or y == 0 or y == 7:  # Edge
                score += 50
                
            # Mobility impact
            new_board = board.apply_move(x, y, self.color)
            opp_moves_before = len(board.get_valid_moves(opponent(self.color)))
            opp_moves_after = len(new_board.get_valid_moves(opponent(self.color)))
            score += (opp_moves_before - opp_moves_after) * 10
            
            # Disc flip count
            if new_board.move_history:
                flipped_count = len(new_board.move_history[-1][3])
                score += flipped_count * 3
                
            move_scores.append((score, move))
            
        move_scores.sort(reverse=True)
        return [move for _, move in move_scores]

    def alphabeta(self, board, depth, alpha, beta, maximizing, start_time):
        """Alpha-beta pruning with enhancements"""
        self.nodes_searched += 1
        
        # Time limit check
        if time.time() - start_time > self.time_limit:
            return self.evaluate_board(board), None
            
        # Transposition table lookup
        board_hash = board.get_hash()
        if board_hash in self.transposition_table:
            entry = self.transposition_table[board_hash]
            if entry['depth'] >= depth:
                return entry['score'], entry['move']
        
        current_color = self.color if maximizing else opponent(self.color)
        moves = board.get_valid_moves(current_color)
        
        # Terminal node or max depth reached
        if depth == 0 or not moves:
            if not moves:
                # No moves available - check if opponent can move
                opponent_moves = board.get_valid_moves(opponent(current_color))
                if not opponent_moves:
                    # Game over
                    return self.evaluate_board(board), None
                else:
                    # Pass turn
                    return self.alphabeta(board, depth, alpha, beta, not maximizing, start_time)
            else:
                return self.evaluate_board(board), None
        
        # Sort moves for better pruning
        sorted_moves = self.sort_moves(board, moves, depth)
        best_move = None
        
        if maximizing:
            max_score = float('-inf')
            for move in sorted_moves:
                new_board = board.apply_move(*move, current_color)
                score, _ = self.alphabeta(new_board, depth - 1, alpha, beta, False, start_time)
                
                if score > max_score:
                    max_score = score
                    best_move = move
                    
                alpha = max(alpha, score)
                if beta <= alpha:
                    # Beta cutoff - save killer move
                    if len(self.killer_moves[depth]) >= 2:
                        self.killer_moves[depth].pop(0)
                    self.killer_moves[depth].append(move)
                    break
                    
            # Update history table
            if best_move:
                self.history_table[best_move] += depth * depth
                
            # Store in transposition table
            self.transposition_table[board_hash] = {
                'score': max_score, 'move': best_move, 'depth': depth
            }
            return max_score, best_move
            
        else:
            min_score = float('inf')
            for move in sorted_moves:
                new_board = board.apply_move(*move, current_color)
                score, _ = self.alphabeta(new_board, depth - 1, alpha, beta, True, start_time)
                
                if score < min_score:
                    min_score = score
                    best_move = move
                    
                beta = min(beta, score)
                if beta <= alpha:
                    if len(self.killer_moves[depth]) >= 2:
                        self.killer_moves[depth].pop(0)
                    self.killer_moves[depth].append(move)
                    break
                    
            if best_move:
                self.history_table[best_move] += depth * depth
                
            self.transposition_table[board_hash] = {
                'score': min_score, 'move': best_move, 'depth': depth
            }
            return min_score, best_move

    def iterative_deepening(self, board):
        """Iterative deepening search"""
        start_time = time.time()
        best_move = None
        
        moves = board.get_valid_moves(self.color)
        if not moves:
            return None
            
        if len(moves) == 1:
            return moves[0]
        
        # Start with depth 1 and increase
        for depth in range(1, self.max_depth + 1):
            try:
                _, move = self.alphabeta(board, depth, float('-inf'), float('inf'), True, start_time)
                if move:
                    best_move = move
                    
                # Time management
                elapsed = time.time() - start_time
                if elapsed > self.time_limit * 0.8:
                    break
                    
            except:
                break
                
        return best_move

    def get_opening_move(self, board):
        """Use opening book for early game"""
        # Simple opening strategy
        moves = board.get_valid_moves(self.color)
        if not moves:
            return None
            
        # Prefer center and edge positions, avoid X-squares
        preferred = []
        for move in moves:
            x, y = move
            if move not in X_SQUARES and move not in C_SQUARES:
                # Calculate distance from center
                center_dist = abs(x - 3.5) + abs(y - 3.5)
                preferred.append((center_dist, move))
                
        if preferred:
            preferred.sort()
            return preferred[0][1]
        
        return random.choice(moves)

    def get_move(self, board):
        """Get the best move for current position"""
        self.nodes_searched = 0
        
        # Use opening book in early game
        if board.get_empty_count() > 50:
            return self.get_opening_move(board)
        
        # Use iterative deepening for main search
        return self.iterative_deepening(board)