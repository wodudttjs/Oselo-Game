
from collections import defaultdict

from multiprocessing import Pool
from constants import BLACK, WHITE, EMPTY, opponent,CORNERS, X_SQUARES, C_SQUARES
from board import Board

class EgaroucidInspiredAI:
   
    def __init__(self, color, difficulty='hard', time_limit=3.0):
        self.color = color
        self.difficulty = difficulty
        self.time_limit = time_limit
        
        # Enhanced transposition table with age-based replacement
        self.transposition_table = {}
        self.tt_age = 0
        self.max_tt_size = 100000
        
        # Multi-level move ordering
        self.killer_moves = defaultdict(list)
        self.history_table = defaultdict(int)
        self.butterfly_table = defaultdict(int)
        self.counter_moves = {}
        
        # Search statistics
        self.nodes_searched = 0
        self.tt_hits = 0
        self.tt_cutoffs = 0
        
        # Egaroucid-inspired pattern tables
        self.pattern_weights = self._initialize_pattern_weights()
        self.stability_table = self._precompute_stability_table()
        
        # Enhanced opening book with Egaroucid's known sequences
        self.opening_book = self._build_egaroucid_opening_book()
        self.game_history = []
        
        # Multi-PV (Principal Variation) search
        self.pv_table = defaultdict(list)
        self.pv_length = defaultdict(int)
        
        # Difficulty settings optimized like Egaroucid
        if difficulty == 'easy':
            self.base_depth = 3
            self.time_limit = 1.0
            self.selective_depth = 1
        elif difficulty == 'medium':
            self.base_depth = 4
            self.time_limit = 2.0
            self.selective_depth = 2
        else:  # hard
            self.base_depth = 5
            self.time_limit = 3.0
            self.selective_depth = 3
        
        # Aspiration window parameters
        self.aspiration_window = 25
        self.prev_score = 0
        
        # Endgame solver threshold
        self.endgame_threshold = 18
        
        # Search extensions
        self.max_extensions = 8
    def get_move(self, board):
        empty = board.get_empty_count()
        
        
        if empty <= self.endgame_threshold:
            move = self._solve_endgame(board)
        else:
            depth = self.get_adaptive_depth(board)
            best_score = -float('inf')
            best_move = None

            for move in board.get_valid_moves(self.color):
                new_board = board.apply_move(*move, self.color)
                score = self._alphabeta(new_board, depth - 1, -float('inf'), float('inf'), False,)
                if score > best_score:
                    best_score = score
                    best_move = move

        self.record_move(best_move)
        return best_move
    

    def record_move(self, move):
        if move:
            self.game_history.append(move)
    def _initialize_pattern_weights(self):
        """Initialize Egaroucid-style pattern recognition weights"""
        patterns = {}
        
        # Edge patterns (8 positions each)
        patterns['edge'] = {
            # Stable edge patterns
            (1,1,1,1,1,1,1,1): 1000,  # Full edge
            (1,1,1,1,1,1,1,0): 800,   # Almost full
            (1,1,1,1,0,0,0,0): 300,   # Half stable
            # Wedge patterns
            (1,1,1,0,0,0,0,0): 200,
            (0,0,0,1,1,1,0,0): 150,
            # Dangerous patterns
            (0,1,0,0,0,0,0,0): -50,   # Isolated disc
        }
        
        # Corner patterns (3x3 around each corner)
        patterns['corner'] = {
            # Perfect corner control
            (1,1,1,1,1,1,1,1,1): 2000,
            # Good corner control
            (1,1,1,1,1,0,1,0,0): 1500,
            (1,1,0,1,1,0,0,0,0): 1200,
            # Bad corner patterns (giving away corner)
            (0,1,0,1,0,0,0,0,0): -1000,
            (0,0,1,0,0,1,0,0,0): -800,
        }
        
        # Diagonal patterns
        patterns['diagonal'] = {
            (1,1,1,1): 400,
            (1,1,1,0): 200,
            (1,1,0,1): 150,
            (0,1,1,1): 180,
        }
        
        return patterns
    
    def _precompute_stability_table(self):
        """Precompute stability patterns like Egaroucid"""
        stability = {}
        
        # Directions for stability check
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        # Precompute common stability patterns
        for pattern in range(256):  # 8-bit patterns
            stable_count = 0
            for i in range(8):
                if pattern & (1 << i):
                    stable_count += 1
            stability[pattern] = stable_count
            
        return stability
    
    def _build_egaroucid_opening_book(self):
        """Build comprehensive opening book based on Egaroucid's known strategies"""
        book = {}
        
        # Egaroucid's preferred openings
        book['tiger'] = {
            # Tiger opening sequence
            (): [(2,4), (4,2), (5,3), (3,5)],
            ((2,4),): [(1,4), (2,5), (3,4)],
            ((4,2),): [(4,1), (5,2), (4,3)],
        }
        
        book['buffalo'] = {
            # Buffalo opening
            (): [(2,3), (3,2), (4,5), (5,4)],
            ((2,3),): [(2,2), (1,3), (3,3)],
            ((3,2),): [(2,2), (3,1), (3,3)],
        }
        
        book['cow'] = {
            # Cow opening (more aggressive)
            (): [(2,4), (4,2)],
            ((2,4),): [(2,3), (2,5), (1,4)],
            ((4,2),): [(3,2), (5,2), (4,1)],
        }
        
        # Anti-human strategies (common human mistake counters)
        book['anti_human'] = {
            # Counter greedy disc-flipping
            ((2,3), (2,2)): [(1,2), (2,1)],  # Punish corner-adjacent
            ((3,2), (2,2)): [(2,1), (1,2)],
            # Counter edge-grabbing
            ((2,4), (1,4)): [(0,4), (2,5)],  # Use edge control properly
        }
        
        return book
    
    def get_adaptive_depth(self, board):
        """Enhanced adaptive depth calculation like Egaroucid"""
        empty_count = board.get_empty_count()
        
        # Endgame perfect play
        if empty_count <= self.endgame_threshold:
            return empty_count  # Solve to the end
        
        # Dynamic depth based on position complexity
        complexity_factor = self._calculate_position_complexity(board)
        
        if empty_count > 50:
            # Opening: deeper search for better foundation
            return self.base_depth + 2 + complexity_factor
        elif empty_count > 35:
            # Early middle game
            return self.base_depth + 1 + complexity_factor
        elif empty_count > 20:
            # Late middle game
            return self.base_depth + complexity_factor
        else:
            # Pre-endgame: increase depth for accuracy
            return self.base_depth + 3 + complexity_factor
    
    def _calculate_position_complexity(self, board):
        """Calculate position complexity to adjust search depth"""
        complexity = 0
        
        # More moves = more complex
        my_moves = len(board.get_valid_moves(self.color))
        opp_moves = len(board.get_valid_moves(opponent(self.color)))
        complexity += min(2, (my_moves + opp_moves) // 8)
        
        # Unstable positions need deeper search
        unstable_discs = 0
        for i in range(8):
            for j in range(8):
                if board.board[i][j] != EMPTY and not board.is_stable(i, j):
                    unstable_discs += 1
        complexity += min(2, unstable_discs // 10)
        
        # Corner fights need deeper analysis
        corner_fights = 0
        for corner in CORNERS:
            x, y = corner
            if board.board[x][y] == EMPTY:
                # Check if there's activity around this corner
                activity = 0
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < 8 and 0 <= ny < 8 and board.board[nx][ny] != EMPTY:
                            activity += 1
                if activity >= 3:
                    corner_fights += 1
        complexity += corner_fights
        
        return min(4, complexity)  # Cap at +4 depth
    
    def evaluate_board_egaroucid_style(self, board):
        """Comprehensive evaluation inspired by Egaroucid's approach"""
        empty_count = board.get_empty_count()
        if empty_count == 0:
            b, w = board.count_stones()
            diff = (b - w) if self.color == BLACK else (w - b)
            return 10000 + diff if diff > 0 else -10000 + diff if diff < 0 else 0
        
       
        
        # Multi-component evaluation
        scores = {
            'mobility': self._evaluate_mobility_advanced(board),
            'stability': self._evaluate_stability_advanced(board),
            'positional': self._evaluate_positional_patterns(board),
            'corner_control': self._evaluate_corner_control_advanced(board),
            'edge_control': self._evaluate_edge_patterns(board),
            'parity': self._evaluate_parity_advanced(board),
            'potential': self._evaluate_potential_mobility(board),
            'tempo': self._evaluate_tempo(board),
        }
        
        # Stage-dependent weights (Egaroucid-style)
        weights = self._get_egaroucid_weights(empty_count)
        
        total_score = sum(weights[key] * scores[key] for key in scores)
        
        # Add opening book bonus
        if empty_count > 45:
            total_score += self._get_opening_bonus(board)
        
        return int(total_score)
    
    def _get_egaroucid_weights(self, empty_count):
        """Egaroucid-inspired stage weights"""
        stages = [
            (55, {'mobility': 1.0, 'stability': 0.1, 'positional': 0.3, 'corner_control': 2.0, 
                  'edge_control': 0.4, 'parity': 0.05, 'potential': 0.8, 'tempo': 0.6}),
            (40, {'mobility': 0.9, 'stability': 0.3, 'positional': 0.5, 'corner_control': 1.8,
                  'edge_control': 0.6, 'parity': 0.1, 'potential': 0.7, 'tempo': 0.5}),
            (25, {'mobility': 0.7, 'stability': 0.6, 'positional': 0.7, 'corner_control': 1.2,
                  'edge_control': 0.8, 'parity': 0.3, 'potential': 0.4, 'tempo': 0.3}),
            (10, {'mobility': 0.3, 'stability': 1.0, 'positional': 0.4, 'corner_control': 0.6,
                  'edge_control': 0.5, 'parity': 1.5, 'potential': 0.1, 'tempo': 0.1}),
            (0,  {'mobility': 0.0, 'stability': 0.5, 'positional': 0.1, 'corner_control': 0.2,
                  'edge_control': 0.2, 'parity': 2.0, 'potential': 0.0, 'tempo': 0.0}),
        ]
        
        for threshold, weights in stages:
            if empty_count > threshold:
                return weights
        return stages[-1][1]
    
    def _evaluate_mobility_advanced(self, board):
        """Advanced mobility evaluation like Egaroucid"""
        my_moves = board.get_valid_moves(self.color)
        opp_moves = board.get_valid_moves(opponent(self.color))
        
        my_count = len(my_moves)
        opp_count = len(opp_moves)
        
        if my_count + opp_count == 0:
            return 0
        
        # Basic mobility ratio
        mobility_ratio = 100 * (my_count - opp_count) / (my_count + opp_count + 1)
        
        # Quality of moves (Egaroucid considers move quality heavily)
        my_quality = sum(self._evaluate_move_quality_advanced(board, move) for move in my_moves)
        opp_quality = sum(self._evaluate_move_quality_advanced(board, move) for move in opp_moves)
        
        quality_bonus = (my_quality - opp_quality) * 0.3
        
        # Mobility advantage in critical situations
        if my_count == 0 and opp_count > 0:
            return -200  # Very bad - no moves
        elif my_count > 0 and opp_count == 0:
            return 200   # Very good - opponent has no moves
        
        # Penalize having too many moves in endgame (parity consideration)
        if board.get_empty_count() < 15 and my_count > opp_count + 3:
            mobility_ratio -= 30
            
        return mobility_ratio + quality_bonus
    
    def _evaluate_move_quality_advanced(self, board, move):
        """Evaluate individual move quality"""
        x, y = move
        quality = 0
        
        # Strategic value of the square
        if (x, y) in CORNERS:
            quality += 100
        elif x == 0 or x == 7 or y == 0 or y == 7:  # Edge
            quality += 30
            # But penalize if it gives away a corner
            for corner in CORNERS:
                if abs(corner[0] - x) <= 1 and abs(corner[1] - y) <= 1:
                    if board.board[corner[0]][corner[1]] == EMPTY:
                        quality -= 60
        elif (x, y) in X_SQUARES:
            quality -= 80  # Dangerous squares
        elif (x, y) in C_SQUARES:
            quality -= 40  # Risky squares
        
        # Evaluate the move's impact
        new_board = board.apply_move(x, y, self.color)
        if new_board.move_history:
            flipped = new_board.move_history[-1][3]
            
            # Bonus for flipping many discs (but not too many in endgame)
            flip_bonus = len(flipped) * 3
            if board.get_empty_count() < 20:
                flip_bonus = min(flip_bonus, 15)  # Cap in endgame
            quality += flip_bonus
            
            # Bonus for creating stable discs
            stable_created = 0
            for fx, fy in flipped:
                if new_board.is_stable(fx, fy):
                    stable_created += 1
            quality += stable_created * 10
        
        return quality
    
    def _evaluate_stability_advanced(self, board):
        """Advanced stability evaluation with pattern recognition"""
        my_stable = 0
        opp_stable = 0
        my_semi_stable = 0
        opp_semi_stable = 0
        
        # Count different types of stability
        for i in range(8):
            for j in range(8):
                if board.board[i][j] == self.color:
                    if board.is_stable(i, j):
                        my_stable += 1
                    elif self._is_semi_stable_advanced(board, i, j):
                        my_semi_stable += 1
                elif board.board[i][j] == opponent(self.color):
                    if board.is_stable(i, j):
                        opp_stable += 1
                    elif self._is_semi_stable_advanced(board, i, j):
                        opp_semi_stable += 1
        
        # Weighted stability score
        stability_score = (my_stable * 50 + my_semi_stable * 20) - (opp_stable * 50 + opp_semi_stable * 20)
        
        # Add pattern-based stability bonuses
        pattern_bonus = self._evaluate_stability_patterns(board)
        
        return stability_score + pattern_bonus
    
    def _is_semi_stable_advanced(self, board, x, y):
        """Enhanced semi-stability check"""
        if board.board[x][y] == EMPTY:
            return False
        
        color = board.board[x][y]
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        stable_directions = 0
        
        for dx, dy in directions:
            # Check if this direction is stable
            stable_in_direction = True
            nx, ny = x + dx, y + dy
            
            while 0 <= nx < 8 and 0 <= ny < 8:
                if board.board[nx][ny] == EMPTY:
                    stable_in_direction = False
                    break
                elif board.board[nx][ny] != color:
                    break
                nx += dx
                ny += dy
            
            # Also check the opposite direction
            if stable_in_direction:
                nx, ny = x - dx, y - dy
                while 0 <= nx < 8 and 0 <= ny < 8:
                    if board.board[nx][ny] == EMPTY:
                        stable_in_direction = False
                        break
                    elif board.board[nx][ny] != color:
                        break
                    nx -= dx
                    ny -= dy
            
            if stable_in_direction:
                stable_directions += 1
        
        return stable_directions >= 3
    
    def _evaluate_stability_patterns(self, board):
        """Pattern-based stability evaluation"""
        score = 0
        
        # Check for stable edge formations
        edges = [
            [(0, i) for i in range(8)],  # Top
            [(7, i) for i in range(8)],  # Bottom  
            [(i, 0) for i in range(8)],  # Left
            [(i, 7) for i in range(8)]   # Right
        ]
        
        for edge in edges:
            my_pattern = []
            for x, y in edge:
                if board.board[x][y] == self.color:
                    my_pattern.append(1)
                elif board.board[x][y] == opponent(self.color):
                    my_pattern.append(-1)
                else:
                    my_pattern.append(0)
            
            # Look for stable patterns
            stable_runs = self._find_stable_runs(my_pattern)
            score += stable_runs * 25
        
        return score
    
    def _find_stable_runs(self, pattern):
        """Find stable runs in a pattern"""
        runs = 0
        current_run = 0
        
        for value in pattern:
            if value == 1:
                current_run += 1
            else:
                if current_run >= 3:  # Stable run of 3+
                    runs += current_run
                current_run = 0
        
        if current_run >= 3:
            runs += current_run
            
        return runs
    
    def _evaluate_positional_patterns(self, board):
        """Pattern-based positional evaluation"""
        score = 0
        
        # Use precomputed pattern weights
        score += self._evaluate_edge_patterns_detailed(board)
        score += self._evaluate_corner_patterns_detailed(board)
        score += self._evaluate_diagonal_patterns(board)
        
        return score
    
    def _evaluate_edge_patterns_detailed(self, board):
        """Detailed edge pattern evaluation"""
        score = 0
        edges = [
            [(0, i) for i in range(8)],
            [(7, i) for i in range(8)],
            [(i, 0) for i in range(8)],
            [(i, 7) for i in range(8)]
        ]
        
        for edge in edges:
            pattern = tuple(1 if board.board[x][y] == self.color else 
                          -1 if board.board[x][y] == opponent(self.color) else 0 
                          for x, y in edge)
            
            # Check against known good/bad patterns
            if pattern in self.pattern_weights.get('edge', {}):
                score += self.pattern_weights['edge'][pattern]
            
            # Reverse pattern for opponent
            reverse_pattern = tuple(-x for x in pattern)
            if reverse_pattern in self.pattern_weights.get('edge', {}):
                score -= self.pattern_weights['edge'][reverse_pattern]
        
        return score
    
    def _evaluate_corner_patterns_detailed(self, board):
        """Detailed corner pattern evaluation"""
        score = 0
        
        for corner_x, corner_y in CORNERS:
            pattern = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    x, y = corner_x + dx, corner_y + dy
                    if 0 <= x < 8 and 0 <= y < 8:
                        if board.board[x][y] == self.color:
                            pattern.append(1)
                        elif board.board[x][y] == opponent(self.color):
                            pattern.append(-1)
                        else:
                            pattern.append(0)
                    else:
                        pattern.append(1)  # Treat off-board as friendly
            
            pattern_tuple = tuple(pattern)
            if pattern_tuple in self.pattern_weights.get('corner', {}):
                score += self.pattern_weights['corner'][pattern_tuple]
        
        return score
    
    def _evaluate_diagonal_patterns(self, board):
        """Evaluate diagonal patterns"""
        score = 0
        
        # Main diagonals
        diagonals = [
            [(i, i) for i in range(8)],                    # Main diagonal
            [(i, 7-i) for i in range(8)],                  # Anti-diagonal
            [(i, i) for i in range(4)],                    # Short diagonals
            [(i+4, i+4) for i in range(4)],
            [(i, 7-i) for i in range(4)],
            [(i+4, 7-(i+4)) for i in range(4)],
        ]
        
        for diagonal in diagonals:
            my_count = sum(1 for x, y in diagonal if board.board[x][y] == self.color)
            opp_count = sum(1 for x, y in diagonal if board.board[x][y] == opponent(self.color))
            
            if my_count >= 3:
                score += my_count * 15
            if opp_count >= 3:
                score -= opp_count * 15
        
        return score
    
    def _evaluate_corner_control_advanced(self, board):
        """Advanced corner control evaluation"""
        my_corners = 0
        opp_corners = 0
        corner_potential = 0
        
        for corner_x, corner_y in CORNERS:
            if board.board[corner_x][corner_y] == self.color:
                my_corners += 1
            elif board.board[corner_x][corner_y] == opponent(self.color):
                opp_corners += 1
            else:
                # Evaluate corner potential
                corner_potential += self._evaluate_corner_potential(board, corner_x, corner_y)
        
        # Base corner score with exponential bonus
        corner_score = 200 * (my_corners - opp_corners)
        if my_corners > 1:
            corner_score += 100 * my_corners * my_corners
        if opp_corners > 1:
            corner_score -= 100 * opp_corners * opp_corners
        
        return corner_score + corner_potential
    
    def _evaluate_corner_potential(self, board, corner_x, corner_y):
        """Evaluate potential to get a corner"""
        score = 0
        
        # Check X-square occupation
        x_squares = []
        if corner_x == 0 and corner_y == 0:
            x_squares = [(1, 1)]
        elif corner_x == 0 and corner_y == 7:
            x_squares = [(1, 6)]
        elif corner_x == 7 and corner_y == 0:
            x_squares = [(6, 1)]
        elif corner_x == 7 and corner_y == 7:
            x_squares = [(6, 6)]
        
        for x, y in x_squares:
            if board.board[x][y] == opponent(self.color):
                score += 50  # Opponent in X-square gives us corner chances
            elif board.board[x][y] == self.color:
                score -= 50  # We're in X-square, bad for us
        
        # Check C-square control
        c_squares = []
        if corner_x == 0 and corner_y == 0:
            c_squares = [(0, 1), (1, 0)]
        elif corner_x == 0 and corner_y == 7:
            c_squares = [(0, 6), (1, 7)]
        elif corner_x == 7 and corner_y == 0:
            c_squares = [(6, 0), (7, 1)]
        elif corner_x == 7 and corner_y == 7:
            c_squares = [(6, 7), (7, 6)]
        
        my_c_control = sum(1 for x, y in c_squares 
                          if 0 <= x < 8 and 0 <= y < 8 and board.board[x][y] == self.color)
        opp_c_control = sum(1 for x, y in c_squares 
                           if 0 <= x < 8 and 0 <= y < 8 and board.board[x][y] == opponent(self.color))
        
        score += (my_c_control - opp_c_control) * 20
        
        return score
    
    def _evaluate_edge_patterns(self, board):
        """Evaluate edge control patterns"""
        score = 0
        
        # Egaroucid heavily values edge control
        edges = [
            [(0, i) for i in range(8)],
            [(7, i) for i in range(8)],
            [(i, 0) for i in range(8)],
            [(i, 7) for i in range(8)]
        ]
        
        for edge in edges:
            my_discs = [i for i, (x, y) in enumerate(edge) if board.board[x][y] == self.color]
            opp_discs = [i for i, (x, y) in enumerate(edge) if board.board[x][y] == opponent(self.color)]
            
            # Continuous edge runs are valuable
            my_runs = self._calculate_runs(my_discs)
            opp_runs = self._calculate_runs(opp_discs)
            
            score += sum(run * run * 10 for run in my_runs)
            score -= sum(run * run * 10 for run in opp_runs)
            
            # Control of edge endpoints (near corners) is extra valuable
            if 0 in my_discs or 7 in my_discs:
                score += 30
            if 0 in opp_discs or 7 in opp_discs:
                score -= 30
        
        return score
    
    def _calculate_runs(self, positions):
        """Calculate consecutive runs in a list of positions"""
        if not positions:
            return []
        
        runs = []
        current_run = 1
        
        for i in range(1, len(positions)):
            if positions[i] == positions[i-1] + 1:
                current_run += 1
            else:
                runs.append(current_run)
                current_run = 1
        runs.append(current_run)
        
        return runs
    
    def _evaluate_parity_advanced(self, board):
        """Advanced parity evaluation"""
        b, w = board.count_stones()
        diff = (b - w) if self.color == BLACK else (w - b)
        
        empty_count = board.get_empty_count()
        
        # Parity becomes critical in endgame
        if empty_count <= 10:
            return diff * 5
        elif empty_count <= 15:
            return diff * 3
        else:
            return diff
    
    def _evaluate_potential_mobility(self, board):
        """Evaluate potential future mobility"""
        score = 0
        
        # Look at empty squares adjacent to our discs vs opponent discs
        my_potential = 0
        opp_potential = 0
        
        for i in range(8):
            for j in range(8):
                if board.board[i][j] == EMPTY:
                    # Check adjacency to our discs and opponent discs
                    adjacent_to_mine = False
                    adjacent_to_opp = False
                    
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if 0 <= ni < 8 and 0 <= nj < 8:
                                if board.board[ni][nj] == self.color:
                                    adjacent_to_mine = True
                                elif board.board[ni][nj] == opponent(self.color):
                                    adjacent_to_opp = True
                    
                    if adjacent_to_mine:
                        my_potential += 1
                    if adjacent_to_opp:
                        opp_potential += 1
        
        return (my_potential - opp_potential) * 2
    
    def _evaluate_tempo(self, board):
        """Evaluate tempo advantages"""
        my_moves = board.get_valid_moves(self.color)
        opp_moves = board.get_valid_moves(opponent(self.color))

        # Tempo score based on forcing limited responses
        if len(opp_moves) == 1:
            return 50  # Very good forcing move
        elif len(opp_moves) == 2:
            return 30
        elif len(opp_moves) == 3:
            return 15

        # If we're limited, penalize
        if len(my_moves) <= 1:
            return -40

        return 0

    def _get_opening_bonus(self, board):
        """Add bonus if opening sequence matches known good ones"""
        bonus = 0
        for name, lines in self.opening_book.items():
            for seq, moves in lines.items():
                if tuple(self.game_history[:len(seq)]) == seq:
                    bonus += 15  # Matching a known opening
        return bonus

    

    def reset(self):
        """Reset internal state for a new game"""
        self.transposition_table.clear()
        self.killer_moves.clear()
        self.history_table.clear()
        self.butterfly_table.clear()
        self.counter_moves.clear()
        self.nodes_searched = 0
        self.tt_hits = 0
        self.tt_cutoffs = 0
        self.pv_table.clear()
        self.pv_length.clear()
        self.tt_age = 0
        self.game_history.clear()
    def _precompute_stability_table(self):
        return {i: bin(i).count('1') for i in range(256)}
    
    
    def _alphabeta(self, board, depth, alpha, beta, maximizing):
        self.nodes_searched += 1

        board_hash = board.get_hash()
        tt_entry = self.transposition_table.get(board_hash)

    # Transposition Table Hit
        if tt_entry and tt_entry['depth'] >= depth:
            self.tt_hits += 1
            return tt_entry['score']

    # Terminal or depth limit
        if depth == 0 or board.get_empty_count() == 0:
            score = self.evaluate_board_egaroucid_style(board)
            self.transposition_table[board_hash] = {'score': score, 'depth': depth}
            return score

        color = self.color if maximizing else opponent(self.color)
        valid_moves = board.get_valid_moves(color)

        if not valid_moves:
        # 패스 상황
            return self._alphabeta(board, depth, alpha, beta, not maximizing)

    # Killer & history ordering
        sorted_moves = sorted(
            valid_moves,
            key=lambda move: (
                move in self.killer_moves[depth],  # True > False
                self.history_table.get(move, 0)
            ),
            reverse=True
        )

        best_score = -float('inf') if maximizing else float('inf')
        best_move = None

        for move in sorted_moves:
            child = board.apply_move(*move, color)
            score = self._alphabeta(child, depth - 1, alpha, beta, not maximizing)

            if maximizing:
                if score > best_score:
                    best_score = score
                    best_move = move
                alpha = max(alpha, best_score)
            else:
                if score < best_score:
                    best_score = score
                    best_move = move
                beta = min(beta, best_score)

            # Beta cutoff
            if beta <= alpha:
                self.tt_cutoffs += 1
                self.killer_moves[depth].append(move)
                self.killer_moves[depth] = self.killer_moves[depth][-2:]  # Keep recent 2
                break

        # History Heuristic update
        if best_move:
            self.history_table[best_move] += depth * depth

        # Store in TT
        self.transposition_table[board_hash] = {'score': best_score, 'depth': depth}
        return best_score

    def _solve_endgame(self, board):
        best_score = -float('inf')
        best_move = None
        for move in board.get_valid_moves(self.color):
            new_board = board.apply_move(*move, self.color)
            score = self._minimax(new_board, True)
            if score > best_score:
                best_score = score
                best_move = move
        self.record_move(best_move)
        return best_move

    def _minimax(self, board, maximizing):
        if board.get_empty_count() == 0:
            return self.evaluate_board_egaroucid_style(board)

        color = self.color if maximizing else opponent(self.color)
        valid_moves = board.get_valid_moves(color)
        if not valid_moves:
            return self._minimax(board, not maximizing)

        scores = []
        for move in valid_moves:
            child = board.apply_move(*move, color)
            score = self._minimax(child, not maximizing)
            scores.append(score)

        return max(scores) if maximizing else min(scores)
