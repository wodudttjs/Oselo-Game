import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
import copy
import time
from collections import defaultdict
import threading
import random
import math

EMPTY, BLACK, WHITE = '.', 'B', 'W'

# Enhanced evaluation weights for different game phases
EARLY_WEIGHTS = [
    [120, -20, 20,  5,  5, 20, -20, 120],
    [-20, -40, -5, -5, -5, -5, -40, -20],
    [20,   -5,  3,  2,  2,  3,  -5,  20],
    [5,    -5,  2,  1,  1,  2,  -5,   5],
    [5,    -5,  2,  1,  1,  2,  -5,   5],
    [20,   -5,  3,  2,  2,  3,  -5,  20],
    [-20, -40, -5, -5, -5, -5, -40, -20],
    [120, -20, 20,  5,  5, 20, -20, 120],
]

MID_WEIGHTS = [
    [120, -10, 25, 10, 10, 25, -10, 120],
    [-10, -25,  2,  2,  2,  2, -25, -10],
    [25,    2,  5,  3,  3,  5,   2,  25],
    [10,    2,  3,  1,  1,  3,   2,  10],
    [10,    2,  3,  1,  1,  3,   2,  10],
    [25,    2,  5,  3,  3,  5,   2,  25],
    [-10, -25,  2,  2,  2,  2, -25, -10],
    [120, -10, 25, 10, 10, 25, -10, 120],
]

LATE_WEIGHTS = [
    [100, 20, 30, 20, 20, 30, 20, 100],
    [20,  10, 15, 10, 10, 15, 10,  20],
    [30,  15, 20, 15, 15, 20, 15,  30],
    [20,  10, 15, 10, 10, 15, 10,  20],
    [20,  10, 15, 10, 10, 15, 10,  20],
    [30,  15, 20, 15, 15, 20, 15,  30],
    [20,  10, 15, 10, 10, 15, 10,  20],
    [100, 20, 30, 20, 20, 30, 20, 100],
]

# Strategic positions
X_SQUARES = [(1, 1), (1, 6), (6, 1), (6, 6)]
C_SQUARES = [(0, 1), (1, 0), (0, 6), (6, 0), (7, 1), (6, 7), (7, 6), (1, 7)]
CORNERS = [(0, 0), (0, 7), (7, 0), (7, 7)]
EDGES = [(i, 0) for i in range(8)] + [(i, 7) for i in range(8)] + [(0, i) for i in range(8)] + [(7, i) for i in range(8)]

# Opening book
OPENING_BOOK = {
    # Standard openings
    (BLACK, 3, 3, WHITE, 3, 4, BLACK, 4, 3, WHITE, 4, 4): [(2, 3), (3, 2), (4, 5), (5, 4)],
    (BLACK, 3, 3, WHITE, 3, 4, BLACK, 4, 3, WHITE, 4, 4, BLACK, 2, 3): [(3, 2), (5, 4)],
    (BLACK, 3, 3, WHITE, 3, 4, BLACK, 4, 3, WHITE, 4, 4, BLACK, 3, 2): [(2, 3), (4, 5)],
}

def opponent(color):
    return BLACK if color == WHITE else WHITE

class Board:
    def __init__(self):
        self.size = 8
        self.board = [[EMPTY] * self.size for _ in range(self.size)]
        self.board[3][3] = WHITE
        self.board[3][4] = BLACK
        self.board[4][3] = BLACK
        self.board[4][4] = WHITE
        self.move_history = []

    def in_bounds(self, x, y):
        return 0 <= x < self.size and 0 <= y < self.size

    def get_valid_moves(self, color):
        moves = []
        for x in range(self.size):
            for y in range(self.size):
                if self.board[x][y] != EMPTY:
                    continue
                if self.is_valid_move(x, y, color):
                    moves.append((x, y))
        return moves

    def is_valid_move(self, x, y, color):
        if self.board[x][y] != EMPTY:
            return False
        
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            found_opponent = False
            
            while self.in_bounds(nx, ny) and self.board[nx][ny] == opponent(color):
                found_opponent = True
                nx += dx
                ny += dy
                
            if found_opponent and self.in_bounds(nx, ny) and self.board[nx][ny] == color:
                return True
        return False

    def apply_move(self, x, y, color):
        new_board = copy.deepcopy(self)
        new_board.board[x][y] = color
        flipped = []
        
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            to_flip = []
            
            while new_board.in_bounds(nx, ny) and new_board.board[nx][ny] == opponent(color):
                to_flip.append((nx, ny))
                nx += dx
                ny += dy
                
            if new_board.in_bounds(nx, ny) and new_board.board[nx][ny] == color and to_flip:
                for fx, fy in to_flip:
                    new_board.board[fx][fy] = color
                flipped.extend(to_flip)
            
        new_board.move_history = self.move_history + [(x, y, color, flipped)]
        return new_board

    def count_stones(self):
        b = sum(row.count(BLACK) for row in self.board)
        w = sum(row.count(WHITE) for row in self.board)
        return b, w
        
    def get_empty_count(self):
        return sum(row.count(EMPTY) for row in self.board)
        
    def is_stable(self, x, y):
        """Check if a stone at position (x, y) is stable"""
        if self.board[x][y] == EMPTY:
            return False
            
        color = self.board[x][y]
        
        # Corners are always stable
        if (x, y) in CORNERS:
            return True
            
        # Check stability in all directions
        directions = [
            [(0, 1), (0, -1)],   # horizontal
            [(1, 0), (-1, 0)],   # vertical
            [(1, 1), (-1, -1)],  # diagonal
            [(1, -1), (-1, 1)]   # anti-diagonal
        ]
        
        for dir_pair in directions:
            stable_in_direction = False
            
            for dx, dy in dir_pair:
                nx, ny = x, y
                while True:
                    nx += dx
                    ny += dy
                    if not self.in_bounds(nx, ny):
                        # Reached edge
                        stable_in_direction = True
                        break
                    if self.board[nx][ny] != color:
                        # Found different color or empty
                        break
                    if (nx, ny) in CORNERS:
                        # Connected to corner
                        stable_in_direction = True
                        break
                
                if stable_in_direction:
                    break
                    
            if not stable_in_direction:
                return False
                
        return True

    def get_frontier_count(self, color):
        """Count frontier discs (discs adjacent to empty squares)"""
        frontier = 0
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        for i in range(8):
            for j in range(8):
                if self.board[i][j] == color:
                    for dx, dy in directions:
                        ni, nj = i + dx, j + dy
                        if self.in_bounds(ni, nj) and self.board[ni][nj] == EMPTY:
                            frontier += 1
                            break
        return frontier

    def get_hash(self):
        """Get a hash representation of the board state"""
        return hash(tuple(tuple(row) for row in self.board))

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

class OthelloGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Othello AI")
        self.root.geometry("600x700")
        
        self.cell_size = 60
        self.board = Board()
        self.game_over = False
        self.ai_thinking = False
        
        # Setup UI
        self.setup_ui()
        
        # Game setup
        self.setup_game()
        
        # Start game
        self.update_display()
        if self.current_player != self.human_color:
            self.root.after(500, self.ai_move)

    def setup_ui(self):
        """Setup the user interface"""
        # Control frame
        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=10)
        
        # New game button
        tk.Button(control_frame, text="New Game", command=self.new_game,
                 font=("Arial", 12), bg="lightblue").pack(side=tk.LEFT, padx=5)
        
        # Difficulty selection
        tk.Label(control_frame, text="Difficulty:", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        self.difficulty_var = tk.StringVar(value="medium")
        difficulty_combo = ttk.Combobox(control_frame, textvariable=self.difficulty_var,
                                       values=["easy", "medium", "hard"], width=8)
        difficulty_combo.pack(side=tk.LEFT, padx=5)
        
        # Canvas for the board
        self.canvas = tk.Canvas(self.root, width=self.cell_size * 8, height=self.cell_size * 8,
                               bg="dark green", highlightthickness=2)
        self.canvas.pack(pady=10)
        self.canvas.bind("<Button-1>", self.handle_click)
        self.canvas.bind("<Motion>", self.handle_hover)
        
        # Status frame
        status_frame = tk.Frame(self.root)
        status_frame.pack(pady=10)
        
        self.status_label = tk.Label(status_frame, text="Game Start", 
                                   font=("Arial", 14), fg="blue")
        self.status_label.pack()
        
        self.score_label = tk.Label(status_frame, text="Black: 2  White: 2",
                                  font=("Arial", 12))
        self.score_label.pack()
        
        # Progress bar for AI thinking
        self.progress = ttk.Progressbar(status_frame, mode='indeterminate', length=200)
        self.progress.pack(pady=5)

    def setup_game(self):
        """Setup a new game"""
        self.board = Board()
        self.game_over = False
        self.ai_thinking = False
        
        # Ask for player preferences
        color_choice = messagebox.askyesno("Color Selection", 
                                         "Do you want to play as Black (go first)?")
        self.human_color = BLACK if color_choice else WHITE
        self.current_player = BLACK
        
        # Create AI with selected difficulty
        difficulty = self.difficulty_var.get()
        ai_color = WHITE if self.human_color == BLACK else BLACK
        self.ai = AdvancedAI(ai_color, difficulty)

    def new_game(self):
        """Start a new game"""
        if self.ai_thinking:
            messagebox.showwarning("Please wait", "AI is thinking. Please wait for the move to complete.")
            return
        self.setup_game()
        self.update_display()
        if self.current_player != self.human_color:
            self.root.after(500, self.ai_move)

    def draw_board(self):
        """Draw the game board"""
        self.canvas.delete("all")
        
        # Draw grid
        for i in range(9):
            # Vertical lines
            self.canvas.create_line(i * self.cell_size, 0, i * self.cell_size, 8 * self.cell_size,
                                  fill="black", width=2)
            # Horizontal lines
            self.canvas.create_line(0, i * self.cell_size, 8 * self.cell_size, i * self.cell_size,
                                  fill="black", width=2)
        
        # Draw stones
        for i in range(8):
            for j in range(8):
                x1, y1 = j * self.cell_size + 5, i * self.cell_size + 5
                x2, y2 = x1 + self.cell_size - 10, y1 + self.cell_size - 10
                
                stone = self.board.board[i][j]
                if stone == BLACK:
                    self.canvas.create_oval(x1, y1, x2, y2, fill="black", outline="gray", width=2)
                elif stone == WHITE:
                    self.canvas.create_oval(x1, y1, x2, y2, fill="white", outline="gray", width=2)
        
        # Highlight valid moves for human player
        if self.current_player == self.human_color and not self.game_over and not self.ai_thinking:
            valid_moves = self.board.get_valid_moves(self.human_color)
            for move in valid_moves:
                x, y = move
                cx = y * self.cell_size + self.cell_size // 2
                cy = x * self.cell_size + self.cell_size // 2
                self.canvas.create_oval(cx - 8, cy - 8, cx + 8, cy + 8,
                                      fill="yellow", outline="orange", width=2)

    def handle_hover(self, event):
        """Handle mouse hover to show preview"""
        if self.current_player != self.human_color or self.game_over or self.ai_thinking:
            return
            
        col, row = event.x // self.cell_size, event.y // self.cell_size
        if self.board.is_valid_move(row, col, self.human_color):
            self.canvas.configure(cursor="hand2")
        else:
            self.canvas.configure(cursor="")

    def handle_click(self, event):
        """Handle mouse click on the board"""
        if self.current_player != self.human_color or self.game_over or self.ai_thinking:
            return
            
        col, row = event.x // self.cell_size, event.y // self.cell_size
        
        if 0 <= row < 8 and 0 <= col < 8 and self.board.is_valid_move(row, col, self.human_color):
            self.make_move(row, col, self.human_color)

    def make_move(self, x, y, color):
        """Make a move and update the game state"""
        self.board = self.board.apply_move(x, y, color)
        self.current_player = opponent(self.current_player)
        self.update_display()
    def update_display(self):
        """Update the board display and status"""
        self.draw_board()
        black_count, white_count = self.board.count_stones()
        self.score_label.config(text=f"Black: {black_count}  White: {white_count}")

        if self.board.get_valid_moves(self.current_player):
            self.status_label.config(text=f"{self.current_player}'s Turn")
        else:
            # Check if opponent has moves
            opponent_moves = self.board.get_valid_moves(opponent(self.current_player))
            if not opponent_moves:
                self.game_over = True
                if black_count > white_count:
                    result = "Black Wins!"
                elif white_count > black_count:
                    result = "White Wins!"
                else:
                    result = "Draw!"
                self.status_label.config(text=f"Game Over: {result}")
                messagebox.showinfo("Game Over", result)
            else:
                self.status_label.config(text=f"{self.current_player} passes turn")
                self.current_player = opponent(self.current_player)
                self.root.after(500, self.ai_move if self.current_player != self.human_color else None)

    def ai_move(self):
        """Trigger AI move"""
        if self.game_over:
            return

        self.ai_thinking = True
        self.progress.start()

        def think():
            move = self.ai.get_move(self.board)
            if move:
                x, y = move
                self.board = self.board.apply_move(x, y, self.ai.color)
                self.current_player = opponent(self.current_player)
            self.ai_thinking = False
            self.progress.stop()
            self.update_display()

        threading.Thread(target=think).start()

    def make_move(self, x, y, color):
        """Make a move and update the game state"""
        self.board = self.board.apply_move(x, y, color)
        self.current_player = opponent(self.current_player)
        self.update_display()
        if not self.game_over and self.current_player != self.human_color:
            self.root.after(500, self.ai_move)

if __name__ == "__main__":
    root = tk.Tk()
    app = OthelloGUI(root)
    root.mainloop()
