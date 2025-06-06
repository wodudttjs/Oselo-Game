import tkinter as tk
from tkinter import messagebox, ttk
from board import Board
from ai import AdvancedAI  # 기존 AI
from egaroucid_ai import EgaroucidStyleAI  # 새로운 강화된 AI
from constants import BLACK, WHITE, EMPTY, opponent, CORNERS
import threading

class EnhancedOthelloGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced Othello AI with Egaroucid Techniques")
        self.root.geometry("800x800")
        
        self.cell_size = 60
        self.margin = 40
        self.board = Board()
        self.game_over = False
        self.ai_thinking = False
        
        # 마지막 수 추적
        self.last_move = None
        
        # AI 선택 변수
        self.ai_type = "egaroucid"  # "advanced" 또는 "egaroucid"
        
        # Setup UI
        self.setup_ui()
        
        # Game setup
        self.setup_game()
        
        # Start game
        self.update_display()
        if self.current_player != self.human_color:
            self.root.after(500, self.ai_move)

    def color_to_string(self, color):
        """Convert color constant to readable string"""
        if color == BLACK:
            return "Black"
        elif color == WHITE:
            return "White"
        else:
            return "Empty"

    def setup_ui(self):
        """Setup the user interface"""
        # Control frame
        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=10)
        
        # New game button
        tk.Button(control_frame, text="New Game", command=self.new_game,
                 font=("Arial", 12), bg="lightblue").pack(side=tk.LEFT, padx=5)
        
        # AI Type selection
        tk.Label(control_frame, text="AI Type:", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        self.ai_type_var = tk.StringVar(value="egaroucid")
        ai_type_combo = ttk.Combobox(control_frame, textvariable=self.ai_type_var,
                                    values=["advanced", "egaroucid"], width=10)
        ai_type_combo.pack(side=tk.LEFT, padx=5)
        
        # Difficulty selection
        tk.Label(control_frame, text="Difficulty:", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        self.difficulty_var = tk.StringVar(value="medium")
        difficulty_combo = ttk.Combobox(control_frame, textvariable=self.difficulty_var,
                                       values=["easy", "medium", "hard"], width=8)
        difficulty_combo.pack(side=tk.LEFT, padx=5)
        
        # Time limit selection
        tk.Label(control_frame, text="Time Limit:", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        self.time_limit_var = tk.StringVar(value="3.0")
        time_limit_combo = ttk.Combobox(control_frame, textvariable=self.time_limit_var,
                                       values=["1.0", "2.0", "3.0", "5.0", "10.0"], width=6)
        time_limit_combo.pack(side=tk.LEFT, padx=5)
        
        # Canvas for the board
        canvas_size = self.cell_size * 8 + self.margin * 2
        self.canvas = tk.Canvas(self.root, width=canvas_size, height=canvas_size,
                               bg="#2E8B57", highlightthickness=2)
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
        
        # AI info label
        self.ai_info_label = tk.Label(status_frame, text="", font=("Arial", 10), fg="gray")
        self.ai_info_label.pack()
        
        # Progress bar for AI thinking
        self.progress = ttk.Progressbar(status_frame, mode='indeterminate', length=200)
        self.progress.pack(pady=5)

    def setup_game(self):
        """Setup a new game"""
        self.board = Board()
        self.game_over = False
        self.ai_thinking = False
        self.last_move = None
        
        # Ask for player preferences
        color_choice = messagebox.askyesno("Color Selection", 
                                         "Do you want to play as Black (go first)?")
        self.human_color = BLACK if color_choice else WHITE
        self.current_player = BLACK
        
        # Create AI with selected type and difficulty
        ai_type = self.ai_type_var.get()
        difficulty = self.difficulty_var.get()
        time_limit = float(self.time_limit_var.get())
        ai_color = WHITE if self.human_color == BLACK else BLACK
        
        if ai_type == "egaroucid":
            self.ai = EgaroucidStyleAI(ai_color, difficulty, time_limit)
            ai_name = "Egaroucid-style"
        else:
            self.ai = AdvancedAI(ai_color, difficulty, time_limit)
            ai_name = "Advanced"
        
        self.ai_info_label.config(text=f"AI: {ai_name} ({difficulty}) - {time_limit}s")

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
        """Draw the game board with coordinates"""
        self.canvas.delete("all")
        
        # Board area background
        board_start = self.margin
        board_end = self.margin + 8 * self.cell_size
        
        # Background rectangle
        self.canvas.create_rectangle(board_start - 5, board_start - 5, 
                                   board_end + 5, board_end + 5,
                                   fill="darkgreen", outline="#888", width=2)
        
        # Draw grid lines
        for i in range(9):
            # Vertical lines
            x = board_start + i * self.cell_size
            self.canvas.create_line(x, board_start, x, board_end,
                                  fill="black", width=1)
            # Horizontal lines
            y = board_start + i * self.cell_size
            self.canvas.create_line(board_start, y, board_end, y,
                                  fill="black", width=1)
        
        # Draw coordinate labels
        for i in range(8):
            x = board_start + i * self.cell_size + self.cell_size // 2
            # Top labels
            self.canvas.create_text(x, board_start - 20, text=chr(ord('a') + i),
                                  font=("Arial", 14, "bold"), fill="white")
            # Bottom labels
            self.canvas.create_text(x, board_end + 20, text=chr(ord('a') + i),
                                  font=("Arial", 14, "bold"), fill="white")
        
        # Row labels
        for i in range(8):
            y = board_start + i * self.cell_size + self.cell_size // 2
            # Left labels
            self.canvas.create_text(board_start - 20, y, text=str(i + 1),
                                  font=("Arial", 14, "bold"), fill="white")
            # Right labels
            self.canvas.create_text(board_end + 20, y, text=str(i + 1),
                                  font=("Arial", 14, "bold"), fill="white")
        
        # Draw guide dots
        guide_dots = [(2, 2), (2, 6), (6, 2), (6, 6)]
        for dot_row, dot_col in guide_dots:
            x = board_start + dot_col * self.cell_size
            y = board_start + dot_row * self.cell_size
            self.canvas.create_oval(x - 4, y - 4, x + 4, y + 4, 
                                  fill="black", outline="black")
        
        # Draw stones
        for row in range(8):
            for col in range(8):
                if self.board.board[row][col] != EMPTY:
                    self.draw_stone(row, col, self.board.board[row][col])
        
        # Draw last move indicator
        if self.last_move:
            self.draw_last_move_indicator()
        
        # Highlight valid moves for human player
        if self.current_player == self.human_color and not self.game_over and not self.ai_thinking:
            valid_moves = self.board.get_valid_moves(self.human_color)
            for move in valid_moves:
                row, col = move
                self.draw_valid_move_hint(row, col)

    def draw_stone(self, row, col, color):
        """Draw a stone on the board"""
        x = self.margin + col * self.cell_size + self.cell_size // 2
        y = self.margin + row * self.cell_size + self.cell_size // 2
        radius = self.cell_size // 2 - 5
        
        # Shadow effect
        shadow_offset = 2
        self.canvas.create_oval(x - radius + shadow_offset, 
                               y - radius + shadow_offset,
                               x + radius + shadow_offset, 
                               y + radius + shadow_offset,
                               fill="#888888", outline="")
        
        # Main stone
        if color == BLACK:
            self.canvas.create_oval(x - radius, y - radius,
                                  x + radius, y + radius,
                                  fill="#2C2C2C", outline="#1C1C1C", width=2)
            # Highlight
            self.canvas.create_oval(x - radius + 5, y - radius + 5,
                                  x - radius + 12, y - radius + 12,
                                  fill="#5C5C5C", outline="")
        else:  # WHITE
            self.canvas.create_oval(x - radius, y - radius,
                                  x + radius, y + radius,
                                  fill="#F5F5F5", outline="#CCCCCC", width=2)
            # Highlight
            self.canvas.create_oval(x - radius + 5, y - radius + 5,
                                  x - radius + 12, y - radius + 12,
                                  fill="white", outline="")

    def draw_valid_move_hint(self, row, col):
        """Draw hint for valid moves"""
        x = self.margin + col * self.cell_size + self.cell_size // 2
        y = self.margin + row * self.cell_size + self.cell_size // 2
        radius = 8
        
        # Semi-transparent circle for hint
        self.canvas.create_oval(x - radius, y - radius,
                              x + radius, y + radius,
                              fill="#FFD700", outline="#FFA500", width=2,
                              stipple="gray25")

    def draw_last_move_indicator(self):
        """Draw last move indicator"""
        if not self.last_move:
            return
            
        row, col, player_type = self.last_move
        x = self.margin + col * self.cell_size + self.cell_size // 2
        y = self.margin + row * self.cell_size + self.cell_size // 2
        
        # Player type indicator
        text = player_type
        if self.board.board[row][col] == BLACK:
            text_color = "white"
            bg_color = "#FF4444" if player_type == "HU" else "#4444FF"
        else:
            text_color = "black"
            bg_color = "#FF4444" if player_type == "HU" else "#4444FF"
        
        # Background circle
        self.canvas.create_oval(x - 12, y - 8, x + 12, y + 8,
                               fill=bg_color, outline="", stipple="gray50")
        
        # Text
        self.canvas.create_text(x, y, text=text, fill=text_color, 
                               font=("Arial", 8, "bold"))

    def get_board_coordinates(self, event_x, event_y):
        """Convert canvas coordinates to board coordinates"""
        board_x = event_x - self.margin
        board_y = event_y - self.margin
        
        if board_x < 0 or board_y < 0:
            return None, None
        
        col = board_x // self.cell_size
        row = board_y // self.cell_size
        
        if 0 <= row < 8 and 0 <= col < 8:
            return row, col
        return None, None

    def handle_hover(self, event):
        """Handle mouse hover"""
        if self.current_player != self.human_color or self.game_over or self.ai_thinking:
            return
            
        row, col = self.get_board_coordinates(event.x, event.y)
        
        if row is not None and col is not None:
            if self.board.is_valid_move(row, col, self.human_color):
                self.canvas.configure(cursor="hand2")
            else:
                self.canvas.configure(cursor="")
        else:
            self.canvas.configure(cursor="")

    def handle_click(self, event):
        """Handle mouse click on the board"""
        if self.current_player != self.human_color or self.game_over or self.ai_thinking:
            return
            
        row, col = self.get_board_coordinates(event.x, event.y)
        
        if row is not None and col is not None:
            if self.board.is_valid_move(row, col, self.human_color):
                self.make_move(row, col, self.human_color)

    def make_move(self, x, y, color):
        """Make a move and update the game state"""
        self.board = self.board.apply_move(x, y, color)
        
        # Update last move info
        self.last_move = (x, y, "HU")
        
        self.current_player = opponent(self.current_player)
        self.update_display()
        if not self.game_over and self.current_player != self.human_color:
            self.root.after(500, self.ai_move)

    def update_display(self):
        """Update the board display and status"""
        self.draw_board()
        black_count, white_count = self.board.count_stones()
        self.score_label.config(text=f"Black: {black_count}  White: {white_count}")

        if self.board.get_valid_moves(self.current_player):
            current_color_str = self.color_to_string(self.current_player)
            if self.ai_thinking and self.current_player != self.human_color:
                self.status_label.config(text=f"AI ({current_color_str}) is thinking...")
            else:
                self.status_label.config(text=f"{current_color_str}'s Turn")
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
                current_color_str = self.color_to_string(self.current_player)
                self.status_label.config(text=f"{current_color_str} passes turn")
                self.current_player = opponent(self.current_player)
                if self.current_player != self.human_color:
                    self.root.after(1000, self.ai_move)

    def ai_move(self):
        """Trigger AI move in a separate thread"""
        if self.game_over or self.ai_thinking:
            return

        self.ai_thinking = True
        self.progress.start()
        
        # Update status
        ai_color_str = self.color_to_string(self.ai.color)
        self.status_label.config(text=f"AI ({ai_color_str}) is thinking...")

        def think_and_move():
            try:
                move = self.ai.get_move(self.board)
                
                # Schedule UI update on main thread
                def update_ui():
                    self.ai_thinking = False
                    self.progress.stop()
                    
                    if move and not self.game_over:
                        x, y = move
                        self.board = self.board.apply_move(x, y, self.ai.color)
                        
                        # Update last move info
                        self.last_move = (x, y, "AI")
                        
                        self.current_player = opponent(self.current_player)
                    
                    self.update_display()
                
                self.root.after(0, update_ui)
                
            except Exception as e:
                def handle_error():
                    self.ai_thinking = False
                    self.progress.stop()
                    messagebox.showerror("AI Error", f"AI encountered an error: {str(e)}")
                    self.update_display()
                
                self.root.after(0, handle_error)

        threading.Thread(target=think_and_move, daemon=True).start()