import tkinter as tk
from tkinter import messagebox, ttk
from board import Board
from ai import AdvancedAI
from constants import BLACK, WHITE, EMPTY, opponent, CORNERS
import threading

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
        if 0 <= row < 8 and 0 <= col < 8 and self.board.is_valid_move(row, col, self.human_color):
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
        
        # Update status to show AI is thinking
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