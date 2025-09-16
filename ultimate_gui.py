import tkinter as tk
from tkinter import messagebox, ttk
from board import Board
from ai import AdvancedAI
from egaroucid_ai import EgaroucidStyleAI
from ultra_strong_ai import UltraStrongAI
from constants import BLACK, WHITE, EMPTY, opponent, CORNERS
import threading
import time

class UltimateOthelloGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🏆 Ultimate Othello AI - The Unbeatable Champion")
        self.root.geometry("900x900")
        
        self.cell_size = 60
        self.margin = 40
        self.board = Board()
        self.game_over = False
        self.ai_thinking = False
        
        # 마지막 수 및 게임 통계
        self.last_move = None
        self.game_stats = {
            'moves_count': 0,
            'ai_total_time': 0,
            'ai_total_nodes': 0,
            'game_start_time': time.time()
        }
        
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
        """Setup the enhanced user interface"""
        # Title frame
        title_frame = tk.Frame(self.root, bg="#1a1a2e")
        title_frame.pack(fill=tk.X, pady=5)
        
        title_label = tk.Label(title_frame, text="🏆 ULTIMATE OTHELLO AI 🏆", 
                              font=("Arial", 16, "bold"), 
                              fg="#ffd700", bg="#1a1a2e")
        title_label.pack()
        
        subtitle_label = tk.Label(title_frame, text="The Unbeatable Champion", 
                                 font=("Arial", 10, "italic"), 
                                 fg="#ff6b6b", bg="#1a1a2e")
        subtitle_label.pack()
        
        # Control frame
        control_frame = tk.Frame(self.root, bg="#16213e")
        control_frame.pack(pady=10, fill=tk.X)
        
        # New game button (enhanced)
        self.new_game_btn = tk.Button(control_frame, text="🆕 New Game", 
                                     command=self.new_game,
                                     font=("Arial", 12, "bold"), 
                                     bg="#4CAF50", fg="white",
                                     relief=tk.RAISED, bd=3)
        self.new_game_btn.pack(side=tk.LEFT, padx=5)
        
        # AI Type selection (enhanced)
        tk.Label(control_frame, text="🤖 AI Type:", 
                font=("Arial", 10, "bold"), fg="white", bg="#16213e").pack(side=tk.LEFT, padx=5)
        self.ai_type_var = tk.StringVar(value="ultra")
        ai_type_combo = ttk.Combobox(control_frame, textvariable=self.ai_type_var,
                                    values=["advanced", "egaroucid", "ultra"], 
                                    width=12, font=("Arial", 10))
        ai_type_combo.pack(side=tk.LEFT, padx=5)
        
        # Difficulty selection
        tk.Label(control_frame, text="⚡ Power:", 
                font=("Arial", 10, "bold"), fg="white", bg="#16213e").pack(side=tk.LEFT, padx=5)
        self.difficulty_var = tk.StringVar(value="ultra")
        difficulty_combo = ttk.Combobox(control_frame, textvariable=self.difficulty_var,
                                       values=["easy", "medium", "hard", "ultra"], 
                                       width=8, font=("Arial", 10))
        difficulty_combo.pack(side=tk.LEFT, padx=5)
        
        # Time limit selection
        tk.Label(control_frame, text="⏱️ Time:", 
                font=("Arial", 10, "bold"), fg="white", bg="#16213e").pack(side=tk.LEFT, padx=5)
        self.time_limit_var = tk.StringVar(value="5.0")
        time_limit_combo = ttk.Combobox(control_frame, textvariable=self.time_limit_var,
                                       values=["1.0", "3.0", "5.0", "10.0", "30.0"], 
                                       width=6, font=("Arial", 10))
        time_limit_combo.pack(side=tk.LEFT, padx=5)
        
        # Resign button (for when you're getting destroyed)
        self.resign_btn = tk.Button(control_frame, text="🏳️ Resign", 
                                   command=self.resign_game,
                                   font=("Arial", 10), 
                                   bg="#FF6B6B", fg="white",
                                   relief=tk.RAISED, bd=2)
        self.resign_btn.pack(side=tk.RIGHT, padx=5)
        
        # Canvas for the board (enhanced)
        canvas_size = self.cell_size * 8 + self.margin * 2
        self.canvas = tk.Canvas(self.root, width=canvas_size, height=canvas_size,
                               bg="#0f3460", highlightthickness=3,
                               highlightbackground="#ffd700")
        self.canvas.pack(pady=10)
        self.canvas.bind("<Button-1>", self.handle_click)
        self.canvas.bind("<Motion>", self.handle_hover)
        
        # Status frame (enhanced)
        status_frame = tk.Frame(self.root, bg="#1a1a2e")
        status_frame.pack(pady=10, fill=tk.X)
        
        self.status_label = tk.Label(status_frame, text="🎮 Game Start", 
                                   font=("Arial", 14, "bold"), 
                                   fg="#4ECDC4", bg="#1a1a2e")
        self.status_label.pack()
        
        self.score_label = tk.Label(status_frame, text="⚫ Black: 2  ⚪ White: 2",
                                  font=("Arial", 12, "bold"), 
                                  fg="#FFFFFF", bg="#1a1a2e")
        self.score_label.pack()
        
        # AI info label (enhanced)
        self.ai_info_label = tk.Label(status_frame, text="", 
                                     font=("Arial", 10), 
                                     fg="#FFD700", bg="#1a1a2e")
        self.ai_info_label.pack()
        
        # Progress bar (enhanced)
        self.progress = ttk.Progressbar(status_frame, mode='indeterminate', 
                                       length=300, style="gold.Horizontal.TProgressbar")
        self.progress.pack(pady=5)
        
        # Game statistics frame
        stats_frame = tk.Frame(self.root, bg="#16213e", relief=tk.SUNKEN, bd=2)
        stats_frame.pack(pady=5, fill=tk.X, padx=20)
        
        tk.Label(stats_frame, text="📊 Game Statistics", 
                font=("Arial", 10, "bold"), fg="#FFD700", bg="#16213e").pack()
        
        self.stats_label = tk.Label(stats_frame, text="", 
                                   font=("Arial", 9), 
                                   fg="#FFFFFF", bg="#16213e", justify=tk.LEFT)
        self.stats_label.pack()

    def setup_game(self):
        """Setup a new game"""
        self.board = Board()
        self.game_over = False
        self.ai_thinking = False
        self.last_move = None
        
        # Reset statistics
        self.game_stats = {
            'moves_count': 0,
            'ai_total_time': 0,
            'ai_total_nodes': 0,
            'game_start_time': time.time()
        }
        
        # Ask for player preferences
        color_choice = messagebox.askyesno("🎯 Color Selection", 
                                         "Do you want to play as Black (go first)?\n\n" +
                                         "⚠️ Warning: You're about to face the Ultimate AI!\n" +
                                         "Good luck... you'll need it! 😈")
        self.human_color = BLACK if color_choice else WHITE
        self.current_player = BLACK
        
        # Create AI with selected type and difficulty
        ai_type = self.ai_type_var.get()
        difficulty = self.difficulty_var.get()
        time_limit = float(self.time_limit_var.get())
        ai_color = WHITE if self.human_color == BLACK else BLACK
        
        if ai_type == "ultra":
            self.ai = UltraStrongAI(ai_color, difficulty, time_limit)
            ai_name = "🚀 ULTRA STRONG"
            ai_desc = "The Unbeatable Beast"
        elif ai_type == "egaroucid":
            self.ai = EgaroucidStyleAI(ai_color, difficulty, time_limit)
            ai_name = "⚡ Egaroucid-style"
            ai_desc = "Tournament Champion"
        else:
            self.ai = AdvancedAI(ai_color, difficulty, time_limit)
            ai_name = "🎯 Advanced"
            ai_desc = "Classic Strong AI"
        
        # Show intimidation message for Ultra AI
        if ai_type == "ultra":
            messagebox.showwarning("⚠️ ULTIMATE CHALLENGE", 
                                 f"You have chosen to face the {ai_name} AI!\n\n" +
                                 "This AI features:\n" +
                                 "• Perfect endgame solver\n" +
                                 "• Advanced pattern recognition\n" +
                                 "• 18-ply deep search\n" +
                                 "• 5M position transposition table\n" +
                                 "• Tournament-level opening book\n\n" +
                                 "Prepare for the fight of your life! 💀")
        
        self.ai_info_label.config(
            text=f"🤖 AI: {ai_name} ({difficulty.upper()}) - {time_limit}s per move\n{ai_desc}")

    def new_game(self):
        """Start a new game"""
        if self.ai_thinking:
            if messagebox.askyesno("⏸️ AI Thinking", 
                                 "AI is currently thinking. Force new game?"):
                self.ai_thinking = False
            else:
                return
        
        self.setup_game()
        self.update_display()
        if self.current_player != self.human_color:
            self.root.after(500, self.ai_move)
    
    def resign_game(self):
        """Resign the current game"""
        if not self.game_over:
            if messagebox.askyesno("🏳️ Resign Game", 
                                 "Are you sure you want to resign?\n\n" +
                                 "The AI will be declared the winner."):
                self.game_over = True
                ai_color_str = self.color_to_string(self.ai.color)
                self.status_label.config(text=f"🏳️ Game Over: {ai_color_str} Wins by Resignation!")
                messagebox.showinfo("Game Over", f"{ai_color_str} wins by resignation!\n\n" +
                                   "Better luck next time! 😅")

    def draw_board(self):
        """Draw the enhanced game board"""
        self.canvas.delete("all")
        
        # Enhanced board background with gradient effect
        board_start = self.margin
        board_end = self.margin + 8 * self.cell_size
        
        # Outer glow effect
        for i in range(3):
            self.canvas.create_rectangle(board_start - 8 + i*2, board_start - 8 + i*2, 
                                       board_end + 8 - i*2, board_end + 8 - i*2,
                                       fill="", outline="#ffd700", width=1)
        
        # Main board background
        self.canvas.create_rectangle(board_start - 5, board_start - 5, 
                                   board_end + 5, board_end + 5,
                                   fill="#0d7377", outline="#40a4c8", width=3)
        
        # Grid lines with enhanced style
        for i in range(9):
            # Vertical lines
            x = board_start + i * self.cell_size
            self.canvas.create_line(x, board_start, x, board_end,
                                  fill="#2c5f63", width=2)
            # Horizontal lines
            y = board_start + i * self.cell_size
            self.canvas.create_line(board_start, y, board_end, y,
                                  fill="#2c5f63", width=2)
        
        # Enhanced coordinate labels
        for i in range(8):
            x = board_start + i * self.cell_size + self.cell_size // 2
            # Top labels
            self.canvas.create_text(x, board_start - 20, text=chr(ord('a') + i),
                                  font=("Arial", 14, "bold"), fill="#ffd700")
            # Bottom labels
            self.canvas.create_text(x, board_end + 20, text=chr(ord('a') + i),
                                  font=("Arial", 14, "bold"), fill="#ffd700")
        
        # Row labels
        for i in range(8):
            y = board_start + i * self.cell_size + self.cell_size // 2
            # Left labels
            self.canvas.create_text(board_start - 20, y, text=str(i + 1),
                                  font=("Arial", 14, "bold"), fill="#ffd700")
            # Right labels
            self.canvas.create_text(board_end + 20, y, text=str(i + 1),
                                  font=("Arial", 14, "bold"), fill="#ffd700")
        
        # Enhanced guide dots
        guide_dots = [(2, 2), (2, 6), (6, 2), (6, 6)]
        for dot_row, dot_col in guide_dots:
            x = board_start + dot_col * self.cell_size
            y = board_start + dot_row * self.cell_size
            self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, 
                                  fill="#ffd700", outline="#ffed4e", width=2)
        
        # Draw stones with enhanced effects
        for row in range(8):
            for col in range(8):
                if self.board.board[row][col] != EMPTY:
                    self.draw_enhanced_stone(row, col, self.board.board[row][col])
        
        # Draw last move indicator
        if self.last_move:
            self.draw_last_move_indicator()
        
        # Highlight valid moves for human player
        if self.current_player == self.human_color and not self.game_over and not self.ai_thinking:
            valid_moves = self.board.get_valid_moves(self.human_color)
            for move in valid_moves:
                row, col = move
                self.draw_enhanced_move_hint(row, col)

    def draw_enhanced_stone(self, row, col, color):
        """Draw enhanced stone with 3D effect and animations"""
        x = self.margin + col * self.cell_size + self.cell_size // 2
        y = self.margin + row * self.cell_size + self.cell_size // 2
        radius = self.cell_size // 2 - 4
        
        # Multiple shadow layers for depth
        for offset in range(4, 1, -1):
            shadow_alpha = 0.3 - (offset * 0.05)
            self.canvas.create_oval(x - radius + offset, 
                                   y - radius + offset,
                                   x + radius + offset, 
                                   y + radius + offset,
                                   fill="#000000", outline="")
        
        # Main stone with gradient effect
        if color == BLACK:
            # Black stone with metallic effect
            self.canvas.create_oval(x - radius, y - radius,
                                  x + radius, y + radius,
                                  fill="#1a1a1a", outline="#4a4a4a", width=3)
            
            # Multiple highlight layers
            self.canvas.create_oval(x - radius + 6, y - radius + 6,
                                  x - radius + 16, y - radius + 16,
                                  fill="#6a6a6a", outline="")
            self.canvas.create_oval(x - radius + 8, y - radius + 8,
                                  x - radius + 12, y - radius + 12,
                                  fill="#8a8a8a", outline="")
        else:  # WHITE
            # White stone with pearl effect
            self.canvas.create_oval(x - radius, y - radius,
                                  x + radius, y + radius,
                                  fill="#f8f8f8", outline="#d0d0d0", width=3)
            
            # Pearl-like highlights
            self.canvas.create_oval(x - radius + 6, y - radius + 6,
                                  x - radius + 16, y - radius + 16,
                                  fill="#ffffff", outline="")
            self.canvas.create_oval(x - radius + 8, y - radius + 8,
                                  x - radius + 12, y - radius + 12,
                                  fill="#fffffa", outline="")

    def draw_enhanced_move_hint(self, row, col):
        """Draw enhanced hint for valid moves"""
        x = self.margin + col * self.cell_size + self.cell_size // 2
        y = self.margin + row * self.cell_size + self.cell_size // 2
        radius = 10
        
        # Pulsing effect with multiple rings
        for r in range(3):
            ring_radius = radius + r * 3
            alpha = 1.0 - (r * 0.3)
            self.canvas.create_oval(x - ring_radius, y - ring_radius,
                                  x + ring_radius, y + ring_radius,
                                  fill="", outline="#00ff88", width=2)
        
        # Center dot
        self.canvas.create_oval(x - 4, y - 4, x + 4, y + 4,
                              fill="#00ff88", outline="#00cc66", width=2)

    def draw_last_move_indicator(self):
        """Draw enhanced last move indicator"""
        if not self.last_move:
            return
            
        row, col, player_type = self.last_move
        x = self.margin + col * self.cell_size + self.cell_size // 2
        y = self.margin + row * self.cell_size + self.cell_size // 2
        
        # Animated ring around last move
        for ring in range(2):
            ring_radius = 25 + ring * 5
            ring_color = "#ff4444" if player_type == "HU" else "#4444ff"
            self.canvas.create_oval(x - ring_radius, y - ring_radius,
                                  x + ring_radius, y + ring_radius,
                                  fill="", outline=ring_color, width=3)
        
        # Player type indicator with enhanced styling
        text = "👤" if player_type == "HU" else "🤖"
        bg_color = "#ff6b6b" if player_type == "HU" else "#4ecdc4"
        
        # Background circle
        self.canvas.create_oval(x - 15, y - 12, x + 15, y + 12,
                               fill=bg_color, outline="white", width=2)
        
        # Enhanced text
        self.canvas.create_text(x, y, text=text, fill="white", 
                               font=("Arial", 12, "bold"))

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
        """Handle mouse hover with enhanced feedback"""
        if self.current_player != self.human_color or self.game_over or self.ai_thinking:
            self.canvas.configure(cursor="")
            return
            
        row, col = self.get_board_coordinates(event.x, event.y)
        
        if row is not None and col is not None:
            if self.board.is_valid_move(row, col, self.human_color):
                self.canvas.configure(cursor="hand2")
                # Could add preview stone here in future
            else:
                self.canvas.configure(cursor="X_cursor")
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
        self.game_stats['moves_count'] += 1
        
        # Show move in algebraic notation
        move_str = f"{chr(y + ord('a'))}{x + 1}"
        print(f"👤 Human plays: {move_str}")
        
        self.current_player = opponent(self.current_player)
        self.update_display()
        if not self.game_over and self.current_player != self.human_color:
            self.root.after(300, self.ai_move)

    def update_display(self):
        """Update the board display and status"""
        self.draw_board()
        black_count, white_count = self.board.count_stones()
        
        # Enhanced score display with emojis and colors
        score_text = f"⚫ Black: {black_count}  ⚪ White: {white_count}"
        if black_count > white_count:
            score_text += " 📈"
        elif white_count > black_count:
            score_text += " 📉"
        else:
            score_text += " ⚖️"
            
        self.score_label.config(text=score_text)
        
        # Update game statistics
        self.update_game_stats()

        if self.board.get_valid_moves(self.current_player):
            current_color_str = self.color_to_string(self.current_player)
            if self.ai_thinking and self.current_player != self.human_color:
                self.status_label.config(text=f"🤖 AI ({current_color_str}) is calculating...")
            else:
                player_emoji = "👤" if self.current_player == self.human_color else "🤖"
                self.status_label.config(text=f"{player_emoji} {current_color_str}'s Turn")
        else:
            # Check if opponent has moves
            opponent_moves = self.board.get_valid_moves(opponent(self.current_player))
            if not opponent_moves:
                self.end_game(black_count, white_count)
            else:
                current_color_str = self.color_to_string(self.current_player)
                self.status_label.config(text=f"⏭️ {current_color_str} passes turn")
                self.current_player = opponent(self.current_player)
                if self.current_player != self.human_color:
                    self.root.after(1000, self.ai_move)
    
    def end_game(self, black_count, white_count):
        """Handle game end with enhanced results"""
        self.game_over = True
        game_duration = time.time() - self.game_stats['game_start_time']
        
        if black_count > white_count:
            winner = "Black"
            margin = black_count - white_count
            result_emoji = "🏆"
        elif white_count > black_count:
            winner = "White"
            margin = white_count - black_count
            result_emoji = "🏆"
        else:
            winner = "Draw"
            margin = 0
            result_emoji = "🤝"
        
        result_text = f"{result_emoji} Game Over: {winner}"
        if margin > 0:
            result_text += f" wins by {margin}!"
        else:
            result_text += "!"
            
        self.status_label.config(text=result_text)
        
        # Detailed game analysis
        human_color_str = self.color_to_string(self.human_color)
        ai_color_str = self.color_to_string(self.ai.color)
        human_won = (self.human_color == BLACK and black_count > white_count) or \
                   (self.human_color == WHITE and white_count > black_count)
        
        if human_won:
            message = f"🎉 INCREDIBLE! You defeated the AI!\n\n"
            message += f"Final Score: {result_text}\n"
            message += f"You ({human_color_str}): {black_count if self.human_color == BLACK else white_count}\n"
            message += f"AI ({ai_color_str}): {white_count if self.human_color == BLACK else black_count}\n\n"
            message += "You are truly a master player! 👑"
        elif winner == "Draw":
            message = f"🤝 Amazing! You managed a draw!\n\n"
            message += f"Final Score: {black_count} - {white_count}\n"
            message += "Drawing against this AI is a great achievement! 🏅"
        else:
            message = f"🤖 AI Victorious!\n\n"
            message += f"Final Score:\n"
            message += f"AI ({ai_color_str}): {white_count if self.human_color == BLACK else black_count}\n"
            message += f"You ({human_color_str}): {black_count if self.human_color == BLACK else white_count}\n\n"
            message += f"Margin of defeat: {margin} stones\n"
            message += "Don't give up! Try again! 💪"
        
        message += f"\n📊 Game Statistics:\n"
        message += f"• Total moves: {self.game_stats['moves_count']}\n"
        message += f"• Game duration: {game_duration:.1f}s\n"
        message += f"• AI total thinking time: {self.game_stats['ai_total_time']:.1f}s\n"
        if self.game_stats['ai_total_nodes'] > 0:
            message += f"• AI total nodes: {self.game_stats['ai_total_nodes']:,}\n"
        
        messagebox.showinfo("🏁 Game Over", message)
    
    def update_game_stats(self):
        """Update game statistics display"""
        game_duration = time.time() - self.game_stats['game_start_time']
        empty_squares = self.board.get_empty_count()
        
        stats_text = f"Moves: {self.game_stats['moves_count']} | "
        stats_text += f"Empty: {empty_squares} | "
        stats_text += f"Duration: {game_duration:.0f}s"
        
        if self.game_stats['ai_total_time'] > 0:
            stats_text += f" | AI Time: {self.game_stats['ai_total_time']:.1f}s"
        
        if self.game_stats['ai_total_nodes'] > 0:
            avg_nps = self.game_stats['ai_total_nodes'] / max(self.game_stats['ai_total_time'], 0.001)
            stats_text += f" | AI NPS: {avg_nps:,.0f}"
        
        self.stats_label.config(text=stats_text)

    def ai_move(self):
        """Trigger AI move with enhanced feedback"""
        if self.game_over or self.ai_thinking:
            return

        self.ai_thinking = True
        self.progress.start(10)  # Faster animation for more excitement
        
        # Update status with dramatic AI thinking message
        ai_color_str = self.color_to_string(self.ai.color)
        thinking_messages = [
            f"🧠 AI ({ai_color_str}) is calculating the perfect move...",
            f"⚡ AI ({ai_color_str}) analyzing millions of positions...",
            f"🎯 AI ({ai_color_str}) searching for victory...",
            f"🚀 AI ({ai_color_str}) computing at maximum power..."
        ]
        import random
        self.status_label.config(text=random.choice(thinking_messages))

        def think_and_move():
            try:
                start_time = time.time()
                move = self.ai.get_move(self.board)
                ai_time = time.time() - start_time
                
                # Update AI statistics
                self.game_stats['ai_total_time'] += ai_time
                if hasattr(self.ai, 'nodes_searched'):
                    self.game_stats['ai_total_nodes'] += self.ai.nodes_searched
                
                # Schedule UI update on main thread
                def update_ui():
                    self.ai_thinking = False
                    self.progress.stop()
                    
                    if move and not self.game_over:
                        x, y = move
                        move_str = f"{chr(y + ord('a'))}{x + 1}"
                        
                        # Show AI's move with dramatic effect
                        ai_type = self.ai_type_var.get()
                        if ai_type == "ultra":
                            print(f"🚀 ULTRA AI plays: {move_str} (calculated in {ai_time:.2f}s)")
                        else:
                            print(f"🤖 AI plays: {move_str} (calculated in {ai_time:.2f}s)")
                        
                        self.board = self.board.apply_move(x, y, self.ai.color)
                        
                        # Update last move info
                        self.last_move = (x, y, "AI")
                        self.game_stats['moves_count'] += 1
                        
                        self.current_player = opponent(self.current_player)
                    
                    self.update_display()
                
                self.root.after(0, update_ui)
                
            except Exception as e:
                def handle_error():
                    self.ai_thinking = False
                    self.progress.stop()
                    messagebox.showerror("🚨 AI Error", 
                                       f"AI encountered an error: {str(e)}\n\n" +
                                       "The AI may be too powerful for this system! 😅")
                    self.update_display()
                
                self.root.after(0, handle_error)

        threading.Thread(target=think_and_move, daemon=True).start()