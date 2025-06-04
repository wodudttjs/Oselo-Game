import tkinter as tk
from tkinter import messagebox, ttk
import threading
import time

# 순환 참조 방지를 위한 조건부 import
try:
    from gpu_board_adapter import BoardAdapter, AIAdapter
    USE_ADAPTER = True
except ImportError:
    USE_ADAPTER = False
    from board import Board
    from ai import AdvancedAI
    try:
        from egaroucid_ai import EgaroucidStyleAI
    except ImportError:
        EgaroucidStyleAI = None
    try:
        from ultra_strong_ai import UltraStrongAI
    except ImportError:
        UltraStrongAI = None

from constants import BLACK, WHITE, EMPTY, opponent, CORNERS

class UltimateOthelloGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🏆 Ultimate Othello AI - The Unbeatable Champion")
        self.root.geometry("900x900")
        self.cell_size = 60
        self.margin = 40
        
        # 보드 초기화 (어댑터 사용 가능시 우선 사용)
        if USE_ADAPTER:
            self.board = BoardAdapter(use_gpu=True)
        else:
            self.board = Board()
            
        self.game_over = False
        self.ai_thinking = False
        
        # 학습 관련 초기화
        self.learning_callback = None
        self.game_data_buffer = []
        
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

    def enable_learning_mode(self, learning_callback):
        """학습 모드 활성화"""
        self.learning_callback = learning_callback
        print("🎓 연속 학습 모드가 활성화되었습니다!")

    def color_to_string(self, color):
        """Convert color constant to readable string"""
        if color == BLACK:
            return "Black"
        elif color == WHITE:
            return "White"
        else:
            return "Empty"

    def setup_game(self):
        """Setup a new game"""
        if USE_ADAPTER:
            self.board = BoardAdapter(use_gpu=True)
        else:
            self.board = Board()
            
        self.game_over = False
        self.ai_thinking = False
        self.last_move = None
        self.game_data_buffer = []  # 게임 데이터 초기화

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

        # AI 생성 (어댑터 사용 가능시 우선 사용)
        if USE_ADAPTER:
            self.ai = AIAdapter(ai_color, ai_type, difficulty, time_limit)
            ai_name = f"🚀 {ai_type.upper()}"
            ai_desc = "GPU Enhanced AI"
        else:
            # 기존 방식
            if ai_type == "ultra" and UltraStrongAI:
                self.ai = UltraStrongAI(ai_color, difficulty, time_limit)
                ai_name = "🚀 ULTRA STRONG"
                ai_desc = "The Unbeatable Beast"
            elif ai_type == "egaroucid" and EgaroucidStyleAI:
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

    def make_move(self, x, y, color):
        """Make a move and update the game state"""
        # 게임 데이터 수집 (학습용)
        if self.learning_callback:
            board_state = self._get_board_state_for_learning()
            self.game_data_buffer.append({
                'board': board_state,
                'move': (x, y),
                'color': color,
                'timestamp': time.time()
            })

        self.board = self.board.apply_move(x, y, color)
        
        # Update last move info
        self.last_move = (x, y, "HU" if color == self.human_color else "AI")
        self.game_stats['moves_count'] += 1

        # Show move in algebraic notation
        move_str = f"{chr(y + ord('a'))}{x + 1}"
        player_type = "👤 Human" if color == self.human_color else "🤖 AI"
        print(f"{player_type} plays: {move_str}")

        self.current_player = opponent(self.current_player)
        self.update_display()

        if not self.game_over and self.current_player != self.human_color:
            self.root.after(300, self.ai_move)

    def _get_board_state_for_learning(self):
        """학습용 보드 상태 추출"""
        if hasattr(self.board, 'board'):
            if isinstance(self.board.board, list):
                return self.board.board
            else:
                # GPU 보드인 경우
                return self.board.board.tolist() if hasattr(self.board.board, 'tolist') else self.board.board
        return [[0]*8 for _ in range(8)]

    def end_game(self, black_count, white_count):
        """Handle game end with enhanced results"""
        self.game_over = True
        game_duration = time.time() - self.game_stats['game_start_time']

        # 학습 데이터 전송
        if self.learning_callback and self.game_data_buffer:
            try:
                # 게임 결과로 가치 라벨링
                human_won = (self.human_color == BLACK and black_count > white_count) or \
                           (self.human_color == WHITE and white_count > black_count)
                
                for data in self.game_data_buffer:
                    if data['color'] == self.human_color:
                        data['value'] = 1.0 if human_won else -1.0
                    else:
                        data['value'] = -1.0 if human_won else 1.0
                
                # 학습 콜백 호출
                self.learning_callback(self.game_data_buffer)
                print(f"🎓 학습 데이터 전송 완료: {len(self.game_data_buffer)}개 수")
            except Exception as e:
                print(f"⚠️ 학습 데이터 전송 실패: {e}")

        # 기존 게임 종료 로직...
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

        # 상세 게임 분석
        human_color_str = self.color_to_string(self.human_color)
        ai_color_str = self.color_to_string(self.ai.color if hasattr(self.ai, 'color') else opponent(self.human_color))
        
        human_won = (self.human_color == BLACK and black_count > white_count) or \
                   (self.human_color == WHITE and white_count > black_count)

        if human_won:
            message = f"🎉 INCREDIBLE! You defeated the AI!\n\n"
        elif winner == "Draw":
            message = f"🤝 Amazing! You managed a draw!\n\n"
        else:
            message = f"🤖 AI Victorious!\n\n"

        message += f"Final Score: {black_count} - {white_count}\n"
        message += f"Game duration: {game_duration:.1f}s\n"
        message += f"Total moves: {self.game_stats['moves_count']}\n"

        messagebox.showinfo("🏁 Game Over", message)

    # 나머지 메서드들은 기존과 동일...
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

        # New game button
        self.new_game_btn = tk.Button(control_frame, text="🆕 New Game",
                                     command=self.new_game,
                                     font=("Arial", 12, "bold"),
                                     bg="#4CAF50", fg="white",
                                     relief=tk.RAISED, bd=3)
        self.new_game_btn.pack(side=tk.LEFT, padx=5)

        # AI Type selection
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

        # Canvas for the board
        canvas_size = self.cell_size * 8 + self.margin * 2
        self.canvas = tk.Canvas(self.root, width=canvas_size, height=canvas_size,
                               bg="#0f3460", highlightthickness=3,
                               highlightbackground="#ffd700")
        self.canvas.pack(pady=10)
        self.canvas.bind("<Button-1>", self.handle_click)
        self.canvas.bind("<Motion>", self.handle_hover)

        # Status frame
        status_frame = tk.Frame(self.root, bg="#1a1a2e")
        status_frame.pack(pady=10, fill=tk.X)

        self.status_label = tk.Label(status_frame, text="🎮 Game Start",
                                    font=("Arial", 14, "bold"),
                                    fg="#4ECDC4", bg="#1a1a2e")
        self.status_label.pack()

        self.score_label = tk.Label(status_frame, text="⚫ Black: 2 ⚪ White: 2",
                                   font=("Arial", 12, "bold"),
                                   fg="#FFFFFF", bg="#1a1a2e")
        self.score_label.pack()

        # AI info label
        self.ai_info_label = tk.Label(status_frame, text="",
                                     font=("Arial", 10),
                                     fg="#FFD700", bg="#1a1a2e")
        self.ai_info_label.pack()

        # Progress bar
        self.progress = ttk.Progressbar(status_frame, mode='indeterminate',
                                       length=300)
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

    def new_game(self):
        """Start a new game"""
        if self.ai_thinking:
            if messagebox.askyesno("⏸️ AI Thinking", "AI is currently thinking. Force new game?"):
                self.ai_thinking = False
            else:
                return

        self.setup_game()
        self.update_display()
        
        if self.current_player != self.human_color:
            self.root.after(500, self.ai_move)

    def draw_board(self):
        """Draw the game board"""
        self.canvas.delete("all")
        
        # Board background
        board_start = self.margin
        board_end = self.margin + 8 * self.cell_size
        
        self.canvas.create_rectangle(board_start - 5, board_start - 5,
                                   board_end + 5, board_end + 5,
                                   fill="#0d7377", outline="#40a4c8", width=3)

        # Grid lines
        for i in range(9):
            x = board_start + i * self.cell_size
            self.canvas.create_line(x, board_start, x, board_end, fill="#2c5f63", width=2)
            
            y = board_start + i * self.cell_size
            self.canvas.create_line(board_start, y, board_end, y, fill="#2c5f63", width=2)

        # Coordinate labels
        for i in range(8):
            x = board_start + i * self.cell_size + self.cell_size // 2
            self.canvas.create_text(x, board_start - 20, text=chr(ord('a') + i),
                                  font=("Arial", 12, "bold"), fill="#ffd700")
            
            y = board_start + i * self.cell_size + self.cell_size // 2
            self.canvas.create_text(board_start - 20, y, text=str(i + 1),
                                  font=("Arial", 12, "bold"), fill="#ffd700")

        # Draw stones
        board_data = self.board.board if hasattr(self.board, 'board') else [[0]*8 for _ in range(8)]
        for row in range(8):
            for col in range(8):
                if board_data[row][col] != EMPTY:
                    self.draw_stone(row, col, board_data[row][col])

        # Highlight valid moves
        if self.current_player == self.human_color and not self.game_over and not self.ai_thinking:
            valid_moves = self.board.get_valid_moves(self.human_color)
            for move in valid_moves:
                row, col = move
                self.draw_move_hint(row, col)

    def draw_stone(self, row, col, color):
        """Draw a stone on the board"""
        x = self.margin + col * self.cell_size + self.cell_size // 2
        y = self.margin + row * self.cell_size + self.cell_size // 2
        radius = self.cell_size // 2 - 4

        if color == BLACK:
            self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius,
                                  fill="#1a1a1a", outline="#4a4a4a", width=2)
        else:  # WHITE
            self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius,
                                  fill="#f8f8f8", outline="#d0d0d0", width=2)

    def draw_move_hint(self, row, col):
        """Draw hint for valid moves"""
        x = self.margin + col * self.cell_size + self.cell_size // 2
        y = self.margin + row * self.cell_size + self.cell_size // 2
        
        self.canvas.create_oval(x - 8, y - 8, x + 8, y + 8,
                              fill="", outline="#00ff88", width=2)

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
            self.canvas.configure(cursor="")
            return

        row, col = self.get_board_coordinates(event.x, event.y)
        if row is not None and col is not None:
            if self.board.is_valid_move(row, col, self.human_color):
                self.canvas.configure(cursor="hand2")
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

    def update_display(self):
        """Update the board display and status"""
        self.draw_board()
        
        black_count, white_count = self.board.count_stones()
        score_text = f"⚫ Black: {black_count} ⚪ White: {white_count}"
        self.score_label.config(text=score_text)

        if self.board.get_valid_moves(self.current_player):
            current_color_str = self.color_to_string(self.current_player)
            if self.ai_thinking and self.current_player != self.human_color:
                self.status_label.config(text=f"🤖 AI ({current_color_str}) is thinking...")
            else:
                player_emoji = "👤" if self.current_player == self.human_color else "🤖"
                self.status_label.config(text=f"{player_emoji} {current_color_str}'s Turn")
        else:
            opponent_moves = self.board.get_valid_moves(opponent(self.current_player))
            if not opponent_moves:
                self.end_game(black_count, white_count)
            else:
                current_color_str = self.color_to_string(self.current_player)
                self.status_label.config(text=f"⏭️ {current_color_str} passes turn")
                self.current_player = opponent(self.current_player)
                if self.current_player != self.human_color:
                    self.root.after(1000, self.ai_move)

    def ai_move(self):
        """Trigger AI move"""
        if self.game_over or self.ai_thinking:
            return

        self.ai_thinking = True
        self.progress.start(10)

        ai_color_str = self.color_to_string(self.ai.color if hasattr(self.ai, 'color') else opponent(self.human_color))
        self.status_label.config(text=f"🤖 AI ({ai_color_str}) is thinking...")

        def think_and_move():
            try:
                start_time = time.time()
                move = self.ai.get_move(self.board)
                ai_time = time.time() - start_time

                self.game_stats['ai_total_time'] += ai_time
                if hasattr(self.ai, 'nodes_searched'):
                    self.game_stats['ai_total_nodes'] += self.ai.nodes_searched

                def update_ui():
                    self.ai_thinking = False
                    self.progress.stop()
                    
                    if move and not self.game_over:
                        x, y = move
                        move_str = f"{chr(y + ord('a'))}{x + 1}"
                        print(f"🤖 AI plays: {move_str} (calculated in {ai_time:.2f}s)")
                        
                        self.make_move(x, y, self.ai.color if hasattr(self.ai, 'color') else opponent(self.human_color))

                self.root.after(0, update_ui)

            except Exception as e:
                def handle_error():
                    self.ai_thinking = False
                    self.progress.stop()
                    messagebox.showerror("🚨 AI Error", f"AI encountered an error: {str(e)}")
                    self.update_display()

                self.root.after(0, handle_error)

        threading.Thread(target=think_and_move, daemon=True).start()
