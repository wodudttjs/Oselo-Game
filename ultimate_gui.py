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
        self.root.geometry("1000x1000")  # 크기 증가
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

    # ultimate_gui.py의 setup_game 메서드를 이것으로 교체하세요

    def setup_game(self):
        """Setup a new game with advanced AI configuration"""
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

        # Collect AI configuration from GUI
        ai_backend = self.ai_backend_var.get()
        ai_algorithm = self.ai_algorithm_var.get()
        difficulty = self.difficulty_var.get()
        time_limit = float(self.time_limit_var.get())
        search_depth = self.search_depth_var.get()
        use_opening_book = self.use_opening_book_var.get()
        use_endgame_solver = self.use_endgame_solver_var.get()
        ai_color = WHITE if self.human_color == BLACK else WHITE

        print(f"\n🤖 === AI 설정 ===")
        print(f"Backend: {ai_backend}")
        print(f"Algorithm: {ai_algorithm}")
        print(f"Difficulty: {difficulty}")
        print(f"Time Limit: {time_limit}s")
        print(f"Search Depth: {search_depth}")
        print(f"Opening Book: {use_opening_book}")
        print(f"Perfect Endgame: {use_endgame_solver}")
        print(f"===================\n")

        # Create AI with advanced configuration
        if USE_ADAPTER:
            # 백엔드와 알고리즘을 조합한 AI 타입 결정
            if ai_backend == "gpu" and ai_algorithm == "neural":
                ai_type = "neural"
            elif ai_backend == "gpu" and ai_algorithm == "mcts":
                ai_type = "mcts"
            elif ai_backend == "gpu":
                ai_type = "gpu"
            else:
                ai_type = "cpu"
            
            # 추가 설정 옵션들을 kwargs로 전달
            ai_kwargs = {
                'backend': ai_backend,
                'algorithm': ai_algorithm,
                'search_depth': search_depth,
                'use_opening_book': use_opening_book,
                'use_endgame_solver': use_endgame_solver
            }
            
            self.ai = AIAdapter(ai_color, ai_type, difficulty, time_limit, **ai_kwargs)
            
            # AI 정보 문자열 생성
            ai_name = f"🚀 {ai_backend.upper()}-{ai_algorithm.upper()}"
            if ai_algorithm == "neural":
                ai_desc = f"Neural Network AI ({ai_backend.upper()})"
            elif ai_algorithm == "mcts":
                ai_desc = f"Monte Carlo Tree Search ({ai_backend.upper()})"
            else:
                ai_desc = f"Alpha-Beta Search ({ai_backend.upper()})"
                
        else:
            # 기존 방식 (어댑터 없는 경우)
            if ai_backend == "gpu" and UltraStrongAI:
                use_neural = (ai_algorithm == "neural")
                self.ai = UltraStrongAI(ai_color, difficulty, time_limit, use_neural_net=use_neural)
                ai_name = f"🚀 GPU-{ai_algorithm.upper()}"
                ai_desc = "GPU Accelerated AI"
            elif ai_algorithm == "egaroucid" and EgaroucidStyleAI:
                self.ai = EgaroucidStyleAI(ai_color, difficulty, time_limit)
                ai_name = "⚡ Egaroucid-style"
                ai_desc = "Tournament Champion"
            else:
                self.ai = AdvancedAI(ai_color, difficulty, time_limit)
                ai_name = "🎯 Advanced"
                ai_desc = "Classic Strong AI"

        # Show comprehensive AI configuration message
        config_msg = f"🤖 AI Configuration Complete!\n\n"
        config_msg += f"📋 Settings:\n"
        config_msg += f"• Backend: {ai_backend.upper()}\n"
        config_msg += f"• Algorithm: {ai_algorithm.upper()}\n"
        config_msg += f"• Difficulty: {difficulty.upper()}\n"
        config_msg += f"• Time Limit: {time_limit}s per move\n"
        
        if search_depth != 'auto':
            config_msg += f"• Search Depth: {search_depth} plies\n"
        else:
            config_msg += f"• Search Depth: Automatic\n"
        
        config_msg += f"• Opening Book: {'Enabled' if use_opening_book else 'Disabled'}\n"
        config_msg += f"• Perfect Endgame: {'Enabled' if use_endgame_solver else 'Disabled'}\n\n"
        
        # 백엔드별 특징 설명
        if ai_backend == "gpu":
            config_msg += "🚀 GPU Acceleration Features:\n"
            config_msg += "• Parallel board evaluation (10x+ faster)\n"
            config_msg += "• Batch move generation (5x+ faster)\n"
            config_msg += "• High-speed transposition table\n"
            config_msg += "• Vectorized pattern recognition\n"
            
            if ai_algorithm == "neural":
                config_msg += "• Deep neural network MCTS\n"
                config_msg += "• Self-play continuous learning\n"
                config_msg += "• Advanced position understanding\n"
            elif ai_algorithm == "mcts":
                config_msg += "• Monte Carlo Tree Search\n"
                config_msg += "• Statistical move evaluation\n"
                config_msg += "• Exploration vs exploitation balance\n"
            else:
                config_msg += "• Deep alpha-beta pruning\n"
                config_msg += "• Perfect endgame solver\n"
                
            config_msg += "\n⚡ Performance: Lightning-fast calculations!"
            
        else:
            config_msg += "💻 CPU Optimization Features:\n"
            config_msg += "• Multi-depth iterative deepening\n"
            config_msg += "• Advanced alpha-beta pruning\n"
            config_msg += "• Sophisticated evaluation function\n"
            config_msg += "• Opening book integration\n"
            
            if use_endgame_solver:
                config_msg += "• Perfect endgame calculations\n"
                
            config_msg += "\n💪 Performance: Highly optimized for CPU!"

        # 난이도별 예상 성능
        config_msg += f"\n\n🎯 Expected Performance ({difficulty.upper()}):\n"
        if difficulty == "easy":
            config_msg += "• Depth: 4-6 plies\n• Thinking time: 0.1-1s\n• Strength: Beginner friendly"
        elif difficulty == "medium":
            config_msg += "• Depth: 6-8 plies\n• Thinking time: 0.5-2s\n• Strength: Intermediate challenge"
        elif difficulty == "hard":
            config_msg += "• Depth: 8-12 plies\n• Thinking time: 1-5s\n• Strength: Advanced opponent"
        else:  # ultra
            config_msg += "• Depth: 12-20 plies\n• Thinking time: 2-10s\n• Strength: Master level"

        config_msg += f"\n\n⚠️ Warning: This AI is extremely powerful!\n"
        config_msg += f"Estimated playing strength: {self._estimate_elo_rating()} ELO\n"
        config_msg += f"Good luck! 🍀"

        response = messagebox.askyesno("🤖 AI Ready", config_msg + "\n\nStart the game?")
        
        if not response:
            # 사용자가 취소한 경우 설정 다시 하기
            self.setup_game()
            return

        # AI 정보 라벨 업데이트
        self.ai_info_label.config(
            text=f"🤖 AI: {ai_name} ({difficulty.upper()}) - {time_limit}s\n"
                f"{ai_desc} | Depth: {search_depth}")

    def _estimate_elo_rating(self):
        """AI 설정에 기반한 대략적인 ELO 레이팅 추정"""
        base_rating = 1200  # 기본 레이팅
        
        # 난이도별 보정
        difficulty_bonus = {
            'easy': 0,
            'medium': 300,
            'hard': 600,
            'ultra': 900
        }
        
        # 백엔드별 보정
        backend_bonus = {
            'cpu': 0,
            'gpu': 200
        }
        
        # 알고리즘별 보정
        algorithm_bonus = {
            'alphabeta': 0,
            'neural': 150,
            'mcts': 100
        }
        
        # 시간 제한별 보정
        time_bonus = min(float(self.time_limit_var.get()) * 50, 300)
        
        # 특수 기능별 보정
        feature_bonus = 0
        if self.use_endgame_solver_var.get():
            feature_bonus += 100
        if self.use_opening_book_var.get():
            feature_bonus += 50
        
        total_rating = (base_rating + 
                    difficulty_bonus.get(self.difficulty_var.get(), 0) +
                    backend_bonus.get(self.ai_backend_var.get(), 0) +
                    algorithm_bonus.get(self.ai_algorithm_var.get(), 0) +
                    time_bonus + feature_bonus)
        
        return int(total_rating)
    
    def setup_ui(self):
        """Setup the enhanced user interface with AI options"""
        # Title frame
        title_frame = tk.Frame(self.root, bg="#1a1a2e")
        title_frame.pack(fill=tk.X, pady=5)

        title_label = tk.Label(title_frame, text="🏆 ULTIMATE OTHELLO AI 🏆",
                              font=("Arial", 16, "bold"),
                              fg="#ffd700", bg="#1a1a2e")
        title_label.pack()

        subtitle_label = tk.Label(title_frame, text="Choose Your AI Challenge",
                                 font=("Arial", 10, "italic"),
                                 fg="#ff6b6b", bg="#1a1a2e")
        subtitle_label.pack()

        # AI Configuration frame
        config_frame = tk.LabelFrame(self.root, text="🤖 AI Configuration", 
                                   font=("Arial", 12, "bold"),
                                   fg="#ffd700", bg="#16213e", 
                                   relief=tk.RAISED, bd=2)
        config_frame.pack(pady=10, fill=tk.X, padx=20)

        # First row: Backend and Algorithm
        row1_frame = tk.Frame(config_frame, bg="#16213e")
        row1_frame.pack(pady=5, fill=tk.X)

        # AI Backend selection
        tk.Label(row1_frame, text="🖥️ Backend:",
                font=("Arial", 10, "bold"), fg="white", bg="#16213e").pack(side=tk.LEFT, padx=5)
        
        self.ai_backend_var = tk.StringVar(value="gpu")
        backend_combo = ttk.Combobox(row1_frame, textvariable=self.ai_backend_var,
                                   values=["cpu", "gpu"], width=8, font=("Arial", 10))
        backend_combo.pack(side=tk.LEFT, padx=5)
        backend_combo.bind('<<ComboboxSelected>>', self.on_backend_change)

        # AI Algorithm selection  
        tk.Label(row1_frame, text="🧠 Algorithm:",
                font=("Arial", 10, "bold"), fg="white", bg="#16213e").pack(side=tk.LEFT, padx=(20,5))
        
        self.ai_algorithm_var = tk.StringVar(value="alphabeta")
        self.algorithm_combo = ttk.Combobox(row1_frame, textvariable=self.ai_algorithm_var,
                                          values=["alphabeta", "neural"], width=12, font=("Arial", 10))
        self.algorithm_combo.pack(side=tk.LEFT, padx=5)

        # Second row: Difficulty and Time
        row2_frame = tk.Frame(config_frame, bg="#16213e")
        row2_frame.pack(pady=5, fill=tk.X)

        # Difficulty selection
        tk.Label(row2_frame, text="⚡ Power:",
                font=("Arial", 10, "bold"), fg="white", bg="#16213e").pack(side=tk.LEFT, padx=5)
        
        self.difficulty_var = tk.StringVar(value="hard")
        difficulty_combo = ttk.Combobox(row2_frame, textvariable=self.difficulty_var,
                                       values=["easy", "medium", "hard", "ultra"],
                                       width=8, font=("Arial", 10))
        difficulty_combo.pack(side=tk.LEFT, padx=5)

        # Time limit selection
        tk.Label(row2_frame, text="⏱️ Time:",
                font=("Arial", 10, "bold"), fg="white", bg="#16213e").pack(side=tk.LEFT, padx=(20,5))
        
        self.time_limit_var = tk.StringVar(value="5.0")
        time_limit_combo = ttk.Combobox(row2_frame, textvariable=self.time_limit_var,
                                       values=["1.0", "2.0", "5.0", "10.0", "30.0"],
                                       width=6, font=("Arial", 10))
        time_limit_combo.pack(side=tk.LEFT, padx=5)

        # Third row: Search depth and special options
        row3_frame = tk.Frame(config_frame, bg="#16213e")
        row3_frame.pack(pady=5, fill=tk.X)

        # Search depth selection
        tk.Label(row3_frame, text="🔍 Depth:",
                font=("Arial", 10, "bold"), fg="white", bg="#16213e").pack(side=tk.LEFT, padx=5)
        
        self.search_depth_var = tk.StringVar(value="auto")
        depth_combo = ttk.Combobox(row3_frame, textvariable=self.search_depth_var,
                                  values=["auto", "8", "12", "16", "20"],
                                  width=8, font=("Arial", 10))
        depth_combo.pack(side=tk.LEFT, padx=5)

        # Special features
        self.use_opening_book_var = tk.BooleanVar(value=True)
        opening_check = tk.Checkbutton(row3_frame, text="📚 Opening Book",
                                     variable=self.use_opening_book_var,
                                     font=("Arial", 9), fg="white", bg="#16213e",
                                     selectcolor="#2c5f63")
        opening_check.pack(side=tk.LEFT, padx=10)

        self.use_endgame_solver_var = tk.BooleanVar(value=True)
        endgame_check = tk.Checkbutton(row3_frame, text="🏁 Perfect Endgame",
                                     variable=self.use_endgame_solver_var,
                                     font=("Arial", 9), fg="white", bg="#16213e",
                                     selectcolor="#2c5f63")
        endgame_check.pack(side=tk.LEFT, padx=10)

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

        # AI Hint button
        self.hint_btn = tk.Button(control_frame, text="💡 AI Hint",
                                 command=self.get_ai_hint,
                                 font=("Arial", 10, "bold"),
                                 bg="#FF9800", fg="white",
                                 relief=tk.RAISED, bd=2)
        self.hint_btn.pack(side=tk.LEFT, padx=5)

        # Performance button
        self.perf_btn = tk.Button(control_frame, text="📊 Performance",
                                 command=self.show_performance,
                                 font=("Arial", 10, "bold"),
                                 bg="#9C27B0", fg="white",
                                 relief=tk.RAISED, bd=2)
        self.perf_btn.pack(side=tk.LEFT, padx=5)

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

    def on_backend_change(self, event=None):
        """Backend 변경시 알고리즘 옵션 업데이트"""
        backend = self.ai_backend_var.get()
        if backend == "gpu":
            self.algorithm_combo['values'] = ["alphabeta", "neural", "mcts"]
            if self.ai_algorithm_var.get() not in ["alphabeta", "neural", "mcts"]:
                self.ai_algorithm_var.set("alphabeta")
        else:
            self.algorithm_combo['values'] = ["alphabeta", "minimax"]
            if self.ai_algorithm_var.get() not in ["alphabeta", "minimax"]:
                self.ai_algorithm_var.set("alphabeta")

    def get_ai_hint(self):
        """AI 힌트 제공"""
        if self.ai_thinking or self.game_over:
            return

        if self.current_player != self.human_color:
            messagebox.showinfo("💡 Hint", "It's not your turn!")
            return

        self.hint_btn.config(state='disabled', text="💭 Thinking...")
        
        def get_hint():
            try:
                # 임시 AI로 힌트 계산
                hint_ai = AIAdapter(self.human_color, 'auto', 'medium', 2.0) if USE_ADAPTER else None
                if hint_ai:
                    hint_move = hint_ai.get_move(self.board)
                    if hint_move:
                        move_str = f"{chr(hint_move[1] + ord('a'))}{hint_move[0] + 1}"
                        
                        def show_hint():
                            messagebox.showinfo("💡 AI Hint", 
                                              f"Suggested move: {move_str}\n\n" +
                                              "This is a medium-strength suggestion.\n" +
                                              "Consider your own strategy too! 🤔")
                            self.hint_btn.config(state='normal', text="💡 AI Hint")
                        
                        self.root.after(0, show_hint)
                    else:
                        def no_hint():
                            messagebox.showinfo("💡 AI Hint", "No valid moves available!")
                            self.hint_btn.config(state='normal', text="💡 AI Hint")
                        self.root.after(0, no_hint)
                        
            except Exception as e:
                def hint_error():
                    messagebox.showerror("💡 Hint Error", f"Could not generate hint: {e}")
                    self.hint_btn.config(state='normal', text="💡 AI Hint")
                self.root.after(0, hint_error)

        threading.Thread(target=get_hint, daemon=True).start()

    def show_performance(self):
        """AI 성능 정보 표시"""
        try:
            perf_info = {}
            if hasattr(self.ai, 'get_performance_info'):
                perf_info = self.ai.get_performance_info()
            
            board_info = {}
            if hasattr(self.board, 'get_performance_info'):
                board_info = self.board.get_performance_info()
            
            perf_text = "📊 Performance Information\n\n"
            
            # AI 정보
            perf_text += "🤖 AI Performance:\n"
            perf_text += f"  Backend: {perf_info.get('ai_type', 'unknown')}\n"
            perf_text += f"  GPU: {'Yes' if perf_info.get('use_gpu', False) else 'No'}\n"
            perf_text += f"  Difficulty: {perf_info.get('difficulty', 'unknown')}\n"
            perf_text += f"  Time Limit: {perf_info.get('time_limit', 'unknown')}s\n"
            
            if 'nodes_searched' in perf_info:
                perf_text += f"  Last Search Nodes: {perf_info['nodes_searched']:,}\n"
            if 'tt_hits' in perf_info:
                perf_text += f"  TT Hits: {perf_info['tt_hits']:,}\n"
            
            # 보드 정보
            perf_text += "\n🎯 Board Performance:\n"
            perf_text += f"  Backend: {board_info.get('backend', 'unknown')}\n"
            perf_text += f"  GPU Available: {'Yes' if board_info.get('gpu_available', False) else 'No'}\n"
            
            if 'gpu_memory_used_mb' in board_info:
                perf_text += f"  GPU Memory: {board_info['gpu_memory_used_mb']:.1f} MB\n"
            
            # 게임 통계
            perf_text += "\n📈 Game Statistics:\n"
            perf_text += f"  Moves Played: {self.game_stats['moves_count']}\n"
            perf_text += f"  AI Total Time: {self.game_stats['ai_total_time']:.1f}s\n"
            perf_text += f"  AI Total Nodes: {self.game_stats['ai_total_nodes']:,}\n"
            
            if self.game_stats['ai_total_time'] > 0:
                avg_nps = self.game_stats['ai_total_nodes'] / self.game_stats['ai_total_time']
                perf_text += f"  Average NPS: {avg_nps:,.0f}\n"
            
            messagebox.showinfo("📊 Performance", perf_text)
            
        except Exception as e:
            messagebox.showerror("📊 Performance Error", f"Could not get performance info: {e}")

    # 나머지 메서드들은 기존과 동일...
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

    def make_move(self, x, y, color):
        """Make a move and update the game state"""
        print(f"📝 수 적용: {chr(y + ord('a'))}{x + 1} ({'흑' if color == BLACK else '백'})")
        
        # 게임 데이터 수집 (학습용)
        if self.learning_callback:
            board_state = self._get_board_state_for_learning()
            self.game_data_buffer.append({
                'board': board_state,
                'move': (x, y),
                'color': color,
                'timestamp': time.time()
            })

        # 수 적용 전 돌 개수
        old_black, old_white = self.board.count_stones()
        
        self.board = self.board.apply_move(x, y, color)
        
        # 수 적용 후 돌 개수
        new_black, new_white = self.board.count_stones()
        flipped_count = (new_black + new_white) - (old_black + old_white) - 1
        
        print(f"🔄 뒤집힌 돌: {flipped_count}개")
        print(f"📊 현재 점수: 흑={new_black}, 백={new_white}")
        
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

    def ai_move(self):
        """Trigger AI move - 향상된 버전"""
        if self.game_over or self.ai_thinking:
            return

        self.ai_thinking = True
        self.progress.start(10)

        ai_color_str = self.color_to_string(self.ai.color if hasattr(self.ai, 'color') else opponent(self.human_color))
        self.status_label.config(text=f"🤖 AI ({ai_color_str}) is thinking...")

        def think_and_move():
            try:
                print(f"\n🤖 === AI 턴 시작 ({ai_color_str}) ===")
                start_time = time.time()
                
                # 현재 설정 출력
                backend = self.ai_backend_var.get()
                algorithm = self.ai_algorithm_var.get()
                difficulty = self.difficulty_var.get()
                print(f"🔧 AI 설정: {backend.upper()}-{algorithm.upper()}, 난이도: {difficulty}")
                
                # 현재 보드 상태 출력
                valid_moves = self.board.get_valid_moves(self.ai.color if hasattr(self.ai, 'color') else opponent(self.human_color))
                print(f"🎯 AI 유효한 수: {len(valid_moves)}개")
                if len(valid_moves) <= 10:  # 너무 많지 않으면 모든 수 출력
                    moves_str = [f"{chr(m[1] + ord('a'))}{m[0] + 1}" for m in valid_moves]
                    print(f"📋 가능한 수들: {', '.join(moves_str)}")
                
                move = self.ai.get_move(self.board)
                ai_time = time.time() - start_time

                self.game_stats['ai_total_time'] += ai_time
                if hasattr(self.ai, 'nodes_searched'):
                    self.game_stats['ai_total_nodes'] += getattr(self.ai, 'nodes_searched', 0)

                def update_ui():
                    self.ai_thinking = False
                    self.progress.stop()
                    
                    if move and not self.game_over:
                        x, y = move
                        move_str = f"{chr(y + ord('a'))}{x + 1}"
                        
                        # 상세한 AI 수 정보 출력
                        print(f"🎯 AI 최종 선택: {move_str}")
                        print(f"⏱️ 계산 시간: {ai_time:.2f}초")
                        
                        # AI 통계 출력
                        if hasattr(self.ai, 'nodes_searched'):
                            nodes = getattr(self.ai, 'nodes_searched', 0)
                            if nodes > 0:
                                nps = nodes / ai_time if ai_time > 0 else 0
                                print(f"🌳 탐색 노드: {nodes:,}개")
                                print(f"🚀 초당 노드: {nps:,.0f} NPS")
                        
                        print(f"🤖 === AI 턴 완료 ===\n")
                        
                        self.make_move(x, y, self.ai.color if hasattr(self.ai, 'color') else opponent(self.human_color))
                    else:
                        print("❌ AI가 유효한 수를 찾지 못했습니다")
                        self.update_display()

                self.root.after(0, update_ui)

            except Exception as e:
                def handle_error():
                    self.ai_thinking = False
                    self.progress.stop()
                    print(f"❌ AI 오류: {str(e)}")
                    messagebox.showerror("🚨 AI Error", f"AI encountered an error: {str(e)}")
                    self.update_display()

                self.root.after(0, handle_error)

        threading.Thread(target=think_and_move, daemon=True).start()

    # 나머지 메서드들은 기존과 동일하므로 생략...
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

        # 실시간 통계 업데이트
        self.update_stats_display()

    def update_stats_display(self):
        """실시간 통계 표시 업데이트"""
        try:
            game_duration = time.time() - self.game_stats['game_start_time']
            
            stats_text = f"Moves: {self.game_stats['moves_count']} | "
            stats_text += f"Duration: {game_duration:.1f}s"
            
            if self.game_stats['ai_total_time'] > 0:
                avg_ai_time = self.game_stats['ai_total_time'] / max(1, self.game_stats['moves_count'] // 2)
                stats_text += f" | AI Avg: {avg_ai_time:.2f}s"
            
            if self.game_stats['ai_total_nodes'] > 0:
                stats_text += f" | Total Nodes: {self.game_stats['ai_total_nodes']:,}"
            
            self.stats_label.config(text=stats_text)
        except Exception as e:
            self.stats_label.config(text="Stats unavailable")

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

        # 게임 결과 분석
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

        # 상세 통계
        message += f"Final Score: {black_count} - {white_count}\n"
        message += f"Game Duration: {game_duration:.1f}s\n"
        message += f"Total Moves: {self.game_stats['moves_count']}\n"
        message += f"AI Configuration: {self.ai_backend_var.get().upper()}-{self.ai_algorithm_var.get().upper()}\n"
        message += f"Difficulty: {self.difficulty_var.get().upper()}\n"
        
        if self.game_stats['ai_total_time'] > 0:
            avg_ai_time = self.game_stats['ai_total_time'] / max(1, self.game_stats['moves_count'] // 2)
            message += f"AI Average Time: {avg_ai_time:.2f}s\n"
        
        if self.game_stats['ai_total_nodes'] > 0:
            message += f"AI Total Nodes: {self.game_stats['ai_total_nodes']:,}\n"
            avg_nps = self.game_stats['ai_total_nodes'] / self.game_stats['ai_total_time']
            message += f"Average NPS: {avg_nps:,.0f}\n"

        messagebox.showinfo("🏁 Game Over", message)