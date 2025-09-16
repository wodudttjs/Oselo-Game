import tkinter as tk
from tkinter import messagebox, ttk
from board import Board
from ai import AdvancedAI
# from egaroucid_ai import EgaroucidStyleAI  # 오류 방지를 위해 주석 처리
# from ultra_strong_ai import UltraStrongAI   # 오류 방지를 위해 주석 처리
from constants import BLACK, WHITE, EMPTY, opponent, CORNERS
import threading
import time
import hashlib
import random

# 간소화된 강력한 AI 클래스 (수정된 버전)
class UltimateAI:
    """최종 강력한 AI - 오류 수정 버전"""
    
    def __init__(self, color, difficulty='hard', time_limit=5.0):
        self.color = color
        self.difficulty = difficulty
        self.time_limit = time_limit
        
        # 난이도별 설정
        if difficulty == 'easy':
            self.max_depth = 4
            self.time_limit = min(time_limit, 2.0)
        elif difficulty == 'medium':
            self.max_depth = 6
            self.time_limit = min(time_limit, 3.0)
        elif difficulty == 'hard':
            self.max_depth = 8
            self.time_limit = min(time_limit, 5.0)
        else:  # ultimate
            self.max_depth = 12
            self.time_limit = time_limit
        
        # 안정적인 TT 구조 (수정됨)
        self.tt = {}
        self.tt_age = 0
        self.max_tt_size = 100000
        
        # 통계
        self.nodes_searched = 0
        self.tt_hits = 0
        self.cutoffs = 0
        
        # 킬러 무브와 히스토리
        self.killer_moves = {}
        self.history_table = {}
    
    def evaluate_position(self, board):
        """강력한 위치 평가"""
        if board.get_empty_count() == 0:
            b, w = board.count_stones()
            diff = (b - w) if self.color == BLACK else (w - b)
            if diff > 0:
                return 10000 + diff
            elif diff < 0:
                return -10000 + diff
            else:
                return 0
        
        score = 0
        empty_count = board.get_empty_count()
        
        # 1. 기동력 평가
        my_moves = len(board.get_valid_moves(self.color))
        opp_moves = len(board.get_valid_moves(opponent(self.color)))
        
        if my_moves + opp_moves > 0:
            mobility = 100 * (my_moves - opp_moves) / (my_moves + opp_moves + 1)
            score += mobility * (2.0 if empty_count > 20 else 1.0)
        
        # 특별 기동력 보너스
        if my_moves > 0 and opp_moves == 0:
            score += 500
        elif my_moves == 0 and opp_moves > 0:
            score -= 500
        
        # 2. 코너 제어
        corner_score = 0
        for corner_x, corner_y in CORNERS:
            if board.board[corner_x][corner_y] == self.color:
                corner_score += 300
            elif board.board[corner_x][corner_y] == opponent(self.color):
                corner_score -= 300
        
        score += corner_score
        
        # 3. 안정성
        stability_score = self.evaluate_stability(board)
        score += stability_score * (30 if empty_count < 30 else 15)
        
        # 4. 위치별 가중치
        position_score = self.evaluate_positions(board)
        score += position_score * (0.5 if empty_count < 20 else 1.0)
        
        # 5. 돌 개수 (후반에 중요)
        if empty_count < 20:
            b, w = board.count_stones()
            disc_diff = (b - w) if self.color == BLACK else (w - b)
            score += disc_diff * (5 if empty_count < 10 else 2)
        
        # 6. 프론티어 (경계 돌은 적을수록 좋음)
        if empty_count > 15:
            my_frontier = board.get_frontier_count(self.color)
            opp_frontier = board.get_frontier_count(opponent(self.color))
            score += (opp_frontier - my_frontier) * 8
        
        return int(score)
    
    def evaluate_stability(self, board):
        """안정성 평가"""
        my_stable = 0
        opp_stable = 0
        
        for i in range(8):
            for j in range(8):
                if board.board[i][j] == self.color:
                    if board.is_stable(i, j):
                        my_stable += 1
                elif board.board[i][j] == opponent(self.color):
                    if board.is_stable(i, j):
                        opp_stable += 1
        
        return my_stable - opp_stable
    
    def evaluate_positions(self, board):
        """위치별 평가"""
        position_weights = [
            [100, -20, 10, 5, 5, 10, -20, 100],
            [-20, -40, -5, -5, -5, -5, -40, -20],
            [10, -5, 3, 2, 2, 3, -5, 10],
            [5, -5, 2, 1, 1, 2, -5, 5],
            [5, -5, 2, 1, 1, 2, -5, 5],
            [10, -5, 3, 2, 2, 3, -5, 10],
            [-20, -40, -5, -5, -5, -5, -40, -20],
            [100, -20, 10, 5, 5, 10, -20, 100]
        ]
        
        score = 0
        for i in range(8):
            for j in range(8):
                if board.board[i][j] == self.color:
                    score += position_weights[i][j]
                elif board.board[i][j] == opponent(self.color):
                    score -= position_weights[i][j]
        
        return score
    
    def order_moves(self, board, moves, depth):
        """무브 정렬"""
        if not moves:
            return moves
        
        move_scores = []
        
        # TT에서 최고 수 찾기 (수정됨)
        board_hash = self.get_board_hash(board)
        tt_best_move = None
        if board_hash in self.tt:
            tt_entry = self.tt[board_hash]
            if isinstance(tt_entry, dict) and 'best_move' in tt_entry:
                tt_best_move = tt_entry['best_move']
        
        for move in moves:
            x, y = move
            score = 0
            
            # TT 최고 수
            if move == tt_best_move:
                score += 10000
            
            # 킬러 무브
            if depth in self.killer_moves and move in self.killer_moves[depth]:
                score += 1000
            
            # 히스토리 휴리스틱
            if move in self.history_table:
                score += self.history_table[move]
            
            # 위치별 점수
            if (x, y) in CORNERS:
                score += 500
            elif x == 0 or x == 7 or y == 0 or y == 7:
                score += 100
            elif (x, y) in [(1,1), (1,6), (6,1), (6,6)]:  # X-squares
                score -= 200
            
            # 뒤집는 돌 개수
            new_board = board.apply_move(x, y, self.color)
            if new_board.move_history:
                flipped_count = len(new_board.move_history[-1][3])
                score += flipped_count * 10
            
            move_scores.append((score, move))
        
        move_scores.sort(reverse=True)
        return [move for _, move in move_scores]
    
    def get_board_hash(self, board):
        """보드 해시"""
        board_str = ''.join(str(cell) for row in board.board for cell in row)
        return hash(board_str)
    
    def store_tt(self, board_hash, depth, score, flag, best_move):
        """TT 저장 - 안전한 딕셔너리 구조 (수정됨)"""
        if len(self.tt) >= self.max_tt_size:
            # 오래된 엔트리 일부 제거
            old_keys = list(self.tt.keys())[:len(self.tt)//4]
            for key in old_keys:
                del self.tt[key]
        
        self.tt[board_hash] = {
            'depth': depth,
            'score': score,
            'flag': flag,
            'best_move': best_move,
            'age': self.tt_age
        }
    
    def probe_tt(self, board_hash, depth, alpha, beta):
        """TT 조회 (수정됨)"""
        if board_hash not in self.tt:
            return None
        
        entry = self.tt[board_hash]
        if not isinstance(entry, dict):
            return None
        
        if entry.get('depth', 0) >= depth:
            self.tt_hits += 1
            flag = entry.get('flag', '')
            score = entry.get('score', 0)
            
            if flag == 'EXACT':
                return score
            elif flag == 'ALPHA' and score <= alpha:
                return alpha
            elif flag == 'BETA' and score >= beta:
                return beta
        
        return None
    
    def negamax(self, board, depth, alpha, beta, maximizing, end_time, passes=0):
        """네가맥스 알고리즘 (패스 처리 수정됨)"""
        self.nodes_searched += 1
        
        # 시간 체크
        if time.time() > end_time:
            return self.evaluate_position(board), None
        
        # TT 조회
        board_hash = self.get_board_hash(board)
        tt_score = self.probe_tt(board_hash, depth, alpha, beta)
        if tt_score is not None:
            return tt_score, None
        
        current_color = self.color if maximizing else opponent(self.color)
        moves = board.get_valid_moves(current_color)
        
        # 터미널 조건 및 패스 처리 (수정됨)
        if not moves:
            # 현재 플레이어가 둘 수 없음
            if passes >= 1:
                # 연속 2번 패스 = 게임 종료
                return self.evaluate_position(board), None
            else:
                # 패스하고 상대방 차례
                return -self.negamax(board, depth, -beta, -alpha, not maximizing, end_time, passes + 1)[0], None
        
        if depth == 0:
            return self.evaluate_position(board), None
        
        # 무브 정렬
        ordered_moves = self.order_moves(board, moves, depth)
        best_move = None
        original_alpha = alpha
        
        if maximizing:
            max_score = float('-inf')
            for move in ordered_moves:
                new_board = board.apply_move(*move, current_color)
                score, _ = self.negamax(new_board, depth - 1, alpha, beta, False, end_time, 0)
                
                if score > max_score:
                    max_score = score
                    best_move = move
                
                alpha = max(alpha, score)
                if beta <= alpha:
                    # Beta cutoff
                    self.cutoffs += 1
                    # 킬러 무브 업데이트
                    if depth not in self.killer_moves:
                        self.killer_moves[depth] = []
                    if move not in self.killer_moves[depth]:
                        self.killer_moves[depth].append(move)
                        if len(self.killer_moves[depth]) > 2:
                            self.killer_moves[depth].pop(0)
                    break
            
            # 히스토리 테이블 업데이트
            if best_move:
                if best_move not in self.history_table:
                    self.history_table[best_move] = 0
                self.history_table[best_move] += depth * depth
            
            # TT 저장
            flag = 'EXACT' if original_alpha < max_score < beta else ('BETA' if max_score >= beta else 'ALPHA')
            self.store_tt(board_hash, depth, max_score, flag, best_move)
            
            return max_score, best_move
        else:
            min_score = float('inf')
            for move in ordered_moves:
                new_board = board.apply_move(*move, current_color)
                score, _ = self.negamax(new_board, depth - 1, alpha, beta, True, end_time, 0)
                
                if score < min_score:
                    min_score = score
                    best_move = move
                
                beta = min(beta, score)
                if beta <= alpha:
                    self.cutoffs += 1
                    # 킬러 무브 업데이트
                    if depth not in self.killer_moves:
                        self.killer_moves[depth] = []
                    if move not in self.killer_moves[depth]:
                        self.killer_moves[depth].append(move)
                        if len(self.killer_moves[depth]) > 2:
                            self.killer_moves[depth].pop(0)
                    break
            
            if best_move:
                if best_move not in self.history_table:
                    self.history_table[best_move] = 0
                self.history_table[best_move] += depth * depth
            
            flag = 'EXACT' if alpha < min_score < original_alpha else ('ALPHA' if min_score <= alpha else 'BETA')
            self.store_tt(board_hash, depth, min_score, flag, best_move)
            
            return min_score, best_move
    
    def iterative_deepening(self, board):
        """반복 심화"""
        start_time = time.time()
        end_time = start_time + self.time_limit
        
        moves = board.get_valid_moves(self.color)
        if not moves:
            return None
        
        if len(moves) == 1:
            return moves[0]
        
        best_move = moves[0]
        
        for depth in range(1, self.max_depth + 1):
            try:
                score, move = self.negamax(board, depth, float('-inf'), float('inf'), 
                                         True, end_time, 0)
                
                if move and time.time() < end_time:
                    best_move = move
                
                # 시간 관리
                elapsed = time.time() - start_time
                if elapsed > self.time_limit * 0.8:
                    break
                    
            except Exception as e:
                print(f"Depth {depth} error: {e}")
                break
        
        return best_move
    
    def get_move(self, board):
        """최고의 수 반환"""
        self.nodes_searched = 0
        self.tt_hits = 0
        self.cutoffs = 0
        self.tt_age += 1
        
        return self.iterative_deepening(board)


class KoreanOthelloGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🏆 한국어 오셀로 AI - 최강 도전 🏆")
        self.root.geometry("900x950")
        
        self.cell_size = 55
        self.margin = 40
        self.board = Board()
        self.game_over = False
        self.ai_thinking = False
        
        # 게임 통계
        self.last_move = None
        self.move_count = 0
        self.game_stats = {
            'moves_count': 0,
            'human_time': 0,
            'ai_time': 0,
            'ai_nodes': 0,
            'game_start': time.time(),
            'difficulty_faced': 'unknown'
        }
        
        # Setup UI
        self.setup_ui()
        self.setup_game()
        self.update_display()
        
        if self.current_player != self.human_color:
            self.root.after(500, self.ai_move)

    def setup_ui(self):
        """한국어 UI 설정"""
        # 다크 테마 설정
        self.root.configure(bg="#0a0a0a")
        
        # 타이틀 프레임
        title_frame = tk.Frame(self.root, bg="#0a0a0a", height=80)
        title_frame.pack(fill=tk.X, pady=10)
        title_frame.pack_propagate(False)
        
        # 메인 타이틀
        main_title = tk.Label(title_frame, 
                             text="🏆 한국어 오셀로 AI 🏆", 
                             font=("맑은 고딕", 20, "bold"), 
                             fg="#0066cc", bg="#0a0a0a")
        main_title.pack()
        
        # 서브 타이틀
        sub_title = tk.Label(title_frame, 
                            text="🎯 최강 AI와의 대결 🎯", 
                            font=("맑은 고딕", 12, "bold"), 
                            fg="#0088ff", bg="#0a0a0a")
        sub_title.pack()
        
        # 컨트롤 프레임
        control_frame = tk.Frame(self.root, bg="#1a1a1a", relief=tk.RAISED, bd=3)
        control_frame.pack(pady=10, padx=20, fill=tk.X)
        
        # 버튼들
        button_frame = tk.Frame(control_frame, bg="#1a1a1a")
        button_frame.pack(pady=10)
        
        # 새 게임 버튼
        self.new_game_btn = tk.Button(button_frame, 
                                     text="🆕 새 게임", 
                                     command=self.new_game,
                                     font=("맑은 고딕", 12, "bold"), 
                                     bg="#4CAF50", fg="white",
                                     relief=tk.RAISED, bd=4,
                                     width=10, height=2)
        self.new_game_btn.pack(side=tk.LEFT, padx=5)
        
        # 항복 버튼
        self.surrender_btn = tk.Button(button_frame, 
                                      text="🏳️ 항복", 
                                      command=self.surrender,
                                      font=("맑은 고딕", 11, "bold"), 
                                      bg="#ff6666", fg="white",
                                      relief=tk.RAISED, bd=3,
                                      width=10, height=2)
        self.surrender_btn.pack(side=tk.LEFT, padx=5)
        
        # 도움말 버튼
        help_btn = tk.Button(button_frame, 
                           text="❓ 도움말", 
                           command=self.show_help,
                           font=("맑은 고딕", 11, "bold"), 
                           bg="#4444ff", fg="white",
                           relief=tk.RAISED, bd=3,
                           width=10, height=2)
        help_btn.pack(side=tk.LEFT, padx=5)
        
        # AI 설정 프레임
        ai_frame = tk.Frame(control_frame, bg="#1a1a1a")
        ai_frame.pack(pady=5)
        
        # AI 타입 선택
        tk.Label(ai_frame, text="🤖 AI 타입:", 
                font=("맑은 고딕", 11, "bold"), fg="#ffffff", bg="#1a1a1a").pack(side=tk.LEFT, padx=5)
        
        self.ai_type_var = tk.StringVar(value="최강")
        ai_combo = ttk.Combobox(ai_frame, textvariable=self.ai_type_var,
                               values=["기본", "고급", "최강"], 
                               width=12, font=("맑은 고딕", 10, "bold"),
                               state="readonly")
        ai_combo.pack(side=tk.LEFT, padx=5)
        
        # 난이도 설정
        tk.Label(ai_frame, text="⚡ 난이도:", 
                font=("맑은 고딕", 11, "bold"), fg="#ffffff", bg="#1a1a1a").pack(side=tk.LEFT, padx=5)
        
        self.difficulty_var = tk.StringVar(value="보통")
        diff_combo = ttk.Combobox(ai_frame, textvariable=self.difficulty_var,
                                 values=["쉬움", "보통", "어려움", "최고"], 
                                 width=10, font=("맑은 고딕", 10, "bold"),
                                 state="readonly")
        diff_combo.pack(side=tk.LEFT, padx=5)
        
        # 시간 제한
        tk.Label(ai_frame, text="⏱️ 시간:", 
                font=("맑은 고딕", 11, "bold"), fg="#ffffff", bg="#1a1a1a").pack(side=tk.LEFT, padx=5)
        
        self.time_var = tk.StringVar(value="5.0")
        time_combo = ttk.Combobox(ai_frame, textvariable=self.time_var,
                                 values=["2.0", "3.0", "5.0", "8.0", "10.0"], 
                                 width=8, font=("맑은 고딕", 10, "bold"),
                                 state="readonly")
        time_combo.pack(side=tk.LEFT, padx=5)
        
        # 게임 보드 캔버스
        canvas_size = self.cell_size * 8 + self.margin * 2
        self.canvas = tk.Canvas(self.root, 
                               width=canvas_size, height=canvas_size,
                               bg="#2E8B57", 
                               highlightthickness=3,
                               highlightbackground="#0066cc",
                               relief=tk.SUNKEN, bd=2)
        self.canvas.pack(pady=15)
        self.canvas.bind("<Button-1>", self.handle_click)
        self.canvas.bind("<Motion>", self.handle_hover)
        
        # 상태 표시 프레임
        status_frame = tk.Frame(self.root, bg="#0a0a0a")
        status_frame.pack(pady=10, fill=tk.X)
        
        # 현재 상태
        self.status_label = tk.Label(status_frame, 
                                   text="🎮 게임 시작", 
                                   font=("맑은 고딕", 16, "bold"), 
                                   fg="#00aa00", bg="#0a0a0a")
        self.status_label.pack()
        
        # 점수 표시
        self.score_label = tk.Label(status_frame, 
                                  text="⚫ 흑돌: 2  ⚪ 백돌: 2",
                                  font=("맑은 고딕", 14, "bold"), 
                                  fg="#ffffff", bg="#0a0a0a")
        self.score_label.pack()
        
        # AI 정보
        self.ai_info_label = tk.Label(status_frame, text="", 
                                     font=("맑은 고딕", 11), 
                                     fg="#66aaff", bg="#0a0a0a")
        self.ai_info_label.pack()
        
        # 프로그레스 바
        progress_frame = tk.Frame(status_frame, bg="#0a0a0a")
        progress_frame.pack(pady=5)
        
        tk.Label(progress_frame, text="🧠 AI 사고 중:",
                font=("맑은 고딕", 10, "bold"), fg="#0088ff", bg="#0a0a0a").pack()
        
        self.progress = ttk.Progressbar(progress_frame, 
                                       mode='indeterminate', 
                                       length=300)
        self.progress.pack(pady=3)
        
        # 게임 통계 프레임
        stats_frame = tk.Frame(self.root, bg="#1a1a1a", relief=tk.SUNKEN, bd=3)
        stats_frame.pack(pady=10, fill=tk.X, padx=20)
        
        tk.Label(stats_frame, text="📊 게임 통계", 
                font=("맑은 고딕", 12, "bold"), fg="#00ccff", bg="#1a1a1a").pack()
        
        self.stats_label = tk.Label(stats_frame, text="", 
                                   font=("Courier", 10), 
                                   fg="#ffffff", bg="#1a1a1a", 
                                   justify=tk.LEFT)
        self.stats_label.pack(pady=5)

    def setup_game(self):
        """게임 설정"""
        self.board = Board()
        self.game_over = False
        self.ai_thinking = False
        self.last_move = None
        self.move_count = 0
        
        # 난이도 매핑
        difficulty_mapping = {
            "쉬움": "easy", "보통": "medium", 
            "어려움": "hard", "최고": "ultimate"
        }
        
        korean_difficulty = self.difficulty_var.get()
        english_difficulty = difficulty_mapping.get(korean_difficulty, "medium")
        
        self.game_stats = {
            'moves_count': 0,
            'human_time': 0,
            'ai_time': 0,
            'ai_nodes': 0,
            'game_start': time.time(),
            'difficulty_faced': korean_difficulty
        }
        
        # 플레이어 색상 선택
        color_msg = ("🎯 색상 선택 🎯\n\n"
                    "흑돌(선공)로 플레이하시겠습니까?\n\n"
                    f"선택된 AI: {self.ai_type_var.get()}\n"
                    f"난이도: {korean_difficulty}\n"
                    f"시간 제한: {self.time_var.get()}초\n\n"
                    "준비되셨나요?")
        
        color_choice = messagebox.askyesno("🎮 게임 설정", color_msg)
        self.human_color = BLACK if color_choice else WHITE
        self.current_player = BLACK
        
        # AI 생성
        ai_type = self.ai_type_var.get()
        time_limit = float(self.time_var.get())
        ai_color = WHITE if self.human_color == BLACK else BLACK
        
        if ai_type == "최강":
            self.ai = UltimateAI(ai_color, english_difficulty, time_limit)
            ai_name = "🚀 최강 AI"
            ai_desc = "고급 알고리즘 적용"
        elif ai_type == "고급":
            self.ai = AdvancedAI(ai_color, english_difficulty, time_limit)
            ai_name = "🎯 고급 AI"
            ai_desc = "향상된 평가 함수"
        else:  # 기본
            # 기본 AI는 AdvancedAI를 쉬운 설정으로
            self.ai = AdvancedAI(ai_color, "easy", time_limit)
            ai_name = "🔰 기본 AI"
            ai_desc = "초보자용 AI"
        
        # AI 정보 업데이트
        ai_info = (f"🤖 상대: {ai_name}\n"
                  f"💪 난이도: {korean_difficulty}\n"
                  f"⏱️ 시간: {time_limit}초\n"
                  f"📋 설명: {ai_desc}")
        self.ai_info_label.config(text=ai_info)

    def new_game(self):
        """새 게임 시작"""
        if self.ai_thinking:
            if messagebox.askyesno("⚠️ AI 사고 중", 
                                 "AI가 현재 생각 중입니다.\n"
                                 "새 게임을 시작하시겠습니까?"):
                self.ai_thinking = False
            else:
                return
        
        self.setup_game()
        self.update_display()
        
        # AI가 먼저 시작하는 경우
        if self.current_player != self.human_color:
            self.root.after(500, self.ai_move)

    def surrender(self):
        """항복"""
        if not self.game_over:
            if messagebox.askyesno("🏳️ 항복", 
                                 "정말 항복하시겠습니까?\n\n"
                                 "AI가 승리하게 됩니다."):
                self.game_over = True
                ai_color_str = self.color_to_string(self.ai.color)
                self.status_label.config(
                    text=f"🏳️ {ai_color_str} 항복승!",
                    fg="#ff6666")
                
                messagebox.showinfo("🏳️ 항복", 
                                   f"{ai_color_str}이 항복승으로 이겼습니다!\n\n"
                                   "다시 도전해보세요!")

    def show_help(self):
        """도움말 표시"""
        help_text = """🎮 한국어 오셀로 AI 도움말 🎮

📋 게임 규칙:
• 8x8 보드에서 흑돌과 백돌로 대전
• 상대방의 돌을 양쪽에서 감싸면 뒤집기
• 더 많은 돌을 가진 쪽이 승리
• 둘 곳이 없으면 패스 (자동)

🤖 AI 타입:
• 기본: 초보자용 쉬운 AI
• 고급: 향상된 평가 함수 적용
• 최강: 고급 알고리즘 및 최적화

⚡ 난이도:
• 쉬움: 빠른 계산, 낮은 깊이
• 보통: 균형잡힌 성능  
• 어려움: 강력한 탐색
• 최고: 최대 성능

💡 팁:
• 코너를 차지하면 유리
• 경계선 근처는 주의
• 기동력(둘 수 있는 곳)이 중요
• 후반에는 돌 개수가 중요

🎯 즐거운 게임 되세요!"""
        
        messagebox.showinfo("❓ 도움말", help_text)

    def color_to_string(self, color):
        """색상을 한국어 문자열로 변환"""
        return "흑돌" if color == BLACK else "백돌"

    def update_display(self):
        """화면 업데이트"""
        self.canvas.delete("all")
        
        # 보드 그리기
        for i in range(9):
            # 세로선
            self.canvas.create_line(
                self.margin + i * self.cell_size, self.margin,
                self.margin + i * self.cell_size, self.margin + 8 * self.cell_size,
                fill="#2c5530", width=2)
            # 가로선
            self.canvas.create_line(
                self.margin, self.margin + i * self.cell_size,
                self.margin + 8 * self.cell_size, self.margin + i * self.cell_size,
                fill="#2c5530", width=2)
        
        # 돌들 그리기
        for i in range(8):
            for j in range(8):
                x = self.margin + j * self.cell_size + self.cell_size // 2
                y = self.margin + i * self.cell_size + self.cell_size // 2
                
                if self.board.board[i][j] != EMPTY:
                    color = "black" if self.board.board[i][j] == BLACK else "white"
                    outline_color = "white" if color == "black" else "black"
                    
                    # 돌 그리기
                    self.canvas.create_oval(
                        x - self.cell_size // 2 + 4, y - self.cell_size // 2 + 4,
                        x + self.cell_size // 2 - 4, y + self.cell_size // 2 - 4,
                        fill=color, outline=outline_color, width=2)
        
        # 마지막 수 표시
        if self.last_move:
            lx, ly = self.last_move
            x = self.margin + ly * self.cell_size + self.cell_size // 2
            y = self.margin + lx * self.cell_size + self.cell_size // 2
            
            self.canvas.create_oval(
                x - 8, y - 8, x + 8, y + 8,
                fill="red", outline="darkred", width=2)
        
        # 가능한 수 표시 (인간 차례일 때만)
        if not self.game_over and self.current_player == self.human_color and not self.ai_thinking:
            valid_moves = self.board.get_valid_moves(self.current_player)
            for move in valid_moves:
                mx, my = move
                x = self.margin + my * self.cell_size + self.cell_size // 2
                y = self.margin + mx * self.cell_size + self.cell_size // 2
                
                self.canvas.create_oval(
                    x - 12, y - 12, x + 12, y + 12,
                    fill="", outline="#FFD700", width=3)
        
        # 상태 업데이트
        self.update_status()
        self.update_stats()

    def update_status(self):
        """상태 텍스트 업데이트"""
        black_count, white_count = self.board.count_stones()
        
        # 점수 업데이트
        self.score_label.config(text=f"⚫ 흑돌: {black_count}  ⚪ 백돌: {white_count}")
        
        if self.game_over:
            return
        
        # 현재 플레이어 표시
        current_color_str = self.color_to_string(self.current_player)
        
        if self.ai_thinking:
            self.status_label.config(
                text=f"🧠 AI가 생각 중...",
                fg="#ff8800")
        elif self.current_player == self.human_color:
            self.status_label.config(
                text=f"🎯 {current_color_str} 차례 (당신)",
                fg="#00aa00")
        else:
            self.status_label.config(
                text=f"🤖 {current_color_str} 차례 (AI)",
                fg="#0088ff")

    def update_stats(self):
        """게임 통계 업데이트"""
        elapsed = time.time() - self.game_stats['game_start']
        
        stats_text = (f"🎮 총 수: {self.move_count}수\n"
                     f"⏰ 경과시간: {elapsed:.1f}초\n"
                     f"🧠 AI 노드: {self.game_stats['ai_nodes']:,}개\n"
                     f"💪 난이도: {self.game_stats['difficulty_faced']}")
        
        if hasattr(self.ai, 'nodes_searched'):
            stats_text += f"\n🔍 현재 탐색: {self.ai.nodes_searched:,}개"
        
        self.stats_label.config(text=stats_text)

    def handle_click(self, event):
        """마우스 클릭 처리"""
        if self.game_over or self.ai_thinking or self.current_player != self.human_color:
            return
        
        # 클릭 위치 계산
        col = (event.x - self.margin) // self.cell_size
        row = (event.y - self.margin) // self.cell_size
        
        if 0 <= row < 8 and 0 <= col < 8:
            if self.board.is_valid_move(row, col, self.current_player):
                self.make_move(row, col)

    def handle_hover(self, event):
        """마우스 호버 처리"""
        if self.game_over or self.ai_thinking or self.current_player != self.human_color:
            return
        
        # 호버 효과는 간단하게 구현 (선택적)
        pass

    def make_move(self, row, col):
        """수 두기 (패스 처리 수정됨)"""
        if self.board.is_valid_move(row, col, self.current_player):
            move_start = time.time()
            
            # 수 두기
            self.board = self.board.apply_move(row, col, self.current_player)
            self.last_move = (row, col)
            self.move_count += 1
            
            # 시간 기록
            if self.current_player == self.human_color:
                self.game_stats['human_time'] += time.time() - move_start
            
            # 다음 플레이어 결정 (수정됨)
            self.switch_player()

    def switch_player(self):
        """플레이어 교체 (패스 처리 포함)"""
        opponent_color = opponent(self.current_player)
        
        # 상대방이 둘 수 있는지 확인
        if self.board.get_valid_moves(opponent_color):
            # 상대방 차례
            self.current_player = opponent_color
        else:
            # 상대방 패스 - 내가 계속 둘 수 있는지 확인
            if self.board.get_valid_moves(self.current_player):
                # 내가 계속 둠 (상대방 패스)
                pass_color_str = self.color_to_string(opponent_color)
                print(f"⏭️ {pass_color_str} 패스!")
                # current_player는 그대로 유지
            else:
                # 둘 다 둘 수 없음 - 게임 종료
                self.end_game()
                return
        
        self.update_display()
        
        # AI 차례이면 AI 수 두기
        if not self.game_over and self.current_player != self.human_color:
            self.root.after(300, self.ai_move)

    def ai_move(self):
        """AI 수 두기"""
        if self.game_over or self.current_player == self.human_color:
            return
        
        moves = self.board.get_valid_moves(self.current_player)
        if not moves:
            # AI 패스
            self.switch_player()
            return
        
        self.ai_thinking = True
        self.progress.start()
        self.update_status()
        
        def ai_thread():
            try:
                start_time = time.time()
                move = self.ai.get_move(self.board)
                ai_time = time.time() - start_time
                
                self.game_stats['ai_time'] += ai_time
                if hasattr(self.ai, 'nodes_searched'):
                    self.game_stats['ai_nodes'] += self.ai.nodes_searched
                
                if move:
                    self.root.after(0, lambda: self.execute_ai_move(move))
                else:
                    self.root.after(0, self.ai_pass)
                    
            except Exception as e:
                print(f"AI 오류: {e}")
                self.root.after(0, self.ai_pass)
        
        threading.Thread(target=ai_thread, daemon=True).start()

    def execute_ai_move(self, move):
        """AI 수 실행"""
        self.ai_thinking = False
        self.progress.stop()
        
        row, col = move
        
        # AI 수 두기
        self.board = self.board.apply_move(row, col, self.current_player)
        self.last_move = (row, col)
        self.move_count += 1
        
        print(f"🤖 AI가 {chr(col + ord('a'))}{row + 1}에 착수")
        
        # 플레이어 교체
        self.switch_player()

    def ai_pass(self):
        """AI 패스 처리"""
        self.ai_thinking = False
        self.progress.stop()
        
        print("🤖 AI 패스!")
        self.switch_player()

    def end_game(self):
        """게임 종료"""
        self.game_over = True
        self.progress.stop()
        
        black_count, white_count = self.board.count_stones()
        
        if black_count > white_count:
            winner = "흑돌"
            winner_color = BLACK
        elif white_count > black_count:
            winner = "백돌"
            winner_color = WHITE
        else:
            winner = "무승부"
            winner_color = None
        
        # 상태 업데이트
        if winner == "무승부":
            self.status_label.config(
                text="🤝 무승부!",
                fg="#ffaa00")
            result_msg = "🤝 무승부입니다!"
        else:
            if winner_color == self.human_color:
                self.status_label.config(
                    text=f"🎉 {winner} 승리! (당신)",
                    fg="#00ff00")
                result_msg = f"🎉 축하합니다!\n{winner}이 승리했습니다!"
            else:
                self.status_label.config(
                    text=f"🤖 {winner} 승리! (AI)",
                    fg="#ff4444")
                result_msg = f"🤖 AI 승리!\n{winner}이 이겼습니다!"
        
        # 최종 통계
        total_time = time.time() - self.game_stats['game_start']
        final_stats = (f"{result_msg}\n\n"
                      f"📊 게임 통계:\n"
                      f"• 총 수: {self.move_count}수\n"
                      f"• 총 시간: {total_time:.1f}초\n"
                      f"• 최종 점수: 흑 {black_count} : 백 {white_count}\n"
                      f"• AI 난이도: {self.game_stats['difficulty_faced']}\n"
                      f"• AI 탐색 노드: {self.game_stats['ai_nodes']:,}개\n\n"
                      f"다시 게임하시겠습니까?")
        
        self.update_display()
        
        # 새 게임 제안
        if messagebox.askyesno("🎮 게임 종료", final_stats):
            self.new_game()


def main():
    """메인 함수"""
    root = tk.Tk()
    
    # 창 아이콘 설정 (선택적)
    try:
        root.wm_iconbitmap('othello.ico')
    except:
        pass
    
    # 창 크기 고정
    root.resizable(False, False)
    
    # 게임 시작
    game = KoreanOthelloGUI(root)
    
    # 프로그램 시작 메시지
    welcome_msg = """🏆 한국어 오셀로 AI에 오신 것을 환영합니다! 🏆

🎮 특징:
• 3가지 AI 타입 (기본/고급/최강)
• 4가지 난이도 설정
• 실시간 게임 통계
• 한국어 완전 지원
• 다크 테마 UI
• 자동 패스 처리

🎯 지금 바로 AI와 대결해보세요!
최고 난이도에서 이길 수 있나요?"""
    
    messagebox.showinfo("🎮 환영합니다!", welcome_msg)
    
    root.mainloop()


if __name__ == "__main__":
    main()