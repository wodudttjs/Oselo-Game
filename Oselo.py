import tkinter as tk
from tkinter import messagebox, simpledialog
import copy
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import random

EMPTY, BLACK, WHITE = '.', 'B', 'W'

# 기본 평가 가중치 (중앙 약함, 모서리 강함)
BASE_WEIGHTS = [
    [100, -20, 10,  5,  5, 10, -20, 100],
    [-20, -50, -2, -2, -2, -2, -50, -20],
    [10,   -2, -1, -1, -1, -1,  -2,  10],
    [5,    -2, -1, -1, -1, -1,  -2,   5],
    [5,    -2, -1, -1, -1, -1,  -2,   5],
    [10,   -2, -1, -1, -1, -1,  -2,  10],
    [-20, -50, -2, -2, -2, -2, -50, -20],
    [100, -20, 10,  5,  5, 10, -20, 100],
]

# X-squares (모서리 바로 대각선 방향) 위치
X_SQUARES = [(1, 1), (1, 6), (6, 1), (6, 6)]

# C-squares (모서리 인접) 위치
C_SQUARES = [(0, 1), (1, 0), (0, 6), (6, 0), (7, 1), (6, 7), (7, 6), (1, 7)]

# 모서리 위치
CORNERS = [(0, 0), (0, 7), (7, 0), (7, 7)]

# 기본 오프닝 패턴 (x, y 좌표)
OPENING_BOOK = {
    # 특정 초기 상태에 대한 응답 패턴
    "empty_board": [(3, 3), (2, 2), (2, 3), (2, 4), (3, 5)],  # 표준 오프닝
    "standard_opening": [(2, 4), (2, 3), (3, 2), (4, 2), (5, 3)]  # 대응 오프닝
}

def opponent(color):
    return BLACK if color == WHITE else WHITE

def evaluate_move(move, board, color, depth):
    ai = AlphaBetaAI(color, depth)
    new_board = board.apply_move(*move, color)
    score = ai.alphabeta(new_board, depth - 1, float('-inf'), float('inf'), False)[0]
    return score, move

class Board:
    def __init__(self):
        self.size = 8
        self.board = [[EMPTY] * self.size for _ in range(self.size)]
        self.board[3][3] = WHITE
        self.board[3][4] = BLACK
        self.board[4][3] = BLACK
        self.board[4][4] = WHITE
        self.move_history = []  # 이동 기록 추가

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
        for dx, dy in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
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
        flipped = []  # 뒤집힌 돌 추적
        
        for dx, dy in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
            nx, ny = x + dx, y + dy
            to_flip = []
            while new_board.in_bounds(nx, ny) and new_board.board[nx][ny] == opponent(color):
                to_flip.append((nx, ny))
                nx += dx
                ny += dy
            if new_board.in_bounds(nx, ny) and new_board.board[nx][ny] == color:
                for fx, fy in to_flip:
                    new_board.board[fx][fy] = color
                    flipped.extend(to_flip)
            
        # 움직임 기록 추가
        new_board.move_history = self.move_history + [(x, y, color, flipped)]
        return new_board

    def count_stones(self):
        b = sum(row.count(BLACK) for row in self.board)
        w = sum(row.count(WHITE) for row in self.board)
        return b, w
        
    def get_empty_count(self):
        """빈 칸 수 반환"""
        return sum(row.count(EMPTY) for row in self.board)
        
    def is_stable(self, x, y):
        """해당 위치의 돌이 안정적인지 확인 (뒤집힐 수 없는 돌)"""
        if self.board[x][y] == EMPTY:
            return False
            
        color = self.board[x][y]
        
        # 모서리는 항상 안정적
        if (x, y) in CORNERS:
            return True
            
        # 수평, 수직, 대각선 방향으로 안정성 검사
        directions = [
            [(0, 1), (0, -1)],  # 수평
            [(1, 0), (-1, 0)],  # 수직
            [(1, 1), (-1, -1)],  # 대각선 1
            [(1, -1), (-1, 1)]   # 대각선 2
        ]
        
        for dir_pair in directions:
            stable_in_direction = False
            
            for dx, dy in dir_pair:
                nx, ny = x, y
                while True:
                    nx += dx
                    ny += dy
                    if not self.in_bounds(nx, ny):
                        # 경계에 도달 = 안정적
                        stable_in_direction = True
                        break
                    if self.board[nx][ny] != color:
                        break
                    if (nx, ny) in CORNERS:
                        # 모서리 연결 = 안정적
                        stable_in_direction = True
                        break
                
                if stable_in_direction:
                    break
                    
            if not stable_in_direction:
                return False
                
        return True

class AlphaBetaAI:
    def __init__(self, color, depth=3):
        self.color = color
        self.default_depth = depth
        self.max_workers = multiprocessing.cpu_count()
        self.transposition_table = {}  # 트랜스포지션 테이블 추가
        
    def board_to_key(self, board):
        """보드 상태를 해시 가능한 키로 변환"""
        return tuple(tuple(row) for row in board.board)

    def evaluate(self, board_obj):
        """게임 상태 평가 함수"""
        board = board_obj.board
        
        # 게임 단계 결정 (초반, 중반, 후반)
        empty_count = board_obj.get_empty_count()
        total_cells = board_obj.size * board_obj.size
        
        # 게임 단계별 가중치 조정
        if empty_count > 0.7 * total_cells:  # 초반
            weights = {
                'position': 0.6,  # 위치 중요
                'mobility': 1.0,  # 이동성 매우 중요
                'stability': 0.3,  # 안정성 낮음
                'corner': 1.5,    # 모서리 매우 중요
                'parity': 0.1     # 패리티 무시
            }
        elif empty_count > 0.3 * total_cells:  # 중반
            weights = {
                'position': 0.8,   # 위치 중요
                'mobility': 0.7,   # 이동성 중요
                'stability': 0.6,  # 안정성 증가
                'corner': 1.0,     # 모서리 중요
                'parity': 0.3      # 패리티 약간 고려
            }
        else:  # 후반
            weights = {
                'position': 0.5,   # 위치 덜 중요
                'mobility': 0.3,   # 이동성 덜 중요
                'stability': 1.0,  # 안정성 매우 중요
                'corner': 0.7,     # 모서리 여전히 중요
                'parity': 1.0      # 패리티 매우 중요
            }
        
        def positional_score():
            """위치 기반 점수"""
            score = 0
            for i in range(8):
                for j in range(8):
                    if board[i][j] == self.color:
                        score += BASE_WEIGHTS[i][j]
                    elif board[i][j] == opponent(self.color):
                        score -= BASE_WEIGHTS[i][j]
                        
            # X-square 페널티 추가 (모서리가 비어있는 경우)
            for corner_x, corner_y in CORNERS:
                if board[corner_x][corner_y] == EMPTY:
                    # 해당 모서리의 X-square 패널티
                    for x, y in X_SQUARES:
                        if (abs(corner_x - x) <= 1 and abs(corner_y - y) <= 1):
                            if board[x][y] == self.color:
                                score -= 25  # X-square에 돌을 놓는 것은 큰 패널티
                            elif board[x][y] == opponent(self.color):
                                score += 25  # 상대방의 X-square는 이득
            
            # C-square 점수 조정
            for c_x, c_y in C_SQUARES:
                corner_x, corner_y = c_x - (c_x % 7), c_y - (c_y % 7)  # 가장 가까운 모서리
                
                if board[corner_x][corner_y] == EMPTY:
                    if board[c_x][c_y] == self.color:
                        score -= 15  # 빈 모서리 옆의 C-square에 돌을 놓는 것은 불리
                elif board[corner_x][corner_y] == self.color:
                    if board[c_x][c_y] == self.color:
                        score += 10  # 내 모서리 옆의 C-square는 유리함
            
            return score

        def mobility_score():
            """이동성 점수 (유효한 이동의 수)"""
            my_moves = len(board_obj.get_valid_moves(self.color))
            opp_moves = len(board_obj.get_valid_moves(opponent(self.color)))
            
            # 이동성 비율 계산
            if my_moves + opp_moves == 0:
                return 0
            
            # 가중치가 적용된 이동성
            return 100 * (my_moves - opp_moves) / (my_moves + opp_moves)

        def corner_score():
            """모서리 점유 점수"""
            my = sum(1 for x, y in CORNERS if board[x][y] == self.color)
            opp = sum(1 for x, y in CORNERS if board[x][y] == opponent(self.color))
            return 25 * (my - opp)
            
        def stability_score():
            """안정성 점수 (뒤집힐 수 없는 돌의 수)"""
            my_stable = 0
            opp_stable = 0
            
            for i in range(8):
                for j in range(8):
                    if board[i][j] != EMPTY and board_obj.is_stable(i, j):
                        if board[i][j] == self.color:
                            my_stable += 1
                        else:
                            opp_stable += 1
            
            return 15 * (my_stable - opp_stable)
            
        def parity_score():
            """패리티 점수 (홀수/짝수 전략)"""
            empty_count = board_obj.get_empty_count()
            # 후반부에 패리티가 중요 - 마지막 수를 둘 수 있는지
            if empty_count % 2 == 0:  # 빈칸이 짝수면
                return 10  # 선공(흑)에게 유리
            else:
                return -10  # 후공(백)에게 유리
        
        # 종합 점수 계산 (각 요소에 가중치 적용)
        final_score = (
            weights['position'] * positional_score() +
            weights['mobility'] * mobility_score() +
            weights['corner'] * corner_score() +
            weights['stability'] * stability_score()
        )
        
        # 후반부에는 패리티 적용
        if empty_count < 12:  # 매우 후반부에는 패리티가 중요
            final_score += weights['parity'] * parity_score()
            
        return final_score

    def get_dynamic_depth(self, board):
        """게임 진행 상황에 따른 동적 탐색 깊이 결정"""
        total_stones = sum(row.count(BLACK) + row.count(WHITE) for row in board.board)
        empty_count = 64 - total_stones
        
        # 초반, 중반, 후반에 따른 깊이 조절
        if empty_count > 50:  # 초반
            return 3
        elif empty_count > 35:  # 중반 초기
            return 4
        elif empty_count > 20:  # 중반 후기
            return 5
        elif empty_count > 10:  # 후반 초기
            return 6
        else:  # 후반 (10개 이하 빈칸)
            return 7  # 더 깊은 탐색
            
    def check_endgame(self, board):
        """엔드게임 상황인지 확인 (정확한 탐색 가능)"""
        empty_count = board.get_empty_count()
        return empty_count <= 8  # 빈칸이 8개 이하면 엔드게임으로 간주

    def alphabeta(self, board, depth, alpha, beta, maximizing):
        """알파-베타 가지치기를 이용한 미니맥스 알고리즘"""
        current_color = self.color if maximizing else opponent(self.color)
        moves = board.get_valid_moves(current_color)
        
        # 트랜스포지션 테이블 조회
        board_key = self.board_to_key(board)
        if board_key in self.transposition_table and self.transposition_table[board_key]['depth'] >= depth:
            entry = self.transposition_table[board_key]
            if entry['type'] == 'exact':
                return entry['value'], entry['move']
            elif entry['type'] == 'lowerbound' and entry['value'] >= beta:
                return entry['value'], entry['move']
            elif entry['type'] == 'upperbound' and entry['value'] <= alpha:
                return entry['value'], entry['move']

        if depth == 0 or not moves:
            if not moves:
                # 현재 플레이어가 움직일 수 없을 때
                opponent_moves = board.get_valid_moves(opponent(current_color))
                if not opponent_moves:  # 게임 종료
                    b, w = board.count_stones()
                    score = float('inf') if (b > w and self.color == BLACK) or (w > b and self.color == WHITE) else float('-inf')
                    return score, None
                
                # 상대도 움직일 수 없으면 게임 종료, 아니면 패스
                next_board = copy.deepcopy(board)
                return self.alphabeta(next_board, depth, alpha, beta, not maximizing)
                
            return self.evaluate(board), None

        # 엔드게임 상황에서는 정확한 계산
        if self.check_endgame(board):
            # 모든 경우를 탐색 - 더 깊게 탐색
            depth = min(depth + 2, 10)  # 최대 깊이 제한
            
        # 움직임 정렬 - 휴리스틱을 사용해 유망한 움직임 먼저 탐색
        sorted_moves = self.sort_moves(board, moves, current_color)
        
        best_move = None
        entry_type = 'upperbound' if maximizing else 'lowerbound'
        
        if maximizing:
            value = float('-inf')
            for move in sorted_moves:
                new_board = board.apply_move(*move, current_color)
                eval_score, _ = self.alphabeta(new_board, depth - 1, alpha, beta, False)
                if eval_score > value:
                    value = eval_score
                    best_move = move
                alpha = max(alpha, value)
                if alpha >= beta:
                    entry_type = 'lowerbound'
                    break
        else:
            value = float('inf')
            for move in sorted_moves:
                new_board = board.apply_move(*move, current_color)
                eval_score, _ = self.alphabeta(new_board, depth - 1, alpha, beta, True)
                if eval_score < value:
                    value = eval_score
                    best_move = move
                beta = min(beta, value)
                if alpha >= beta:
                    entry_type = 'upperbound'
                    break
                    
        # 트랜스포지션 테이블에 결과 저장
        if alpha < value < beta:
            entry_type = 'exact'
            
        self.transposition_table[board_key] = {
            'value': value,
            'move': best_move,
            'depth': depth,
            'type': entry_type
        }
        
        return value, best_move
        
    def sort_moves(self, board, moves, color):
        """휴리스틱을 사용하여 유망한 움직임을 먼저 탐색하도록 정렬"""
        move_scores = []
        
        for move in moves:
            x, y = move
            score = 0
            
            # 모서리는 가장 높은 우선순위
            if move in CORNERS:
                score += 100
            # X-squares는 낮은 우선순위
            elif move in X_SQUARES:
                score -= 50
            # C-squares도 낮은 우선순위지만 X보다는 높음
            elif move in C_SQUARES:
                score -= 25
            # 변(edge)은 중간 우선순위
            elif x == 0 or x == 7 or y == 0 or y == 7:
                score += 50
                
            # 해당 움직임으로 뒤집히는 돌의 수도 고려
            new_board = board.apply_move(x, y, color)
            flipped = len(new_board.move_history[-1][3]) if new_board.move_history else 0
            score += flipped * 2
            
            move_scores.append((score, move))
            
        # 점수 기준으로 내림차순 정렬
        move_scores.sort(reverse=True)
        return [move for _, move in move_scores]

    def get_move(self, board):
        """최적의 움직임 결정"""
        # 게임 초반 오프닝북 사용
        if board.get_empty_count() > 55:  # 거의 초반
            return self.get_opening_move(board)
            
        # 엔드게임 전략
        if self.check_endgame(board):
            return self.get_endgame_move(board)
        
        # 일반적인 움직임 결정
        dynamic_depth = self.get_dynamic_depth(board)
        moves = board.get_valid_moves(self.color)
        
        if not moves:
            return None
            
        # 확률적 탐색으로 다양성 추가
        if random.random() < 0.05:  # 5% 확률로 랜덤 움직임
            return random.choice(moves)

        evaluate_fn = partial(evaluate_move, board=board, color=self.color, depth=dynamic_depth)
        
        # 병렬 처리로 성능 향상
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(evaluate_fn, moves))
            
        results.sort(reverse=True)
        return results[0][1] if results else None
        
    def get_opening_move(self, board):
        """오프닝 전략"""
        moves = board.get_valid_moves(self.color)
        if not moves:
            return None
            
        # 표준 오프닝 패턴 적용
        for pattern_name, pattern_moves in OPENING_BOOK.items():
            for move in pattern_moves:
                if move in moves:
                    return move
                    
        # 휴리스틱 기반 초기 전략
        # 1. 모서리 선호
        for move in moves:
            if move in CORNERS:
                return move
                
        # 2. X-square와 C-square 회피
        safe_moves = [m for m in moves if m not in X_SQUARES and m not in C_SQUARES]
        if safe_moves:
            return random.choice(safe_moves)
            
        # 3. 기본값으로 중앙부 선호
        center_preference = []
        for move in moves:
            x, y = move
            # 중앙에 가까울수록 높은 점수
            distance_from_center = abs(x - 3.5) + abs(y - 3.5)
            center_preference.append((distance_from_center, move))
            
        center_preference.sort()  # 중앙에 가까운 것부터 정렬
        return center_preference[0][1]
        
    def get_endgame_move(self, board):
        """엔드게임 전략"""
        moves = board.get_valid_moves(self.color)
        if not moves:
            return None
            
        # 승리를 확정할 수 있는 움직임 확인
        empty_count = board.get_empty_count()
        
        if empty_count <= 8:  # 완전한 엔드게임
            # 최대 깊이로 탐색 (마지막까지)
            best_score = float('-inf')
            best_move = None
            
            for move in moves:
                new_board = board.apply_move(*move, self.color)
                # 완전 탐색
                score, _ = self.alphabeta(new_board, min(empty_count, 10), float('-inf'), float('inf'), False)
                
                if score > best_score:
                    best_score = score
                    best_move = move
                    
            return best_move
            
        # 일반적인 평가
        return self.get_move(board)

# 이하 GUI 클래스와 실행 부분 동일
class OthelloGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Othello (Tkinter)")
        self.cell_size = 60
        self.board = Board()

        # 사용자에게 흑/백 선택받기
        player_choice = simpledialog.askstring("Player Color", "Do you want to play first? (yes/no)").lower()
        if player_choice in ['yes', 'y']:
            self.current_player = BLACK
            self.ai = AlphaBetaAI(WHITE)
        else:
            self.current_player = WHITE
            self.ai = AlphaBetaAI(BLACK)

        self.canvas = tk.Canvas(self.root, width=self.cell_size*8, height=self.cell_size*8, bg="dark green")
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.handle_click)

        self.status_label = tk.Label(self.root, text="Game Start", font=("Arial", 14))
        self.status_label.pack()
        self.update_gui()

        if self.current_player == WHITE:
            self.root.after(500, self.ai_move)

    def handle_click(self, event):
        x = event.x // self.cell_size
        y = event.y // self.cell_size
        if (y, x) in self.board.get_valid_moves(self.current_player):
            self.board = self.board.apply_move(y, x, self.current_player)
            self.update_gui()
            if not self.board.get_valid_moves(self.ai.color) and not self.board.get_valid_moves(self.current_player):
                self.check_game_end()
                return
            if self.board.get_valid_moves(self.ai.color):
                self.root.after(500, self.ai_move)
            else:
                messagebox.showinfo("Turn Skipped", "AI has no valid move. Your turn again.")

    def ai_move(self):
        if self.board.get_valid_moves(self.ai.color):
            move = self.ai.get_move(self.board)
            if move:
                self.board = self.board.apply_move(*move, self.ai.color)
                self.update_gui()

        if not self.board.get_valid_moves(self.ai.color) and not self.board.get_valid_moves(self.current_player):
            self.check_game_end()
            return

        if not self.board.get_valid_moves(self.current_player):
            messagebox.showinfo("Turn Skipped", "You have no valid move. AI plays again.")
            self.root.after(500, self.ai_move)

    def check_game_end(self):
        b, w = self.board.count_stones()
        msg = f"Game Over!\nBlack: {b} | White: {w}\n"
        msg += "You Win!" if b > w else "AI Wins!" if w > b else "Draw!"
        messagebox.showinfo("Result", msg)
        self.root.destroy()
    def update_gui(self):
        self.canvas.delete("all")
        for i in range(8):
            for j in range(8):
                x0, y0 = j * self.cell_size, i * self.cell_size
                x1, y1 = x0 + self.cell_size, y0 + self.cell_size
                self.canvas.create_rectangle(x0, y0, x1, y1, outline="black", fill="dark green")

                stone = self.board.board[i][j]
                if stone == BLACK:
                    self.canvas.create_oval(x0+5, y0+5, x1-5, y1-5, fill="black")
                elif stone == WHITE:
                    self.canvas.create_oval(x0+5, y0+5, x1-5, y1-5, fill="white")

        b, w = self.board.count_stones()
        self.status_label.config(text=f"Black: {b} | White: {w}")


# 실행
if __name__ == "__main__":
    root = tk.Tk()
    app = OthelloGUI(root)
    root.mainloop()
