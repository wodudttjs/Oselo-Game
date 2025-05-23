import tkinter as tk
from tkinter import messagebox, simpledialog
import copy
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import random
import time
from collections import defaultdict

EMPTY, BLACK, WHITE = '.', 'B', 'W'

# 개선된 평가 가중치 - 게임 단계별로 다른 가중치 사용
EARLY_WEIGHTS = [
    [100, -20, 10,  5,  5, 10, -20, 100],
    [-20, -50, -2, -2, -2, -2, -50, -20],
    [10,   -2, -1, -1, -1, -1,  -2,  10],
    [5,    -2, -1, -1, -1, -1,  -2,   5],
    [5,    -2, -1, -1, -1, -1,  -2,   5],
    [10,   -2, -1, -1, -1, -1,  -2,  10],
    [-20, -50, -2, -2, -2, -2, -50, -20],
    [100, -20, 10,  5,  5, 10, -20, 100],
]

MID_WEIGHTS = [
    [100, -10, 15,  8,  8, 15, -10, 100],
    [-10, -30, -1, -1, -1, -1, -30, -10],
    [15,   -1,  2,  1,  1,  2,  -1,  15],
    [8,    -1,  1,  0,  0,  1,  -1,   8],
    [8,    -1,  1,  0,  0,  1,  -1,   8],
    [15,   -1,  2,  1,  1,  2,  -1,  15],
    [-10, -30, -1, -1, -1, -1, -30, -10],
    [100, -10, 15,  8,  8, 15, -10, 100],
]

LATE_WEIGHTS = [
    [100,  10, 20, 15, 15, 20,  10, 100],
    [10,   -5,  5,  5,  5,  5,  -5,  10],
    [20,    5, 10,  8,  8, 10,   5,  20],
    [15,    5,  8,  5,  5,  8,   5,  15],
    [15,    5,  8,  5,  5,  8,   5,  15],
    [20,    5, 10,  8,  8, 10,   5,  20],
    [10,   -5,  5,  5,  5,  5,  -5,  10],
    [100,  10, 20, 15, 15, 20,  10, 100],
]

# X-squares, C-squares, 모서리 정의
X_SQUARES = [(1, 1), (1, 6), (6, 1), (6, 6)]
C_SQUARES = [(0, 1), (1, 0), (0, 6), (6, 0), (7, 1), (6, 7), (7, 6), (1, 7)]
CORNERS = [(0, 0), (0, 7), (7, 0), (7, 7)]
EDGES = [(i, 0) for i in range(8)] + [(i, 7) for i in range(8)] + [(0, i) for i in range(8)] + [(7, i) for i in range(8)]

# 패턴 기반 평가를 위한 패턴 정의
PATTERNS = {
    'corner_wedge': [
        [(0, 0), (0, 1), (1, 0)],  # 왼쪽 위 모서리 웨지
        [(0, 7), (0, 6), (1, 7)],  # 오른쪽 위 모서리 웨지
        [(7, 0), (7, 1), (6, 0)],  # 왼쪽 아래 모서리 웨지
        [(7, 7), (7, 6), (6, 7)]   # 오른쪽 아래 모서리 웨지
    ]
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
        flipped = []
        
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
            
        new_board.move_history = self.move_history + [(x, y, color, flipped)]
        return new_board

    def count_stones(self):
        b = sum(row.count(BLACK) for row in self.board)
        w = sum(row.count(WHITE) for row in self.board)
        return b, w
        
    def get_empty_count(self):
        return sum(row.count(EMPTY) for row in self.board)
        
    def is_stable(self, x, y):
        if self.board[x][y] == EMPTY:
            return False
            
        color = self.board[x][y]
        
        if (x, y) in CORNERS:
            return True
            
        directions = [
            [(0, 1), (0, -1)],
            [(1, 0), (-1, 0)],
            [(1, 1), (-1, -1)],
            [(1, -1), (-1, 1)]
        ]
        
        for dir_pair in directions:
            stable_in_direction = False
            
            for dx, dy in dir_pair:
                nx, ny = x, y
                while True:
                    nx += dx
                    ny += dy
                    if not self.in_bounds(nx, ny):
                        stable_in_direction = True
                        break
                    if self.board[nx][ny] != color:
                        break
                    if (nx, ny) in CORNERS:
                        stable_in_direction = True
                        break
                
                if stable_in_direction:
                    break
                    
            if not stable_in_direction:
                return False
                
        return True

    def get_frontier_count(self, color):
        """프론티어 디스크 개수 (인접한 빈 칸이 있는 디스크)"""
        frontier = 0
        for i in range(8):
            for j in range(8):
                if self.board[i][j] == color:
                    for dx, dy in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
                        ni, nj = i + dx, j + dy
                        if self.in_bounds(ni, nj) and self.board[ni][nj] == EMPTY:
                            frontier += 1
                            break
        return frontier

class EnhancedAlphaBetaAI:
    def __init__(self, color, time_limit=5.0):
        self.color = color
        self.time_limit = time_limit
        self.transposition_table = {}
        self.killer_moves = defaultdict(list)  # Killer moves 휴리스틱
        self.history_table = defaultdict(int)  # History heuristic
        self.pv_table = {}  # Principal Variation table
        self.max_workers = min(4, multiprocessing.cpu_count())
        
    def clear_search_tables(self):
        """탐색 테이블 초기화"""
        self.killer_moves.clear()
        self.history_table.clear()
        self.pv_table.clear()

    def board_to_key(self, board):
        return tuple(tuple(row) for row in board.board)

    def get_weights_for_stage(self, board):
        """게임 단계에 맞는 가중치 반환"""
        empty_count = board.get_empty_count()
        if empty_count > 45:
            return EARLY_WEIGHTS
        elif empty_count > 20:
            return MID_WEIGHTS
        else:
            return LATE_WEIGHTS

    def evaluate_patterns(self, board):
        """패턴 기반 평가"""
        score = 0
        
        # 모서리 웨지 패턴 평가
        for wedge in PATTERNS['corner_wedge']:
            my_count = sum(1 for pos in wedge if board.board[pos[0]][pos[1]] == self.color)
            opp_count = sum(1 for pos in wedge if board.board[pos[0]][pos[1]] == opponent(self.color))
            
            if my_count == 3:  # 완전한 웨지
                score += 50
            elif my_count == 2 and opp_count == 0:  # 부분 웨지
                score += 20
            elif opp_count == 3:
                score -= 50
            elif opp_count == 2 and my_count == 0:
                score -= 20
                
        return score

    def evaluate(self, board):
        """개선된 평가 함수"""
        empty_count = board.get_empty_count()
        total_cells = 64
        
        # 게임 단계별 가중치
        if empty_count > 0.7 * total_cells:  # 초반
            weights = {
                'position': 0.4, 'mobility': 1.2, 'stability': 0.2,
                'corner': 1.8, 'frontier': -0.5, 'pattern': 0.8
            }
        elif empty_count > 0.3 * total_cells:  # 중반
            weights = {
                'position': 0.8, 'mobility': 0.9, 'stability': 0.7,
                'corner': 1.2, 'frontier': -0.3, 'pattern': 1.0
            }
        else:  # 후반
            weights = {
                'position': 0.6, 'mobility': 0.4, 'stability': 1.2,
                'corner': 0.8, 'frontier': -0.1, 'pattern': 0.5
            }

        def positional_score():
            score = 0
            position_weights = self.get_weights_for_stage(board)
            
            for i in range(8):
                for j in range(8):
                    if board.board[i][j] == self.color:
                        score += position_weights[i][j]
                    elif board.board[i][j] == opponent(self.color):
                        score -= position_weights[i][j]
            
            # 동적 X-square 패널티
            for corner_x, corner_y in CORNERS:
                if board.board[corner_x][corner_y] == EMPTY:
                    for x, y in X_SQUARES:
                        if abs(corner_x - x) <= 1 and abs(corner_y - y) <= 1:
                            if board.board[x][y] == self.color:
                                score -= 30
                            elif board.board[x][y] == opponent(self.color):
                                score += 30
            return score

        def mobility_score():
            my_moves = len(board.get_valid_moves(self.color))
            opp_moves = len(board.get_valid_moves(opponent(self.color)))
            
            if my_moves + opp_moves == 0:
                return 0
            
            # 상대적 이동성과 절대적 이동성 모두 고려
            relative_mobility = 100 * (my_moves - opp_moves) / (my_moves + opp_moves)
            absolute_mobility = my_moves - opp_moves
            
            return relative_mobility + absolute_mobility * 2

        def corner_score():
            my = sum(1 for x, y in CORNERS if board.board[x][y] == self.color)
            opp = sum(1 for x, y in CORNERS if board.board[x][y] == opponent(self.color))
            
            # 모서리 점유에 따른 비선형 보상
            score = 0
            if my > 0:
                score += 25 * my + 10 * my * my
            if opp > 0:
                score -= 25 * opp + 10 * opp * opp
            return score
            
        def stability_score():
            my_stable = sum(1 for i in range(8) for j in range(8) 
                          if board.board[i][j] == self.color and board.is_stable(i, j))
            opp_stable = sum(1 for i in range(8) for j in range(8) 
                           if board.board[i][j] == opponent(self.color) and board.is_stable(i, j))
            return 20 * (my_stable - opp_stable)

        def frontier_score():
            """프론티어 디스크 점수 (적을수록 좋음)"""
            my_frontier = board.get_frontier_count(self.color)
            opp_frontier = board.get_frontier_count(opponent(self.color))
            return opp_frontier - my_frontier  # 상대방 프론티어가 많을수록 좋음

        def pattern_score():
            return self.evaluate_patterns(board)

        # 종합 점수 계산
        final_score = (
            weights['position'] * positional_score() +
            weights['mobility'] * mobility_score() +
            weights['corner'] * corner_score() +
            weights['stability'] * stability_score() +
            weights['frontier'] * frontier_score() +
            weights['pattern'] * pattern_score()
        )
        
        return final_score

    def is_quiescent(self, board, last_move):
        """Quiescence 검사 - 불안정한 위치인지 확인"""
        if not last_move:
            return True
            
        x, y = last_move
        # 방금 둔 수가 모서리나 중요한 위치면 불안정
        if (x, y) in CORNERS or (x, y) in X_SQUARES or (x, y) in C_SQUARES:
            return False
            
        # 뒤집힌 돌의 수가 많으면 불안정
        if board.move_history and len(board.move_history[-1][3]) > 3:
            return False
            
        return True

    def quiescence_search(self, board, alpha, beta, maximizing, last_move=None):
        """Quiescence search - 불안정한 위치에서 추가 탐색"""
        if self.is_quiescent(board, last_move):
            return self.evaluate(board)
            
        current_color = self.color if maximizing else opponent(self.color)
        moves = board.get_valid_moves(current_color)
        
        if not moves:
            return self.evaluate(board)
            
        # 중요한 움직임만 고려 (모서리, 많은 돌 뒤집기)
        important_moves = []
        for move in moves:
            x, y = move
            if (x, y) in CORNERS:
                important_moves.append(move)
            else:
                new_board = board.apply_move(x, y, current_color)
                if new_board.move_history and len(new_board.move_history[-1][3]) > 2:
                    important_moves.append(move)
        
        if not important_moves:
            return self.evaluate(board)
            
        if maximizing:
            value = float('-inf')
            for move in important_moves:
                new_board = board.apply_move(*move, current_color)
                eval_score = self.quiescence_search(new_board, alpha, beta, False, move)
                value = max(value, eval_score)
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
        else:
            value = float('inf')
            for move in important_moves:
                new_board = board.apply_move(*move, current_color)
                eval_score = self.quiescence_search(new_board, alpha, beta, True, move)
                value = min(value, eval_score)
                beta = min(beta, value)
                if alpha >= beta:
                    break
                    
        return value

    def principal_variation_search(self, board, depth, alpha, beta, maximizing, start_time):
        """Principal Variation Search (PVS)"""
        if time.time() - start_time > self.time_limit:
            raise TimeoutError
            
        current_color = self.color if maximizing else opponent(self.color)
        moves = board.get_valid_moves(current_color)
        
        board_key = self.board_to_key(board)
        if board_key in self.transposition_table:
            entry = self.transposition_table[board_key]
            if entry['depth'] >= depth:
                if entry['type'] == 'exact':
                    return entry['value'], entry['move']
                elif entry['type'] == 'lowerbound' and entry['value'] >= beta:
                    return entry['value'], entry['move']
                elif entry['type'] == 'upperbound' and entry['value'] <= alpha:
                    return entry['value'], entry['move']

        if depth == 0 or not moves:
            if not moves:
                opponent_moves = board.get_valid_moves(opponent(current_color))
                if not opponent_moves:
                    b, w = board.count_stones()
                    if b == w:
                        return 0, None
                    score = 1000 if ((b > w and self.color == BLACK) or (w > b and self.color == WHITE)) else -1000
                    return score, None
                
                # Pass to opponent
                return self.principal_variation_search(board, depth, alpha, beta, not maximizing, start_time)
                
            # Quiescence search 적용
            return self.quiescence_search(board, alpha, beta, maximizing), None

        # 움직임 정렬 개선
        sorted_moves = self.enhanced_sort_moves(board, moves, current_color, depth)
        
        best_move = None
        first_child = True
        
        if maximizing:
            value = float('-inf')
            for i, move in enumerate(sorted_moves):
                new_board = board.apply_move(*move, current_color)
                
                if first_child:
                    # 첫 번째 자식은 full window로 탐색
                    eval_score, _ = self.principal_variation_search(
                        new_board, depth - 1, alpha, beta, False, start_time)
                    first_child = False
                else:
                    # Late Move Reduction 적용
                    reduction = 0
                    if i >= 4 and depth >= 3:  # 4번째 수부터 reduction 적용
                        reduction = 1
                    
                    # Null window search
                    eval_score, _ = self.principal_variation_search(
                        new_board, max(1, depth - 1 - reduction), alpha, alpha + 1, False, start_time)
                    
                    # Re-search if necessary
                    if alpha < eval_score < beta:
                        eval_score, _ = self.principal_variation_search(
                            new_board, depth - 1, alpha, beta, False, start_time)
                
                if eval_score > value:
                    value = eval_score
                    best_move = move
                    
                alpha = max(alpha, value)
                if alpha >= beta:
                    # Killer move 저장
                    if len(self.killer_moves[depth]) >= 2:
                        self.killer_moves[depth].pop(0)
                    self.killer_moves[depth].append(move)
                    break
                    
            # History heuristic 업데이트
            if best_move:
                self.history_table[best_move] += depth * depth
                
        else:
            value = float('inf')
            for i, move in enumerate(sorted_moves):
                new_board = board.apply_move(*move, current_color)
                
                if first_child:
                    eval_score, _ = self.principal_variation_search(
                        new_board, depth - 1, alpha, beta, True, start_time)
                    first_child = False
                else:
                    reduction = 0
                    if i >= 4 and depth >= 3:
                        reduction = 1
                        
                    eval_score, _ = self.principal_variation_search(
                        new_board, max(1, depth - 1 - reduction), beta - 1, beta, True, start_time)
                    
                    if alpha < eval_score < beta:
                        eval_score, _ = self.principal_variation_search(
                            new_board, depth - 1, alpha, beta, True, start_time)
                
                if eval_score < value:
                    value = eval_score
                    best_move = move
                    
                beta = min(beta, value)
                if alpha >= beta:
                    if len(self.killer_moves[depth]) >= 2:
                        self.killer_moves[depth].pop(0)
                    self.killer_moves[depth].append(move)
                    break
                    
            if best_move:
                self.history_table[best_move] += depth * depth
                
        # Transposition table 저장
        entry_type = 'exact'
        if value <= alpha:
            entry_type = 'upperbound'
        elif value >= beta:
            entry_type = 'lowerbound'
            
        self.transposition_table[board_key] = {
            'value': value,
            'move': best_move,
            'depth': depth,
            'type': entry_type
        }
        
        return value, best_move

    def enhanced_sort_moves(self, board, moves, color, depth):
        """개선된 움직임 정렬"""
        move_scores = []
        
        for move in moves:
            x, y = move
            score = 0
            
            # PV move 최우선
            board_key = self.board_to_key(board)
            if board_key in self.pv_table and self.pv_table[board_key] == move:
                score += 10000
            
            # Killer moves
            if move in self.killer_moves.get(depth, []):
                score += 1000
                
            # History heuristic
            score += self.history_table.get(move, 0)
            
            # 위치별 우선순위
            if move in CORNERS:
                score += 500
            elif move in X_SQUARES:
                score -= 200
            elif move in C_SQUARES:
                score -= 100
            elif x == 0 or x == 7 or y == 0 or y == 7:  # 변
                score += 100
                
            # 뒤집는 돌 수
            new_board = board.apply_move(x, y, color)
            if new_board.move_history:
                flipped = len(new_board.move_history[-1][3])
                score += flipped * 5
                
            # 상대방 이동성 제한
            opp_moves_before = len(board.get_valid_moves(opponent(color)))
            opp_moves_after = len(new_board.get_valid_moves(opponent(color)))
            score += (opp_moves_before - opp_moves_after) * 10
            
            move_scores.append((score, move))
            
        move_scores.sort(reverse=True)
        return [move for _, move in move_scores]

    def iterative_deepening(self, board):
        """Iterative Deepening with time management"""
        start_time = time.time()
        best_move = None
        best_score = float('-inf')
        
        moves = board.get_valid_moves(self.color)
        if not moves:
            return None
            
        if len(moves) == 1:
            return moves[0]
        
        # 최소 깊이 3부터 시작
        max_depth = min(12, board.get_empty_count())
        
        for depth in range(3, max_depth + 1):
            try:
                # 각 깊이마다 새로운 탐색
                current_score, current_move = self.principal_variation_search(
                    board, depth, float('-inf'), float('inf'), True, start_time
                )
                
                if current_move:
                    best_move = current_move
                    best_score = current_score
                    
                    # PV table 업데이트
                    board_key = self.board_to_key(board)
                    self.pv_table[board_key] = best_move
                
                # 시간 체크
                if time.time() - start_time > self.time_limit * 0.8:
                    break
                    
                # 확실한 승리/패배면 조기 종료
                if abs(current_score) > 900:
                    break
                    
            except TimeoutError:
                break
                
        return best_move

    def get_move(self, board):
        """최적의 움직임 결정"""
        # 엔드게임에서는 완전 탐색
        empty_count = board.get_empty_count()
        if empty_count <= 8:
            return self.get_endgame_move(board)
            
        # 초반 전략
        if empty_count > 55:
            return self.get_opening_move(board)
            
        # 테이블 주기적 정리
        if len(self.transposition_table) > 100000:
            self.transposition_table.clear()
            
        return self.iterative_deepening(board)

    def get_opening_move(self, board):
        """개선된 오프닝 전략"""
        moves = board.get_valid_moves(self.color)
        if not moves:
            return None

        # 표준 오프닝 패턴
        preferred_openings = [
            (2, 3), (3, 2), (4, 5), (5, 4),
            (2, 4), (3, 5), (4, 2), (5, 3)
        ]

        for move in preferred_openings:
            if move in moves:
                return move

        # 안전한 수 선택
        safe_moves = [m for m in moves if m not in X_SQUARES and m not in C_SQUARES]
        if safe_moves:
            center_moves = []
            for move in safe_moves:
                x, y = move
                center_distance = abs(x - 3.5) + abs(y - 3.5)
                center_moves.append((center_distance, move))

            center_moves.sort()
            return center_moves[0][1]

        # 모든 수가 위험한 경우 랜덤 선택
        return random.choice(moves)

    def get_endgame_move(self, board):
        """엔드게임 완전 탐색"""
        best_move = None
        best_score = float('-inf')
        current_color = self.color
        moves = board.get_valid_moves(current_color)

        if not moves:
            return None

        for move in moves:
            new_board = board.apply_move(*move, current_color)
            score = self.minimax_final_count(new_board, False)
            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def minimax_final_count(self, board, maximizing):
        """엔드게임 완전 탐색 재귀"""
        current_color = self.color if maximizing else opponent(self.color)
        moves = board.get_valid_moves(current_color)

        if not moves:
            opponent_moves = board.get_valid_moves(opponent(current_color))
            if not opponent_moves:
                b, w = board.count_stones()
                return b - w if self.color == BLACK else w - b
            else:
                return self.minimax_final_count(board, not maximizing)

        best = float('-inf') if maximizing else float('inf')
        for move in moves:
            new_board = board.apply_move(*move, current_color)
            score = self.minimax_final_count(new_board, not maximizing)
            if maximizing:
                best = max(best, score)
            else:
                best = min(best, score)

        return best
class OthelloGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Othello (Enhanced Alpha-Beta AI)")
        self.cell_size = 60
        self.board = Board()

        player_choice = simpledialog.askstring("Player Color", "Do you want to play first? (yes/no)").lower()
        if player_choice in ['yes', 'y']:
            self.current_player = BLACK
            self.ai = EnhancedAlphaBetaAI(WHITE)
        else:
            self.current_player = WHITE
            self.ai = EnhancedAlphaBetaAI(BLACK)

        self.canvas = tk.Canvas(self.root, width=self.cell_size * 8, height=self.cell_size * 8, bg="dark green")
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.handle_click)

        self.status_label = tk.Label(self.root, text="Game Start", font=("Arial", 14))
        self.status_label.pack()

        self.update_gui()

        if self.current_player == WHITE:
            self.root.after(500, self.ai_move)

    def draw_board(self):
        self.canvas.delete("all")
        for i in range(8):
            for j in range(8):
                x1, y1 = j * self.cell_size, i * self.cell_size
                x2, y2 = x1 + self.cell_size, y1 + self.cell_size
                self.canvas.create_rectangle(x1, y1, x2, y2, outline="black")

                stone = self.board.board[i][j]
                if stone == BLACK:
                    self.canvas.create_oval(x1 + 5, y1 + 5, x2 - 5, y2 - 5, fill="black")
                elif stone == WHITE:
                    self.canvas.create_oval(x1 + 5, y1 + 5, x2 - 5, y2 - 5, fill="white")

    def update_gui(self):
        self.draw_board()
        b_count, w_count = self.board.count_stones()
        status = f"Black: {b_count}  White: {w_count}"
        self.status_label.config(text=status)

    def handle_click(self, event):
        if self.current_player != self.ai.color:
            x, y = event.y // self.cell_size, event.x // self.cell_size
            if self.board.is_valid_move(x, y, self.current_player):
                self.board = self.board.apply_move(x, y, self.current_player)
                self.current_player = opponent(self.current_player)
                self.update_gui()
                self.root.after(500, self.ai_move)

    def ai_move(self):
        move = self.ai.get_move(self.board)
        if move:
            self.board = self.board.apply_move(*move, self.ai.color)
            self.current_player = opponent(self.ai.color)
            self.update_gui()

        # 다음 차례가 사용자 차례인데 둘 수 없으면 AI 다시 둠
        if not self.board.get_valid_moves(self.current_player):
            if not self.board.get_valid_moves(self.ai.color):
                self.end_game()
            else:
                self.root.after(500, self.ai_move)
        elif not self.board.get_valid_moves(self.ai.color) and not self.board.get_valid_moves(self.current_player):
            self.end_game()

    def end_game(self):
        b, w = self.board.count_stones()
        if b > w:
            winner = "Black wins!"
        elif w > b:
            winner = "White wins!"
        else:
            winner = "Draw!"

        messagebox.showinfo("Game Over", f"Game over!\nBlack: {b}  White: {w}\n{winner}")
        self.root.quit()
        self.root.destroy()
if __name__ == "__main__":
    root = tk.Tk()
    app = OthelloGUI(root)
    root.mainloop()
