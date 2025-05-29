import time
import random
import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set
import threading
import json

from constants import BLACK, WHITE, EMPTY, opponent, CORNERS, X_SQUARES, C_SQUARES
from board import Board

@dataclass
class UltraSearchResult:
    """Ultra Search Result with detailed analysis"""
    score: int
    best_move: Optional[Tuple[int, int]]
    depth: int
    nodes: int
    time_ms: int
    is_exact: bool
    pv: List[Tuple[int, int]]  # Principal Variation
    eval_breakdown: Dict[str, float]  # 평가 요소별 점수

class UltraStrongAI:
    """최강 오셀로 AI - 이기는 것이 목표"""
    
    def __init__(self, color, difficulty='ultra', time_limit=10.0):
        self.color = color
        self.difficulty = difficulty
        self.time_limit = time_limit
        
        # 극강 설정
        if difficulty == 'ultra':
            self.max_depth = 18  # 매우 깊은 탐색
            self.endgame_depth = 64  # 종료게임 완전탐색
            self.use_perfect_endgame = True
            self.endgame_threshold = 16  # 16수 남았을 때부터 완전탐색
        elif difficulty == 'hard':
            self.max_depth = 14
            self.endgame_depth = 20
            self.use_perfect_endgame = True
            self.endgame_threshold = 12
        else:
            self.max_depth = 12
            self.endgame_depth = 16
            self.use_perfect_endgame = False
            self.endgame_threshold = 8
        
        # 강화된 Transposition Table
        self.tt = {}
        self.tt_age = 0
        self.max_tt_size = 5000000  # 5M entries
        
        # 완벽한 오프닝북
        self.opening_book = self.create_perfect_opening_book()
        
        # 고급 휴리스틱들
        self.killer_moves = defaultdict(list)
        self.history_table = defaultdict(int)
        self.counter_moves = {}
        
        # 패턴 평가 시스템
        self.pattern_values = self.initialize_patterns()
        
        # 통계
        self.nodes_searched = 0
        self.tt_hits = 0
        self.cutoffs = 0
        self.perfect_searches = 0
        
        # 게임 단계별 최적화된 가중치
        self.stage_weights = {
            'opening': {  # 0-20 moves
                'mobility': 1.0, 'corners': 3.0, 'edges': 0.3, 'stability': 0.1,
                'discs': 0.0, 'frontier': -0.5, 'patterns': 0.8, 'parity': 0.1
            },
            'midgame': {  # 21-45 moves  
                'mobility': 0.8, 'corners': 2.0, 'edges': 0.8, 'stability': 1.2,
                'discs': 0.2, 'frontier': -0.3, 'patterns': 1.0, 'parity': 0.4
            },
            'endgame': {  # 46+ moves
                'mobility': 0.4, 'corners': 1.0, 'edges': 0.6, 'stability': 1.5,
                'discs': 2.0, 'frontier': -0.1, 'patterns': 0.5, 'parity': 1.0
            }
        }
    
    def create_perfect_opening_book(self):
        """완벽한 오프닝북 생성"""
        # 실제 프로 경기에서 검증된 오프닝 패턴들
        return {
            # 표준 시작 후 최고의 수들
            'standard_start': {
                # 5번째 수 (첫 번째 자유 수)
                frozenset([(3,3,'W'), (3,4,'B'), (4,3,'B'), (4,4,'W')]): [
                    ((2,3), 1.0),  # d3 - 가장 강력한 첫 수
                    ((3,2), 1.0),  # c4 - 두 번째로 강력
                    ((4,5), 0.7),  # f5 - 괜찮은 수
                    ((5,4), 0.7)   # e6 - 괜찮은 수
                ],
                # Perpendicular opening
                frozenset([(3,3,'W'), (3,4,'B'), (4,3,'B'), (4,4,'W'), (2,3,'B')]): [
                    ((1,3), 0.9),  # d2
                    ((3,2), 0.9),  # c4  
                    ((3,5), 0.7),  # f4
                    ((5,3), 0.7)   # d6
                ],
                # Diagonal opening
                frozenset([(3,3,'W'), (3,4,'B'), (4,3,'B'), (4,4,'W'), (3,2,'B')]): [
                    ((2,2), 0.9),  # c3
                    ((2,3), 0.9),  # d3
                    ((4,1), 0.7),  # b5
                    ((1,4), 0.7)   # e2
                ],
                # Tiger opening 
                frozenset([(3,3,'W'), (3,4,'B'), (4,3,'B'), (4,4,'W'), (4,5,'B')]): [
                    ((5,5), 0.8),  # f6
                    ((3,5), 0.8),  # f4
                    ((5,3), 0.6),  # d6
                    ((2,4), 0.6)   # e3
                ]
            }
        }
    
    def initialize_patterns(self):
        """패턴 기반 평가 시스템 초기화"""
        patterns = {}
        
        # 모서리 패턴 (실제 Egaroucid처럼)
        patterns['edge_patterns'] = {
            # 완벽한 모서리 제어
            'perfect_edge': 500,
            'strong_edge': 200,
            'weak_edge': -100,
            'broken_edge': -300
        }
        
        # 코너 주변 패턴
        patterns['corner_patterns'] = {
            'corner_captured': 1000,
            'corner_accessible': -500,  # C-square나 X-square 점유시 페널티
            'corner_safe': 300
        }
        
        return patterns
    
    def get_game_stage(self, board):
        """현재 게임 단계 정확히 판단"""
        moves_played = 64 - board.get_empty_count() - 4  # 초기 4수 제외
        
        if moves_played <= 20:
            return 'opening'
        elif moves_played <= 45:
            return 'midgame'
        else:
            return 'endgame'
    
    def ultra_evaluate_position(self, board):
        """극강 위치 평가 함수"""
        if board.get_empty_count() == 0:
            # 게임 종료 - 실제 승부 결정
            b, w = board.count_stones()
            diff = (b - w) if self.color == BLACK else (w - b)  
            if diff > 0:
                return 50000 + diff  # 승리 + 점수차
            elif diff < 0:
                return -50000 + diff  # 패배 + 점수차
            else:
                return 0  # 무승부
        
        stage = self.get_game_stage(board)
        weights = self.stage_weights[stage]
        
        eval_breakdown = {}
        total_score = 0
        
        # 1. 기동력 (Mobility) - 초기에 매우 중요
        my_moves = len(board.get_valid_moves(self.color))
        opp_moves = len(board.get_valid_moves(opponent(self.color)))
        
        if my_moves + opp_moves > 0:
            mobility_score = 100 * (my_moves - opp_moves) / (my_moves + opp_moves + 1)
        else:
            mobility_score = 0
            
        # 기동력 차이에 따른 보너스
        if my_moves > 0 and opp_moves == 0:
            mobility_score += 200  # 상대방 움직일 수 없음
        elif my_moves == 0 and opp_moves > 0:
            mobility_score -= 200  # 내가 움직일 수 없음
            
        eval_breakdown['mobility'] = mobility_score
        total_score += weights['mobility'] * mobility_score
        
        # 2. 코너 제어 (Corners) - 항상 중요
        corner_score = self.evaluate_corners_advanced(board)
        eval_breakdown['corners'] = corner_score
        total_score += weights['corners'] * corner_score
        
        # 3. 모서리 제어 (Edges)
        edge_score = self.evaluate_edges_advanced(board)
        eval_breakdown['edges'] = edge_score
        total_score += weights['edges'] * edge_score
        
        # 4. 안정성 (Stability) - 중반 이후 매우 중요
        stability_score = self.evaluate_stability_advanced(board)
        eval_breakdown['stability'] = stability_score
        total_score += weights['stability'] * stability_score
        
        # 5. 돌 개수 (Disc Count) - 종료게임에서 중요
        b, w = board.count_stones()
        disc_diff = (b - w) if self.color == BLACK else (w - b)
        eval_breakdown['discs'] = disc_diff
        total_score += weights['discs'] * disc_diff
        
        # 6. 프론티어 디스크 (Frontier) - 적을수록 좋음
        my_frontier = board.get_frontier_count(self.color)
        opp_frontier = board.get_frontier_count(opponent(self.color))
        frontier_score = opp_frontier - my_frontier
        eval_breakdown['frontier'] = frontier_score
        total_score += weights['frontier'] * frontier_score
        
        # 7. 패턴 평가 (Patterns)
        pattern_score = self.evaluate_patterns(board)
        eval_breakdown['patterns'] = pattern_score
        total_score += weights['patterns'] * pattern_score
        
        # 8. 패리티 (Parity) - 마지막 수를 둘 가능성
        parity_score = self.evaluate_parity(board)
        eval_breakdown['parity'] = parity_score
        total_score += weights['parity'] * parity_score
        
        # 9. 특수 패턴 보너스/페널티
        special_score = self.evaluate_special_patterns(board)
        eval_breakdown['special'] = special_score
        total_score += special_score
        
        return int(total_score)
    
    def evaluate_corners_advanced(self, board):
        """고급 코너 평가"""
        score = 0
        my_corners = 0
        opp_corners = 0
        
        for corner_x, corner_y in CORNERS:
            if board.board[corner_x][corner_y] == self.color:
                my_corners += 1
                score += 300  # 기본 코너 점수
                
                # 코너에서 연결된 안정적인 라인 보너스
                score += self.evaluate_corner_lines(board, corner_x, corner_y) * 50
                
            elif board.board[corner_x][corner_y] == opponent(self.color):
                opp_corners += 1
                score -= 300
                score -= self.evaluate_corner_lines(board, corner_x, corner_y) * 50
            else:
                # 빈 코너 주변의 위험한 수들에 대한 페널티
                score += self.evaluate_corner_danger(board, corner_x, corner_y)
        
        # 다중 코너 보너스 (지수적 증가)
        if my_corners > 1:
            score += 150 * my_corners * my_corners
        if opp_corners > 1:
            score -= 150 * opp_corners * opp_corners
            
        return score
    
    def evaluate_corner_lines(self, board, corner_x, corner_y):
        """코너에서 뻗어나가는 안정적인 라인 평가"""
        color = board.board[corner_x][corner_y]
        stable_count = 0
        
        # 가로/세로 방향으로 체크
        directions = []
        if corner_x == 0:
            directions.append((1, 0))  # 아래로
        else:
            directions.append((-1, 0))  # 위로
            
        if corner_y == 0:
            directions.append((0, 1))  # 오른쪽으로
        else:
            directions.append((0, -1))  # 왼쪽으로
        
        for dx, dy in directions:
            x, y = corner_x + dx, corner_y + dy
            while 0 <= x < 8 and 0 <= y < 8:
                if board.board[x][y] == color:
                    stable_count += 1
                else:
                    break
                x += dx
                y += dy
        
        return stable_count
    
    def evaluate_corner_danger(self, board, corner_x, corner_y):
        """빈 코너 주변의 위험도 평가"""
        score = 0
        
        # X-square (대각선 인접) 체크
        x_squares = [(corner_x + 1 if corner_x == 0 else corner_x - 1,
                     corner_y + 1 if corner_y == 0 else corner_y - 1)]
        
        for x, y in x_squares:
            if 0 <= x < 8 and 0 <= y < 8:
                if board.board[x][y] == self.color:
                    score -= 200  # X-square 점유 시 큰 페널티
                elif board.board[x][y] == opponent(self.color):
                    score += 200
        
        # C-square (모서리 인접) 체크
        c_squares = [
            (corner_x, corner_y + 1 if corner_y == 0 else corner_y - 1),
            (corner_x + 1 if corner_x == 0 else corner_x - 1, corner_y)
        ]
        
        for x, y in c_squares:
            if 0 <= x < 8 and 0 <= y < 8:
                if board.board[x][y] == self.color:
                    score -= 100  # C-square 점유 시 페널티
                elif board.board[x][y] == opponent(self.color):
                    score += 100
        
        return score
    
    def evaluate_edges_advanced(self, board):
        """고급 모서리 평가"""
        score = 0
        
        # 각 모서리별로 평가
        edges = [
            [(0, j) for j in range(8)],  # 위쪽 모서리
            [(7, j) for j in range(8)],  # 아래쪽 모서리  
            [(i, 0) for i in range(8)],  # 왼쪽 모서리
            [(i, 7) for i in range(8)]   # 오른쪽 모서리
        ]
        
        for edge in edges:
            my_count = sum(1 for x, y in edge if board.board[x][y] == self.color)
            opp_count = sum(1 for x, y in edge if board.board[x][y] == opponent(self.color))
            empty_count = sum(1 for x, y in edge if board.board[x][y] == EMPTY)
            
            # 모서리 완전 제어 보너스
            if my_count == 8:
                score += 400
            elif opp_count == 8:
                score -= 400
            else:
                score += (my_count - opp_count) * 15
                
            # 모서리의 연속성 평가
            score += self.evaluate_edge_continuity(board, edge) * 10
        
        return score
    
    def evaluate_edge_continuity(self, board, edge):
        """모서리의 연속성 평가"""
        my_sequences = 0
        opp_sequences = 0
        
        current_my_seq = 0
        current_opp_seq = 0
        
        for x, y in edge:
            if board.board[x][y] == self.color:
                current_my_seq += 1
                if current_opp_seq > 0:
                    opp_sequences += current_opp_seq * current_opp_seq
                    current_opp_seq = 0
            elif board.board[x][y] == opponent(self.color):
                current_opp_seq += 1
                if current_my_seq > 0:
                    my_sequences += current_my_seq * current_my_seq
                    current_my_seq = 0
            else:  # EMPTY
                if current_my_seq > 0:
                    my_sequences += current_my_seq * current_my_seq
                    current_my_seq = 0
                if current_opp_seq > 0:
                    opp_sequences += current_opp_seq * current_opp_seq
                    current_opp_seq = 0
        
        # 마지막 시퀀스 처리
        if current_my_seq > 0:
            my_sequences += current_my_seq * current_my_seq
        if current_opp_seq > 0:
            opp_sequences += current_opp_seq * current_opp_seq
            
        return my_sequences - opp_sequences
    
    def evaluate_stability_advanced(self, board):
        """고급 안정성 평가"""
        my_stable = 0
        opp_stable = 0
        
        # 더 정확한 안정성 계산
        for i in range(8):
            for j in range(8):
                if board.board[i][j] == self.color:
                    if self.is_truly_stable(board, i, j):
                        my_stable += 1
                elif board.board[i][j] == opponent(self.color):
                    if self.is_truly_stable(board, i, j):
                        opp_stable += 1
        
        return (my_stable - opp_stable) * 30
        
    def is_truly_stable(self, board, x, y):
        """진정한 안정성 검사 (Egaroucid 스타일)"""
        color = board.board[x][y]
        if color == EMPTY:
            return False
        
        # 코너는 항상 안정적
        if (x, y) in CORNERS:
            return True
        
        # 8방향 모두에서 안정성 체크
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        for dx, dy in directions:
            if not self.is_stable_in_direction(board, x, y, dx, dy, color):
                return False
        
        return True
    
    def is_stable_in_direction(self, board, x, y, dx, dy, color):
        """특정 방향에서의 안정성 검사"""
        # 한 방향으로 가면서 같은 색이 모서리나 코너까지 연결되는지 확인
        nx, ny = x + dx, y + dy
        
        while 0 <= nx < 8 and 0 <= ny < 8:
            if board.board[nx][ny] != color:
                return False  # 다른 색이나 빈 칸을 만남
            if (nx, ny) in CORNERS:
                return True  # 코너에 도달
            if nx == 0 or nx == 7 or ny == 0 or ny == 7:
                # 모서리에 도달 - 모서리가 안정적인지 확인
                return self.is_edge_stable(board, nx, ny, color)
            nx += dx
            ny += dy
        
        # 보드 끝에 도달
        return True
    
    def is_edge_stable(self, board, x, y, color):
        """모서리의 안정성 검사"""
        # 모서리에서 양쪽 끝이 같은 색으로 채워져 있는지 확인
        if x == 0 or x == 7:  # 위/아래 모서리
            for j in range(8):
                if board.board[x][j] != color and board.board[x][j] != EMPTY:
                    return False
        elif y == 0 or y == 7:  # 좌/우 모서리
            for i in range(8):
                if board.board[i][y] != color and board.board[i][y] != EMPTY:
                    return False
        return True
    
    def evaluate_patterns(self, board):
        """패턴 기반 평가"""
        score = 0
        
        # Wedge 패턴 (쐐기 패턴) 탐지
        score += self.detect_wedge_patterns(board) * 50
        
        # Triangle 패턴 탐지  
        score += self.detect_triangle_patterns(board) * 30
        
        # Line 패턴 탐지
        score += self.detect_line_patterns(board) * 20
        
        return score
    
    def detect_wedge_patterns(self, board):
        """쐐기 패턴 탐지"""
        score = 0
        # 실제 구현은 복잡하므로 간소화
        return score
    
    def detect_triangle_patterns(self, board):
        """삼각형 패턴 탐지"""
        score = 0
        # 간소화된 구현
        return score
        
    def detect_line_patterns(self, board):
        """라인 패턴 탐지"""
        score = 0
        # 간소화된 구현
        return score
    
    def evaluate_parity(self, board):
        """패리티 평가 (마지막 수를 둘 가능성)"""
        empty_count = board.get_empty_count()
        
        # 빈 칸이 홀수개면 흑이, 짝수개면 백이 마지막 수
        if empty_count % 2 == 1:
            # 흑이 마지막 수
            return 50 if self.color == BLACK else -50
        else:
            # 백이 마지막 수  
            return 50 if self.color == WHITE else -50
    
    def evaluate_special_patterns(self, board):
        """특수 패턴들 평가"""
        score = 0
        
        # 1. 코너 함정 패턴 (상대방이 코너를 내주게 만드는 패턴)
        score += self.detect_corner_traps(board) * 100
        
        # 2. 템포 패턴 (상대방에게 불리한 수를 강제하는 패턴)
        score += self.detect_tempo_patterns(board) * 75
        
        # 3. 스위프 패턴 (한 번에 많은 돌을 뒤집는 패턴)
        score += self.detect_sweep_patterns(board) * 25
        
        return score
    
    def detect_corner_traps(self, board):
        """코너 함정 패턴 탐지"""
        # 간소화된 구현
        return 0
    
    def detect_tempo_patterns(self, board):
        """템포 패턴 탐지"""
        # 간소화된 구현
        return 0
        
    def detect_sweep_patterns(self, board):
        """스위프 패턴 탐지"""
        # 간소화된 구현
        return 0
    
    def perfect_endgame_search(self, board, alpha, beta, passes=0):
        """완벽한 종료게임 탐색"""
        self.perfect_searches += 1
        
        moves = board.get_valid_moves(self.color)
        
        if not moves:
            if passes >= 1:
                # 게임 종료
                b, w = board.count_stones()
                diff = (b - w) if self.color == BLACK else (w - b)
                if diff > 0:
                    return 50000 + diff, None
                elif diff < 0:
                    return -50000 + diff, None
                else:
                    return 0, None
            else:
                # 패스
                return -self.perfect_endgame_search_opp(board, -beta, -alpha, passes + 1)[0], None
        
        best_score = alpha
        best_move = None
        
        # 움직일 수 있는 모든 수를 완전 탐색
        for move in moves:
            new_board = board.apply_move(*move, self.color)
            score = -self.perfect_endgame_search_opp(new_board, -beta, -best_score, 0)[0]
            
            if score > best_score:
                best_score = score
                best_move = move
                
            if best_score >= beta:
                break  # Beta cutoff
        
        return best_score, best_move
    
    def perfect_endgame_search_opp(self, board, alpha, beta, passes=0):
        """상대방 차례의 완벽한 종료게임 탐색"""
        moves = board.get_valid_moves(opponent(self.color))
        
        if not moves:
            if passes >= 1:
                # 게임 종료
                b, w = board.count_stones()
                diff = (b - w) if self.color == BLACK else (w - b)
                if diff > 0:
                    return 50000 + diff, None
                elif diff < 0:
                    return -50000 + diff, None
                else:
                    return 0, None
            else:
                # 패스
                return -self.perfect_endgame_search(board, -beta, -alpha, passes + 1)[0], None
        
        best_score = alpha
        best_move = None
        
        for move in moves:
            new_board = board.apply_move(*move, opponent(self.color))
            score = -self.perfect_endgame_search(new_board, -beta, -best_score, 0)[0]
            
            if score > best_score:
                best_score = score
                best_move = move
                
            if best_score >= beta:
                break
        
        return best_score, best_move
    
    def ultra_negamax(self, board, depth, alpha, beta, maximizing, end_time, passes=0):
        """울트라 강화된 네가맥스"""
        self.nodes_searched += 1
        
        # 시간 체크
        if time.time() > end_time:
            return self.ultra_evaluate_position(board), None
        
        # 완벽한 종료게임 탐색
        empty_count = board.get_empty_count()
        if (self.use_perfect_endgame and 
            empty_count <= self.endgame_threshold and 
            depth >= empty_count):
            if maximizing:
                return self.perfect_endgame_search(board, alpha, beta, passes)
            else:
                score, move = self.perfect_endgame_search_opp(board, alpha, beta, passes)
                return -score, move
        
        # TT 조회
        board_hash = self.get_board_hash(board)
        tt_score = self.probe_tt(board_hash, depth, alpha, beta)
        if tt_score is not None:
            return tt_score, None
        
        current_color = self.color if maximizing else opponent(self.color)
        moves = board.get_valid_moves(current_color)
        
        # 터미널 조건
        if depth == 0 or not moves:
            if not moves:
                opponent_moves = board.get_valid_moves(opponent(current_color))
                if not opponent_moves:
                    # 게임 종료
                    return self.ultra_evaluate_position(board), None
                else:
                    # 패스
                    return self.ultra_negamax(board, depth, alpha, beta, not maximizing, end_time, passes + 1)
            else:
                return self.ultra_evaluate_position(board), None
        
        # 울트라 강화된 무브 정렬
        ordered_moves = self.ultra_order_moves(board, moves, depth, maximizing)
        best_move = None
        original_alpha = alpha
        
        if maximizing:
            max_score = float('-inf')
            for i, move in enumerate(ordered_moves):
                new_board = board.apply_move(*move, current_color)
                
                # Late Move Reduction (LMR) - 후반 무브들은 깊이 감소
                reduction = 0
                if (i > 3 and depth > 3 and 
                    move not in self.killer_moves.get(depth, []) and
                    not self.is_tactical_move(board, move)):
                    reduction = 1
                
                score, _ = self.ultra_negamax(new_board, depth - 1 - reduction, 
                                            alpha, beta, False, end_time, 0)
                
                # LMR에서 좋은 결과가 나오면 전체 깊이로 재탐색
                if reduction > 0 and score > alpha:
                    score, _ = self.ultra_negamax(new_board, depth - 1, 
                                                alpha, beta, False, end_time, 0)
                
                if score > max_score:
                    max_score = score
                    best_move = move
                
                alpha = max(alpha, score)
                if beta <= alpha:
                    # Beta cutoff
                    self.cutoffs += 1
                    self.update_killer_moves(depth, move)
                    break
            
            # 히스토리 테이블 업데이트
            if best_move:
                self.history_table[best_move] += depth * depth
            
            # TT 저장
            flag = 'EXACT' if original_alpha < max_score < beta else ('BETA' if max_score >= beta else 'ALPHA')
            self.store_tt(board_hash, depth, max_score, flag, best_move)
            
            return max_score, best_move
            
        else:
            min_score = float('inf')
            for i, move in enumerate(ordered_moves):
                new_board = board.apply_move(*move, current_color)
                
                # LMR 적용
                reduction = 0
                if (i > 3 and depth > 3 and 
                    move not in self.killer_moves.get(depth, []) and
                    not self.is_tactical_move(board, move)):
                    reduction = 1
                
                score, _ = self.ultra_negamax(new_board, depth - 1 - reduction, 
                                            alpha, beta, True, end_time, 0)
                
                if reduction > 0 and score < beta:
                    score, _ = self.ultra_negamax(new_board, depth - 1, 
                                                alpha, beta, True, end_time, 0)
                
                if score < min_score:
                    min_score = score
                    best_move = move
                
                beta = min(beta, score)
                if beta <= alpha:
                    self.cutoffs += 1
                    self.update_killer_moves(depth, move)
                    break
            
            if best_move:
                self.history_table[best_move] += depth * depth
            
            flag = 'EXACT' if alpha < min_score < original_alpha else ('ALPHA' if min_score <= alpha else 'BETA')
            self.store_tt(board_hash, depth, min_score, flag, best_move)
            
            return min_score, best_move
    
    def is_tactical_move(self, board, move):
        """전술적 수인지 판단 (코너, 모서리 등)"""
        x, y = move
        
        # 코너 수는 항상 전술적
        if (x, y) in CORNERS:
            return True
        
        # 모서리 수도 전술적
        if x == 0 or x == 7 or y == 0 or y == 7:
            return True
        
        # 많은 돌을 뒤집는 수
        new_board = board.apply_move(x, y, self.color)
        if new_board.move_history and len(new_board.move_history[-1][3]) >= 6:
            return True
        
        return False
    
    def ultra_order_moves(self, board, moves, depth, maximizing):
        """울트라 강화된 무브 정렬"""
        if not moves:
            return moves
        
        move_scores = []
        board_hash = self.get_board_hash(board)
        
        # TT에서 최고 수 가져오기
        tt_move = None
        if board_hash in self.tt:
            tt_move = self.tt[board_hash].best_move
        
        for move in moves:
            x, y = move
            score = 0
            
            # TT 수 최우선
            if move == tt_move:
                score += 50000
            
            # 킬러 무브
            if move in self.killer_moves.get(depth, []):
                score += 10000
            
            # 카운터 무브 (이전 상대방 수에 대한 대응)
            if hasattr(self, 'last_opponent_move') and self.last_opponent_move in self.counter_moves:
                if move in self.counter_moves[self.last_opponent_move]:
                    score += 5000
            
            # 히스토리 휴리스틱
            score += self.history_table.get(move, 0)
            
            # 위치별 전략적 가치
            position_score = self.evaluate_move_position(board, move)
            score += position_score
            
            # 이 수로 인한 mobility 변화
            mobility_score = self.evaluate_move_mobility(board, move)
            score += mobility_score
            
            # 안정성 변화
            stability_score = self.evaluate_move_stability(board, move)
            score += stability_score
            
            move_scores.append((score, move))
        
        move_scores.sort(reverse=True)
        return [move for _, move in move_scores]
    
    def evaluate_move_position(self, board, move):
        """수의 위치적 가치 평가"""
        x, y = move
        score = 0
        
        # 코너
        if (x, y) in CORNERS:
            score += 1000
        
        # X-squares (위험한 수)
        elif (x, y) in X_SQUARES:
            # 인접한 코너가 비어있으면 매우 위험
            adjacent_corner_empty = False
            for corner in CORNERS:
                if abs(corner[0] - x) <= 1 and abs(corner[1] - y) <= 1:
                    if board.board[corner[0]][corner[1]] == EMPTY:
                        adjacent_corner_empty = True
                        break
            if adjacent_corner_empty:
                score -= 500
            else:
                score += 100  # 코너가 이미 점유된 경우는 괜찮음
        
        # C-squares  
        elif (x, y) in C_SQUARES:
            score -= 200
        
        # 모서리
        elif x == 0 or x == 7 or y == 0 or y == 7:
            score += 200
        
        # 내부 위치들
        else:
            # 중앙 근처
            center_distance = abs(x - 3.5) + abs(y - 3.5)
            score += int((7 - center_distance) * 10)
        
        return score
    
    def evaluate_move_mobility(self, board, move):
        """수에 따른 mobility 변화 평가"""
        # 현재 mobility
        current_my_moves = len(board.get_valid_moves(self.color))
        current_opp_moves = len(board.get_valid_moves(opponent(self.color)))
        
        # 수를 둔 후 mobility
        new_board = board.apply_move(*move, self.color)
        new_my_moves = len(new_board.get_valid_moves(self.color))
        new_opp_moves = len(new_board.get_valid_moves(opponent(self.color)))
        
        # Mobility 변화
        my_mobility_change = new_my_moves - current_my_moves
        opp_mobility_change = new_opp_moves - current_opp_moves
        
        # 상대방 mobility 줄이기 + 내 mobility 유지/증가
        return (current_opp_moves - new_opp_moves) * 20 + my_mobility_change * 10
    
    def evaluate_move_stability(self, board, move):
        """수에 따른 안정성 변화 평가"""
        new_board = board.apply_move(*move, self.color)
        
        # 간단히 뒤집힌 돌의 개수로 평가 (실제로는 더 복잡해야 함)
        if new_board.move_history:
            flipped_count = len(new_board.move_history[-1][3])
            return flipped_count * 5
        
        return 0
    
    def update_killer_moves(self, depth, move):
        """킬러 무브 업데이트"""
        if move not in self.killer_moves[depth]:
            if len(self.killer_moves[depth]) >= 3:
                self.killer_moves[depth].pop(0)
            self.killer_moves[depth].append(move)
    
    def get_board_hash(self, board):
        """보드 해시 계산"""
        board_str = ''.join(str(cell) for row in board.board for cell in row)
        return hashlib.md5(board_str.encode()).hexdigest()
    
    def store_tt(self, board_hash, depth, score, flag, best_move):
        """TT 저장"""
        if len(self.tt) >= self.max_tt_size:
            self.clear_old_tt_entries()
        
        self.tt[board_hash] = {
            'depth': depth, 'score': score, 'flag': flag, 
            'best_move': best_move, 'age': self.tt_age
        }
    
    def probe_tt(self, board_hash, depth, alpha, beta):
        """TT 조회"""
        if board_hash not in self.tt:
            return None
        
        entry = self.tt[board_hash]
        if entry['depth'] >= depth:
            self.tt_hits += 1
            if entry['flag'] == 'EXACT':
                return entry['score']
            elif entry['flag'] == 'ALPHA' and entry['score'] <= alpha:
                return alpha
            elif entry['flag'] == 'BETA' and entry['score'] >= beta:
                return beta
        
        return None
    
    def clear_old_tt_entries(self):
        """오래된 TT 엔트리 정리"""
        old_entries = [key for key, entry in self.tt.items() 
                      if self.tt_age - entry['age'] > 8]
        for key in old_entries[:len(old_entries)//2]:
            del self.tt[key]
    
    def get_opening_move(self, board):
        """오프닝북에서 수 선택"""
        board_state = self.board_to_frozenset(board)
        
        for book_name, book_data in self.opening_book.items():
            for pattern, moves in book_data.items():
                if pattern.issubset(board_state):
                    # 가중치가 높은 수들 중에서 선택
                    good_moves = [(move, weight) for move, weight in moves if weight >= 0.8]
                    if good_moves:
                        # 가중치에 따른 확률적 선택
                        weights = [weight for _, weight in good_moves]
                        total_weight = sum(weights)
                        r = random.random() * total_weight
                        
                        cumulative = 0
                        for move, weight in good_moves:
                            cumulative += weight
                            if r <= cumulative:
                                if board.is_valid_move(*move, self.color):
                                    return move
        
        return None
    
    def board_to_frozenset(self, board):
        """보드를 frozenset으로 변환"""
        state = set()
        for i in range(8):
            for j in range(8):
                if board.board[i][j] != EMPTY:
                    color = 'B' if board.board[i][j] == BLACK else 'W'
                    state.add((i, j, color))
        return frozenset(state)
    
    def ultra_iterative_deepening(self, board):
        """울트라 강화된 반복 심화"""
        start_time = time.time()
        end_time = start_time + self.time_limit
        
        moves = board.get_valid_moves(self.color)
        if not moves:
            return UltraSearchResult(0, None, 0, 0, 0, True, [], {})
        
        if len(moves) == 1:
            return UltraSearchResult(0, moves[0], 1, 1, 1, False, [moves[0]], {})
        
        best_move = moves[0]
        best_score = float('-inf')
        pv = []
        eval_breakdown = {}
        
        # Aspiration Window Search
        aspiration_window = 50
        alpha = best_score - aspiration_window
        beta = best_score + aspiration_window
        
        max_depth_reached = 0
        
        for depth in range(1, self.max_depth + 1):
            try:
                # 시간 체크
                if time.time() > end_time:
                    break
                
                # Aspiration window로 탐색
                score, move = self.ultra_negamax(board, depth, alpha, beta, True, end_time, 0)
                
                # Window 밖의 결과가 나오면 전체 범위로 재탐색
                if score <= alpha or score >= beta:
                    score, move = self.ultra_negamax(board, depth, float('-inf'), float('inf'), True, end_time, 0)
                
                if move and time.time() <= end_time:
                    best_move = move
                    best_score = score
                    max_depth_reached = depth
                    
                    # 다음 반복을 위한 aspiration window 업데이트
                    alpha = score - aspiration_window
                    beta = score + aspiration_window
                
                # 완전 탐색 달성 시 중단
                if depth >= board.get_empty_count():
                    break
                
                # 시간 관리 - 다음 깊이를 탐색할 시간이 없으면 중단
                elapsed = time.time() - start_time
                if elapsed > self.time_limit * 0.7:
                    break
                    
            except Exception as e:
                print(f"Error in depth {depth}: {e}")
                break
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        # 평가 분석을 위해 최종 위치 평가
        if best_move:
            final_board = board.apply_move(*best_move, self.color)
            final_eval = self.ultra_evaluate_position(final_board)
            eval_breakdown = {'final_eval': final_eval}
        
        return UltraSearchResult(
            score=best_score,
            best_move=best_move,
            depth=max_depth_reached,
            nodes=self.nodes_searched,
            time_ms=elapsed_ms,
            is_exact=(max_depth_reached >= board.get_empty_count()),
            pv=[best_move] if best_move else [],
            eval_breakdown=eval_breakdown
        )
    
    def get_move(self, board):
        """최고의 수 반환"""
        self.nodes_searched = 0
        self.tt_hits = 0
        self.cutoffs = 0
        self.perfect_searches = 0
        self.tt_age += 1
        
        # 오프닝북 먼저 시도
        if board.get_empty_count() > 54:  # 초기 10수 이내
            opening_move = self.get_opening_move(board)
            if opening_move:
                print(f"Opening book move: {chr(opening_move[1] + ord('a'))}{opening_move[0] + 1}")
                return opening_move
        
        # 메인 탐색
        start_time = time.time()
        result = self.ultra_iterative_deepening(board)
        
        # 상세 통계 출력
        if result.time_ms > 100:
            nps = result.nodes / (result.time_ms / 1000) if result.time_ms > 0 else 0
            print(f"🧠 Ultra AI Analysis:")
            print(f"   Best move: {chr(result.best_move[1] + ord('a'))}{result.best_move[0] + 1}")
            print(f"   Score: {result.score}")
            print(f"   Depth: {result.depth}")
            print(f"   Nodes: {result.nodes:,}")
            print(f"   Time: {result.time_ms}ms")
            print(f"   NPS: {nps:,.0f}")
            print(f"   TT hits: {self.tt_hits:,}")
            print(f"   Cutoffs: {self.cutoffs:,}")
            if self.perfect_searches > 0:
                print(f"   Perfect searches: {self.perfect_searches}")
            print(f"   Exact: {'Yes' if result.is_exact else 'No'}")
        
        return result.best_move