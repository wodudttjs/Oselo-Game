import time
import random
import hashlib
import math
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set
import threading
import json
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
from othello_net import OthelloNet, AlphaZeroMCTS

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('othello_ai.log')
    ]
)
logger = logging.getLogger('OthelloAI')

# Constants
BLACK = 1
WHITE = 2
EMPTY = 0

CORNERS = [(0, 0), (0, 7), (7, 0), (7, 7)]
X_SQUARES = [(1, 1), (1, 6), (6, 1), (6, 6)]
C_SQUARES = [(0, 1), (1, 0), (0, 6), (1, 7), (6, 0), (7, 1), (6, 7), (7, 6)]

def opponent(color):
    return WHITE if color == BLACK else BLACK

class Board:
    """오델로 보드 클래스"""
    
    def __init__(self):
        self.board = [[EMPTY for _ in range(8)] for _ in range(8)]
        self.board[3][3] = WHITE
        self.board[3][4] = BLACK
        self.board[4][3] = BLACK
        self.board[4][4] = WHITE
        self.move_history = []
    
    def copy(self):
        """보드 복사"""
        new_board = Board()
        new_board.board = [row[:] for row in self.board]
        new_board.move_history = self.move_history[:]
        return new_board
    
    def is_valid_move(self, x, y, color):
        """유효한 수인지 확인"""
        if not (0 <= x < 8 and 0 <= y < 8) or self.board[x][y] != EMPTY:
            return False
        
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        for dx, dy in directions:
            if self._check_direction(x, y, dx, dy, color):
                return True
        return False
    
    def _check_direction(self, x, y, dx, dy, color):
        """특정 방향으로 뒤집을 수 있는지 확인"""
        nx, ny = x + dx, y + dy
        if not (0 <= nx < 8 and 0 <= ny < 8) or self.board[nx][ny] != opponent(color):
            return False
        
        while 0 <= nx < 8 and 0 <= ny < 8:
            if self.board[nx][ny] == EMPTY:
                return False
            if self.board[nx][ny] == color:
                return True
            nx += dx
            ny += dy
        return False
    
    def get_valid_moves(self, color):
        """유효한 수 목록 반환"""
        moves = []
        for x in range(8):
            for y in range(8):
                if self.is_valid_move(x, y, color):
                    moves.append((x, y))
        return moves
    
    def apply_move(self, x, y, color):
        """수를 두고 새로운 보드 반환"""
        new_board = self.copy()
        if not new_board.is_valid_move(x, y, color):
            return new_board
        
        new_board.board[x][y] = color
        flipped = []
        
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        for dx, dy in directions:
            line_flipped = new_board._flip_direction(x, y, dx, dy, color)
            flipped.extend(line_flipped)
        
        new_board.move_history.append((x, y, color, flipped))
        return new_board
    
    def _flip_direction(self, x, y, dx, dy, color):
        """특정 방향의 돌들을 뒤집기"""
        flipped = []
        nx, ny = x + dx, y + dy
        
        while (0 <= nx < 8 and 0 <= ny < 8 and 
               self.board[nx][ny] == opponent(color)):
            flipped.append((nx, ny))
            nx += dx
            ny += dy
        
        if (0 <= nx < 8 and 0 <= ny < 8 and 
            self.board[nx][ny] == color and flipped):
            for fx, fy in flipped:
                self.board[fx][fy] = color
            return flipped
        return []
    
    def count_stones(self):
        """돌 개수 세기"""
        black_count = sum(row.count(BLACK) for row in self.board)
        white_count = sum(row.count(WHITE) for row in self.board)
        return black_count, white_count
    
    def get_empty_count(self):
        """빈 칸 개수"""
        return sum(row.count(EMPTY) for row in self.board)
    
    def get_frontier_count(self, color):
        """프론티어 디스크 개수 (인접한 빈 칸이 있는 돌)"""
        count = 0
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        for x in range(8):
            for y in range(8):
                if self.board[x][y] == color:
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < 8 and 0 <= ny < 8 and 
                            self.board[nx][ny] == EMPTY):
                            count += 1
                            break
        return count

@dataclass
class UltraSearchResult:
    """Ultra Search Result with detailed analysis"""
    score: int
    best_move: Optional[Tuple[int, int]]
    depth: int
    nodes: int
    time_ms: int
    is_exact: bool
    pv: List[Tuple[int, int]]
    eval_breakdown: Dict[str, float]

class SearchTimeoutException(Exception):
    """탐색 시간 초과 예외"""
    pass

class UltraStrongAI:
    """최강 오델로 AI - 이기는 것이 목표"""
    
    def __init__(self, color, difficulty='ultra', time_limit=10.0, use_neural_net=False):
        self.color = color
        self.difficulty = difficulty
        self.time_limit = time_limit
        
        # 통계
        self.nodes_searched = 0
        self.tt_hits = 0
        self.cutoffs = 0
        self.perfect_searches = 0
        self.multi_cut_prunes = 0  # ← 여기 추가

        # Multi-Cut 설정 ← 여기 추가
        self.multi_cut_depth = 3
        self.multi_cut_margin = 50
        self.multi_cut_attempts = 3

        # 신경망 관련 추가
        self.use_neural_net = use_neural_net
        self.neural_net = None
        self.mcts = None
        self.training_mode = False

        if use_neural_net and TORCH_AVAILABLE:
            try:
                from othello_net import OthelloNet, AlphaZeroMCTS
                self.neural_net = OthelloNet()
                self.mcts = AlphaZeroMCTS(self.neural_net, num_simulations=800)
                self.load_model()
            except ImportError as e:
                logger.warning(f"신경망 모듈 로드 실패: {e}")
                self.use_neural_net = False
        elif use_neural_net and not TORCH_AVAILABLE:
            logger.warning("PyTorch가 설치되지 않음. 신경망 기능 비활성화")
            self.use_neural_net = False

        # 하이브리드 평가 가중치
        self.hybrid_weights = {
            'neural': 0.7,    # 신경망 비중
            'heuristic': 0.3  # 기존 휴리스틱 비중
        }

        # 극강 설정
        if difficulty == 'ultra':
            self.max_depth = 20  # 증가
            self.endgame_depth = 64
            self.use_perfect_endgame = True
            self.endgame_threshold = 16
        elif difficulty == 'hard':
            self.max_depth = 16
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
        self.max_tt_size = 1000000
        
        # 완벽한 오프닝북
        self.opening_book = self.create_perfect_opening_book()
        
        # 고급 휴리스틱들
        self.killer_moves = defaultdict(list)
        self.history_table = defaultdict(int)
        self.counter_moves = defaultdict(list)
        
        # 패턴 평가 시스템
        self.pattern_values = self.initialize_patterns()
        
        # 시간 관리
        self.start_time = 0
        self.end_time = 0
        
        # 게임 단계별 최적화된 가중치
        self.stage_weights = {
            'opening': {
                'mobility': 1.0, 'corners': 3.0, 'edges': 0.3, 'stability': 0.1,
                'discs': 0.0, 'frontier': -0.5, 'patterns': 0.8, 'parity': 0.1
            },
            'midgame': {
                'mobility': 0.8, 'corners': 2.0, 'edges': 0.8, 'stability': 1.2,
                'discs': 0.2, 'frontier': -0.3, 'patterns': 1.0, 'parity': 0.4
            },
            'endgame': {
                'mobility': 0.4, 'corners': 1.0, 'edges': 0.6, 'stability': 1.5,
                'discs': 2.0, 'frontier': -0.1, 'patterns': 0.5, 'parity': 1.0
            }
        }
        
        logger.info(f"AI 초기화 완료: color={color}, difficulty={difficulty}, time_limit={time_limit}")

    def load_model(self, model_path='models/best_model.pth'):
        """훈련된 모델 로드"""
        if not self.use_neural_net:
            return
        
        try:
            import torch
            checkpoint = torch.load(model_path, map_location='cpu')
            self.neural_net.load_state_dict(checkpoint['model_state_dict'])
            self.neural_net.eval()
            logger.info("신경망 모델 로드 완료")
        except FileNotFoundError:
            logger.warning("모델 파일을 찾을 수 없음. 랜덤 가중치 사용")
        except Exception as e:
            logger.error(f"모델 로드 중 오류: {e}")

    def get_move(self, board, use_neural_net=None):
        """신경망 사용 여부를 동적으로 결정"""
        if use_neural_net is None:
            use_neural_net = self.use_neural_net
            
        if use_neural_net and self.neural_net:
            return self.get_move_with_neural_net(board)
        else:
            return self.get_move_traditional(board)
    
    def get_move_with_neural_net(self, board):
        """신경망 기반 수 선택"""
        if not self.mcts:
            return self.get_move_traditional(board)
        
        # MCTS 탐색
        action_probs = self.mcts.search(board, self.color)
        
        # 온도 매개변수를 사용한 수 선택
        temperature = 0.1 if not self.training_mode else 1.0
        
        if temperature == 0:
            # 가장 높은 확률의 수 선택
            valid_moves = board.get_valid_moves(self.color)
            best_prob = 0
            best_move = valid_moves[0] if valid_moves else None
            
            for move in valid_moves:
                action_idx = move[0] * 8 + move[1]
                if action_probs[action_idx] > best_prob:
                    best_prob = action_probs[action_idx]
                    best_move = move
            
            return best_move
        else:
            # 확률적 선택
            return self.sample_move_from_probs(board, action_probs, temperature)
    
    def hybrid_evaluate(self, board):
        """하이브리드 평가 (기존 + 신경망)"""
        traditional_score = self.ultra_evaluate_position(board)
        
        if self.use_neural_net and self.neural_net:
            try:
                _, neural_value = self.mcts.neural_net_predict(board, self.color)
                neural_score = neural_value * 10000  # 스케일 조정
                
                # 게임 단계별 가중치 조정
                stage = self.get_game_stage(board)
                if stage == 'opening':
                    neural_weight = 0.3
                elif stage == 'midgame':
                    neural_weight = 0.6
                else:
                    neural_weight = 0.8
                
                final_score = (traditional_score * (1 - neural_weight) + 
                             neural_score * neural_weight)
                return int(final_score)
            except:
                return traditional_score
        
        return traditional_score
    
    def sample_move_from_probs(self, board, action_probs, temperature):
        """확률 분포에서 수 샘플링"""
        valid_moves = board.get_valid_moves(self.color)
        if not valid_moves:
            return None
        
        move_probs = []
        for move in valid_moves:
            action_idx = move[0] * 8 + move[1]
            prob = action_probs[action_idx] ** (1 / temperature)
            move_probs.append(prob)
        
        total_prob = sum(move_probs)
        if total_prob > 0:
            move_probs = [p / total_prob for p in move_probs]
            import numpy as np
            return np.random.choice(valid_moves, p=move_probs)
        else:
            import random
            return random.choice(valid_moves)

    
    def create_perfect_opening_book(self):
        """완벽한 오프닝북 생성"""
        return {
            # 표준 시작 후 최고의 수들
            ((3,3,'W'), (3,4,'B'), (4,3,'B'), (4,4,'W')): [
                ((2,3), 1.0),  # d3 - 가장 강력한 첫 수
                ((3,2), 1.0),  # c4 - 두 번째로 강력
                ((4,5), 0.7),  # f5 - 괜찮은 수
                ((5,4), 0.7)   # e6 - 괜찮은 수
            ]
        }
    
    def initialize_patterns(self):
        """패턴 기반 평가 시스템 초기화"""
        patterns = {}
        
        patterns['edge_patterns'] = {
            'perfect_edge': 500,
            'strong_edge': 200,
            'weak_edge': -100,
            'broken_edge': -300
        }
        
        patterns['corner_patterns'] = {
            'corner_captured': 1000,
            'corner_accessible': -500,
            'corner_safe': 300
        }
        
        return patterns
    
    def check_time_limit(self):
        """시간 제한 확인"""
        if time.time() > self.end_time:
            raise SearchTimeoutException("시간 초과")
    
    def get_game_stage(self, board):
        """현재 게임 단계 정확히 판단"""
        moves_played = 64 - board.get_empty_count() - 4
        
        if moves_played <= 20:
            return 'opening'
        elif moves_played <= 45:
            return 'midgame'
        else:
            return 'endgame'
    
    
    def ultra_evaluate_position(self, board):
        """극강 위치 평가 함수"""
        if board.get_empty_count() == 0:
            b, w = board.count_stones()
            diff = (b - w) if self.color == BLACK else (w - b)  
            if diff > 0:
                return 50000 + diff
            elif diff < 0:
                return -50000 + diff
            else:
                return 0
        
        stage = self.get_game_stage(board)
        weights = self.stage_weights[stage]
        
        eval_breakdown = {}
        total_score = 0
        
        # 1. 기동력 (Mobility)
        my_moves = len(board.get_valid_moves(self.color))
        opp_moves = len(board.get_valid_moves(opponent(self.color)))
        
        if my_moves + opp_moves > 0:
            mobility_score = 100 * (my_moves - opp_moves) / (my_moves + opp_moves + 1)
        else:
            mobility_score = 0
            
        if my_moves > 0 and opp_moves == 0:
            mobility_score += 200
        elif my_moves == 0 and opp_moves > 0:
            mobility_score -= 200
            
        eval_breakdown['mobility'] = mobility_score
        total_score += weights['mobility'] * mobility_score
        
        # 2. 코너 제어
        corner_score = self.evaluate_corners_advanced(board)
        eval_breakdown['corners'] = corner_score
        total_score += weights['corners'] * corner_score
        
        # 3. 모서리 제어
        edge_score = self.evaluate_edges_advanced(board)
        eval_breakdown['edges'] = edge_score
        total_score += weights['edges'] * edge_score
        
        # 4. 안정성
        stability_score = self.evaluate_stability_advanced(board)
        eval_breakdown['stability'] = stability_score
        total_score += weights['stability'] * stability_score
        
        # 5. 돌 개수
        b, w = board.count_stones()
        disc_diff = (b - w) if self.color == BLACK else (w - b)
        eval_breakdown['discs'] = disc_diff
        total_score += weights['discs'] * disc_diff
        
        # 6. 프론티어 디스크
        my_frontier = board.get_frontier_count(self.color)
        opp_frontier = board.get_frontier_count(opponent(self.color))
        frontier_score = opp_frontier - my_frontier
        eval_breakdown['frontier'] = frontier_score
        total_score += weights['frontier'] * frontier_score
        
        # 7. 패턴 평가
        pattern_score = self.evaluate_patterns(board)
        eval_breakdown['patterns'] = pattern_score
        total_score += weights['patterns'] * pattern_score
        
        # 8. 패리티
        parity_score = self.evaluate_parity(board)
        eval_breakdown['parity'] = parity_score
        total_score += weights['parity'] * parity_score
        
        return int(total_score)
    
    def evaluate_corners_advanced(self, board):
        """고급 코너 평가"""
        score = 0
        my_corners = 0
        opp_corners = 0
        
        for corner_x, corner_y in CORNERS:
            if board.board[corner_x][corner_y] == self.color:
                my_corners += 1
                score += 300
                score += self.evaluate_corner_lines(board, corner_x, corner_y) * 50
            elif board.board[corner_x][corner_y] == opponent(self.color):
                opp_corners += 1
                score -= 300
                score -= self.evaluate_corner_lines(board, corner_x, corner_y) * 50
            else:
                score += self.evaluate_corner_danger(board, corner_x, corner_y)
        
        if my_corners > 1:
            score += 150 * my_corners * my_corners
        if opp_corners > 1:
            score -= 150 * opp_corners * opp_corners
            
        return score
    
    def evaluate_corner_lines(self, board, corner_x, corner_y):
        """코너에서 뻗어나가는 안정적인 라인 평가"""
        color = board.board[corner_x][corner_y]
        stable_count = 0
        
        directions = []
        if corner_x == 0:
            directions.append((1, 0))
        else:
            directions.append((-1, 0))
            
        if corner_y == 0:
            directions.append((0, 1))
        else:
            directions.append((0, -1))
        
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
        
        # X-square 체크
        x_squares = [(corner_x + 1 if corner_x == 0 else corner_x - 1,
                     corner_y + 1 if corner_y == 0 else corner_y - 1)]
        
        for x, y in x_squares:
            if 0 <= x < 8 and 0 <= y < 8:
                if board.board[x][y] == self.color:
                    score -= 200
                elif board.board[x][y] == opponent(self.color):
                    score += 200
        
        # C-square 체크
        c_squares = [
            (corner_x, corner_y + 1 if corner_y == 0 else corner_y - 1),
            (corner_x + 1 if corner_x == 0 else corner_x - 1, corner_y)
        ]
        
        for x, y in c_squares:
            if 0 <= x < 8 and 0 <= y < 8:
                if board.board[x][y] == self.color:
                    score -= 100
                elif board.board[x][y] == opponent(self.color):
                    score += 100
        
        return score
    
    def evaluate_edges_advanced(self, board):
        """고급 모서리 평가"""
        score = 0
        
        edges = [
            [(0, j) for j in range(8)],
            [(7, j) for j in range(8)],
            [(i, 0) for i in range(8)],
            [(i, 7) for i in range(8)]
        ]
        
        for edge in edges:
            my_count = sum(1 for x, y in edge if board.board[x][y] == self.color)
            opp_count = sum(1 for x, y in edge if board.board[x][y] == opponent(self.color))
            
            if my_count == 8:
                score += 400
            elif opp_count == 8:
                score -= 400
            else:
                score += (my_count - opp_count) * 15
                
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
            else:
                if current_my_seq > 0:
                    my_sequences += current_my_seq * current_my_seq
                    current_my_seq = 0
                if current_opp_seq > 0:
                    opp_sequences += current_opp_seq * current_opp_seq
                    current_opp_seq = 0
        
        if current_my_seq > 0:
            my_sequences += current_my_seq * current_my_seq
        if current_opp_seq > 0:
            opp_sequences += current_opp_seq * current_opp_seq
            
        return my_sequences - opp_sequences
    
    def evaluate_stability_advanced(self, board):
        """고급 안정성 평가"""
        my_stable = 0
        opp_stable = 0
        
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
        """진정한 안정성 검사"""
        color = board.board[x][y]
        if color == EMPTY:
            return False
        
        if (x, y) in CORNERS:
            return True
        
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        for dx, dy in directions:
            if not self.is_stable_in_direction(board, x, y, dx, dy, color):
                return False
        
        return True
    
    def is_stable_in_direction(self, board, x, y, dx, dy, color):
        """특정 방향에서의 안정성 검사"""
        nx, ny = x + dx, y + dy
        
        while 0 <= nx < 8 and 0 <= ny < 8:
            if board.board[nx][ny] != color:
                return False
            if (nx, ny) in CORNERS:
                return True
            if nx == 0 or nx == 7 or ny == 0 or ny == 7:
                return self.is_edge_stable(board, nx, ny, color)
            nx += dx
            ny += dy
        
        return True
    
    def is_edge_stable(self, board, x, y, color):
        """모서리의 안정성 검사"""
        if x == 0 or x == 7:
            for j in range(8):
                if board.board[x][j] != color and board.board[x][j] != EMPTY:
                    return False
        elif y == 0 or y == 7:
            for i in range(8):
                if board.board[i][y] != color and board.board[i][y] != EMPTY:
                    return False
        return True
    
    def evaluate_patterns(self, board):
        """패턴 기반 평가"""
        score = 0
        
        # 웨지 패턴 탐지
        score += self.detect_wedge_patterns(board)
        
        # 삼각형 패턴 탐지  
        score += self.detect_triangle_patterns(board)
        
        # 라인 패턴 탐지
        score += self.detect_line_patterns(board)
        
        return score
    
    def detect_wedge_patterns(self, board):
        """쐐기 패턴 탐지 - 코너을 향한 대각선 패턴"""
        score = 0
        
        for corner in CORNERS:
            cx, cy = corner
            # 코너에서 대각선으로 뻗어나가는 패턴 체크
            directions = []
            if cx == 0 and cy == 0:
                directions = [(1, 1)]
            elif cx == 0 and cy == 7:
                directions = [(1, -1)]
            elif cx == 7 and cy == 0:
                directions = [(-1, 1)]
            elif cx == 7 and cy == 7:
                directions = [(-1, -1)]
            
            for dx, dy in directions:
                my_count = 0
                opp_count = 0
                x, y = cx, cy
                
                for step in range(1, 8):
                    x += dx
                    y += dy
                    if not (0 <= x < 8 and 0 <= y < 8):
                        break
                    
                    if board.board[x][y] == self.color:
                        my_count += 1
                    elif board.board[x][y] == opponent(self.color):
                        opp_count += 1
                        break
                    else:
                        break
                
                if my_count >= 3:
                    score += my_count * 20
                if opp_count >= 3:
                    score -= opp_count * 20
        
        return score
    
    def detect_triangle_patterns(self, board):
        """삼각형 패턴 탐지 - 안정적인 삼각형 구조"""
        score = 0
        
        # 각 코너에서 L자 형태의 삼각형 패턴 체크
        for corner in CORNERS:
            cx, cy = corner
            if board.board[cx][cy] == self.color:
                # 코너에서 시작하는 L자 패턴 체크
                adjacent_cells = []
                if cx == 0:
                    adjacent_cells.append((1, cy))
                else:
                    adjacent_cells.append((6, cy))
                    
                if cy == 0:
                    adjacent_cells.append((cx, 1))
                else:
                    adjacent_cells.append((cx, 6))
                
                triangle_count = 1  # 코너 자체
                for ax, ay in adjacent_cells:
                    if (0 <= ax < 8 and 0 <= ay < 8 and 
                        board.board[ax][ay] == self.color):
                        triangle_count += 1
                
                if triangle_count >= 2:
                    score += triangle_count * 15
            elif board.board[cx][cy] == opponent(self.color):
                # 상대방 삼각형 페널티
                adjacent_cells = []
                if cx == 0:
                    adjacent_cells.append((1, cy))
                else:
                    adjacent_cells.append((6, cy))
                    
                if cy == 0:
                    adjacent_cells.append((cx, 1))
                else:
                    adjacent_cells.append((cx, 6))
                
                triangle_count = 1
                for ax, ay in adjacent_cells:
                    if (0 <= ax < 8 and 0 <= ay < 8 and 
                        board.board[ax][ay] == opponent(self.color)):
                        triangle_count += 1
                
                if triangle_count >= 2:
                    score -= triangle_count * 15
        
        return score
        
    def detect_line_patterns(self, board):
        """라인 패턴 탐지 - 연속된 직선 패턴"""
        score = 0
        
        # 가로, 세로, 대각선 방향의 연속 패턴 체크
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for x in range(8):
            for y in range(8):
                for dx, dy in directions:
                    my_line = 0
                    opp_line = 0
                    
                    # 한 방향으로 연속된 돌 세기
                    nx, ny = x, y
                    while 0 <= nx < 8 and 0 <= ny < 8:
                        if board.board[nx][ny] == self.color:
                            my_line += 1
                        elif board.board[nx][ny] == opponent(self.color):
                            if my_line > 0:
                                break
                            opp_line += 1
                        else:
                            break
                        nx += dx
                        ny += dy
                    
                    # 긴 라인에 보너스
                    if my_line >= 4:
                        score += my_line * 10
                    if opp_line >= 4:
                        score -= opp_line * 10
        
        return score
    
    def evaluate_parity(self, board):
        """패리티 평가"""
        empty_count = board.get_empty_count()
        
        if empty_count % 2 == 1:
            return 50 if self.color == BLACK else -50
        else:
            return 50 if self.color == WHITE else -50
    
    def perfect_endgame_search(self, board, alpha, beta, player, passes=0):
        """완벽한 종료게임 탐색"""
        self.check_time_limit()
        self.perfect_searches += 1
        
        current_color = player
        moves = board.get_valid_moves(current_color)
        
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
                score, move = self.perfect_endgame_search(board, -beta, -alpha, 
                                                        opponent(current_color), passes + 1)
                return -score, None
        
        best_score = alpha
        best_move = None
        
        for move in moves:
            new_board = board.apply_move(*move, current_color)
            score, _ = self.perfect_endgame_search(new_board, -beta, -best_score, 
                                                 opponent(current_color), 0)
            score = -score
            
            if score > best_score:
                best_score = score
                best_move = move
                
            if best_score >= beta:
                break
        
        return best_score, best_move
    
    def ultra_negamax(self, board, depth, alpha, beta, player, passes=0):
        """울트라 강화된 네가맥스"""
        try:
            self.check_time_limit()
            self.nodes_searched += 1
            
            # 완벽한 종료게임 탐색
            empty_count = board.get_empty_count()
            if (self.use_perfect_endgame and 
                empty_count <= self.endgame_threshold and 
                depth >= empty_count):
                return self.perfect_endgame_search(board, alpha, beta, player, passes)
            
            # TT 조회
            board_hash = self.get_board_hash(board)
            tt_result = self.probe_tt(board_hash, depth, alpha, beta)
            if tt_result is not None:
                return tt_result, None
            
            current_color = player
            moves = board.get_valid_moves(current_color)


            # 터미널 조건
            if depth == 0:
                return self.ultra_evaluate_position(board), None
                
            if not moves:
                opponent_moves = board.get_valid_moves(opponent(current_color))
                if not opponent_moves:
                    # 게임 종료
                    return self.ultra_evaluate_position(board), None
                else:
                    # 패스
                    if passes >= 1:
                        return self.ultra_evaluate_position(board), None
                    score, move = self.ultra_negamax(board, depth, -beta, -alpha, 
                                                   opponent(current_color), passes + 1)
                    return -score, None
            # Multi-Cut Pruning
            if (depth >= self.multi_cut_depth and 
                len(moves) > self.multi_cut_attempts and
                not self.is_critical_position(board)):

                multi_cut_count = 0
                multi_cut_tested = 0

                # 처음 몇 개 수로 Multi-Cut 테스트
                for i, move in enumerate(moves[:self.multi_cut_attempts]):
                    new_board = board.apply_move(*move, current_color)

                    # 얕은 탐색으로 빠른 평가
                    test_score, _ = self.ultra_negamax(new_board, depth - 3, 
                                                     -beta, -alpha, opponent(current_color), 0)
                    test_score = -test_score
                    multi_cut_tested += 1

                    if test_score >= beta:
                        multi_cut_count += 1

                    # 충분한 수가 베타 컷오프를 보이면 전체 가지치기
                    if multi_cut_count >= 2 and multi_cut_tested >= 2:
                        self.multi_cut_prunes += 1
                        return beta, None
            
            # 울트라 강화된 무브 정렬
            ordered_moves = self.ultra_order_moves(board, moves, depth, current_color)
            best_move = None
            original_alpha = alpha
            best_score = alpha
            
            for i, move in enumerate(ordered_moves):
                new_board = board.apply_move(*move, current_color)
                
                # Late Move Reduction (LMR) - 더 보수적으로 적용
                reduction = 0
                if (i > 4 and depth > 4 and 
                    move not in self.killer_moves.get(depth, []) and
                    not self.is_tactical_move(board, move)):
                    reduction = min(2, depth // 4)  # 최대 2 감소
                
                # PVS (Principal Variation Search)
                if i == 0:
                    # 첫 번째 수는 전체 윈도우로 탐색
                    score, _ = self.ultra_negamax(new_board, depth - 1 - reduction, 
                                                -beta, -best_score, opponent(current_color), 0)
                else:
                    # null window 탐색
                    score, _ = self.ultra_negamax(new_board, depth - 1 - reduction, 
                                                -best_score - 1, -best_score, opponent(current_color), 0)
                    
                    # null window에서 좋은 결과가 나오면 전체 윈도우로 재탐색
                    if -score > best_score and -score < beta:
                        score, _ = self.ultra_negamax(new_board, depth - 1 - reduction, 
                                                    -beta, -score, opponent(current_color), 0)
                
                score = -score
                
                # LMR에서 좋은 결과가 나오면 전체 깊이로 재탐색
                if reduction > 0 and score > best_score:
                    score, _ = self.ultra_negamax(new_board, depth - 1, 
                                                -beta, -best_score, opponent(current_color), 0)
                    score = -score
                
                if score > best_score:
                    best_score = score
                    best_move = move
                
                if best_score >= beta:
                    # Beta cutoff
                    self.cutoffs += 1
                    self.update_killer_moves(depth, move)
                    break
            
            # 히스토리 테이블 업데이트
            if best_move:
                self.history_table[best_move] += depth * depth
            
            # TT 저장
            flag = 'EXACT' if original_alpha < best_score < beta else ('BETA' if best_score >= beta else 'ALPHA')
            self.store_tt(board_hash, depth, best_score, flag, best_move)
            
            return best_score, best_move
            
        except SearchTimeoutException:
            # 시간 초과 시 현재까지의 평가값 반환
            return self.ultra_evaluate_position(board), None
    
    def is_critical_position(self, board):
        """중요한 포지션인지 판단 (기존 코드에 추가)"""
        # 코너 근처 수가 있는지
        moves = board.get_valid_moves(self.color)
        for move in moves:
            if self.is_near_corner(move):
                return True
        
        # Mobility가 매우 제한적인지
        my_moves = len(board.get_valid_moves(self.color))
        opp_moves = len(board.get_valid_moves(opponent(self.color)))
        
        if my_moves <= 2 or abs(my_moves - opp_moves) >= 5:
            return True
        
        # 게임 후반부
        if board.get_empty_count() <= 20:
            return True
        
        return False

    def is_near_corner(self, move):
        """코너 근처 수인지 판단"""
        x, y = move
        for corner in CORNERS:
            if abs(x - corner[0]) <= 2 and abs(y - corner[1]) <= 2:
                return True
        return False
    
    def is_tactical_move(self, board, move):
        """전술적 수인지 판단"""
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
    # ← 아래 메서드들이 누락됨
    def mtdf(self, board, guess, depth, player):
        """MTD(f) - Memory-enhanced Test Driver"""
        try:
            g = guess
            upper_bound = float('inf')
            lower_bound = float('-inf')
            
            while lower_bound < upper_bound:
                self.check_time_limit()
                
                if g == lower_bound:
                    beta = g + 1
                else:
                    beta = g
                
                g, _ = self.ultra_negamax(board, depth, beta - 1, beta, player, 0)
                
                if g < beta:
                    upper_bound = g
                else:
                    lower_bound = g
                    
            return g
            
        except SearchTimeoutException:
            return guess

    def get_best_move_from_tt(self, board, depth):
        """TT에서 현재 보드의 최고 수 가져오기"""
        board_hash = self.get_board_hash(board)
        if board_hash in self.tt:
            entry = self.tt[board_hash]
            if entry.get('depth', 0) >= depth and entry.get('best_move'):
                return entry['best_move']
        
        # TT에 없으면 첫 번째 유효한 수 반환
        moves = board.get_valid_moves(self.color)
        return moves[0] if moves else None

    def ultra_iterative_deepening_mtdf(self, board):
        """MTD(f)를 사용한 반복 심화"""
        self.start_time = time.time()
        self.end_time = self.start_time + self.time_limit
        
        logger.info(f"MTD(f) 탐색 시작 - 시간 제한: {self.time_limit}초")
        
        moves = board.get_valid_moves(self.color)
        if not moves:
            return UltraSearchResult(0, None, 0, 0, 0, True, [], {})
        
        if len(moves) == 1:
            return UltraSearchResult(0, moves[0], 1, 1, 1, False, [moves[0]], {})
        
        # 초기 추정값
        guess = self.ultra_evaluate_position(board)
        best_move = moves[0]
        max_depth_reached = 0
        
        try:
            for depth in range(1, self.max_depth + 1):
                if time.time() > self.end_time:
                    break
                    
                # MTD(f)로 탐색
                score = self.mtdf(board, guess, depth, self.color)
                
                # 최고 수 찾기 (root에서만)
                move = self.get_best_move_from_tt(board, depth)
                if move:
                    best_move = move
                    max_depth_reached = depth
                    guess = score  # 다음 반복의 초기값
                    
                    logger.info(f"MTD(f) 깊이 {depth} 완료: score={score}, move={best_move}")
                
                if time.time() > self.start_time + self.time_limit * 0.85:
                    break
                    
        except Exception as e:
            logger.error(f"MTD(f) 탐색 중 오류: {e}")
        
        elapsed_ms = int((time.time() - self.start_time) * 1000)
        
        return UltraSearchResult(
            score=guess,
            best_move=best_move,
            depth=max_depth_reached,
            nodes=self.nodes_searched,
            time_ms=elapsed_ms,
            is_exact=False,
            pv=self.extract_pv(board, best_move, max_depth_reached),
            eval_breakdown={}
        )
    
    def ultra_order_moves(self, board, moves, depth, current_color):
        """울트라 강화된 무브 정렬"""
        if not moves:
            return moves
        
        move_scores = []
        board_hash = self.get_board_hash(board)
        
        # TT에서 최고 수 가져오기
        tt_move = None
        if board_hash in self.tt:
            tt_move = self.tt[board_hash].get('best_move')
        
        for move in moves:
            x, y = move
            score = 0
            
            # TT 수 최우선
            if move == tt_move:
                score += 50000
            
            # 킬러 무브
            if move in self.killer_moves.get(depth, []):
                score += 10000
            
            # 카운터 무브
            if hasattr(self, 'last_opponent_move') and self.last_opponent_move:
                if move in self.counter_moves.get(self.last_opponent_move, []):
                    score += 5000
            
            # 히스토리 휴리스틱
            score += self.history_table.get(move, 0)
            
            # 위치별 전략적 가치
            position_score = self.evaluate_move_position(board, move)
            score += position_score
            
            # 이 수로 인한 mobility 변화
            mobility_score = self.evaluate_move_mobility(board, move, current_color)
            score += mobility_score
            
            # 안정성 변화
            stability_score = self.evaluate_move_stability(board, move, current_color)
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
            adjacent_corner_empty = False
            for corner in CORNERS:
                if abs(corner[0] - x) <= 1 and abs(corner[1] - y) <= 1:
                    if board.board[corner[0]][corner[1]] == EMPTY:
                        adjacent_corner_empty = True
                        break
            if adjacent_corner_empty:
                score -= 500
            else:
                score += 100
        
        # C-squares  
        elif (x, y) in C_SQUARES:
            score -= 200
        
        # 모서리
        elif x == 0 or x == 7 or y == 0 or y == 7:
            score += 200
        
        # 내부 위치들
        else:
            center_distance = abs(x - 3.5) + abs(y - 3.5)
            score += int((7 - center_distance) * 10)
        
        return score
    
    def evaluate_move_mobility(self, board, move, current_color):
        """수에 따른 mobility 변화 평가"""
        current_my_moves = len(board.get_valid_moves(current_color))
        current_opp_moves = len(board.get_valid_moves(opponent(current_color)))
        
        new_board = board.apply_move(*move, current_color)
        new_my_moves = len(new_board.get_valid_moves(current_color))
        new_opp_moves = len(new_board.get_valid_moves(opponent(current_color)))
        
        my_mobility_change = new_my_moves - current_my_moves
        opp_mobility_change = new_opp_moves - current_opp_moves
        
        return (current_opp_moves - new_opp_moves) * 20 + my_mobility_change * 10
    
    def evaluate_move_stability(self, board, move, current_color):
        """수에 따른 안정성 변화 평가"""
        new_board = board.apply_move(*move, current_color)
        
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
        if len(self.tt) < self.max_tt_size * 0.8:
            return
            
        # 나이와 깊이를 고려한 우선순위로 삭제
        entries_to_remove = []
        for key, entry in list(self.tt.items()):
            age_score = self.tt_age - entry['age']
            depth_score = entry['depth']
            priority = age_score - depth_score  # 오래되고 얕은 것부터 삭제
            
            entries_to_remove.append((priority, key))
        
        entries_to_remove.sort(reverse=True)
        
        # 절반 정도 삭제
        for i in range(min(len(entries_to_remove) // 2, len(self.tt) // 4)):
            del self.tt[entries_to_remove[i][1]]
    
    def get_opening_move(self, board):
        """오프닝북에서 수 선택"""
        board_state = self.board_to_tuple(board)
        
        if board_state in self.opening_book:
            moves = self.opening_book[board_state]
            good_moves = [(move, weight) for move, weight in moves if weight >= 0.8]
            if good_moves:
                weights = [weight for _, weight in good_moves]
                total_weight = sum(weights)
                if total_weight > 0:
                    r = random.random() * total_weight
                    
                    cumulative = 0
                    for move, weight in good_moves:
                        cumulative += weight
                        if r <= cumulative:
                            if board.is_valid_move(*move, self.color):
                                return move
        
        return None
    
    def board_to_tuple(self, board):
        """보드를 튜플로 변환 (오프닝북용)"""
        state = []
        for i in range(8):
            for j in range(8):
                if board.board[i][j] != EMPTY:
                    color = 'B' if board.board[i][j] == BLACK else 'W'
                    state.append((i, j, color))
        return tuple(sorted(state))
    
    def ultra_iterative_deepening(self, board):
        """울트라 강화된 반복 심화"""
        self.start_time = time.time()
        self.end_time = self.start_time + self.time_limit
        
        logger.info(f"탐색 시작 - 시간 제한: {self.time_limit}초")
        
        moves = board.get_valid_moves(self.color)
        if not moves:
            logger.info("가능한 수가 없음")
            return UltraSearchResult(0, None, 0, 0, 0, True, [], {})
        
        if len(moves) == 1:
            logger.info(f"유일한 수: {moves[0]}")
            return UltraSearchResult(0, moves[0], 1, 1, 1, False, [moves[0]], {})
        
        best_move = moves[0]
        best_score = float('-inf')
        pv = []
        eval_breakdown = {}
        
        # Aspiration Window Search - 더 보수적으로 설정
        aspiration_window = 100
        alpha = best_score - aspiration_window
        beta = best_score + aspiration_window
        
        max_depth_reached = 0
        
        try:
            for depth in range(1, self.max_depth + 1):
                depth_start_time = time.time()
                logger.debug(f"깊이 {depth} 탐색 시작")
                
                if time.time() > self.end_time:
                    logger.info(f"시간 초과로 깊이 {depth}에서 중단")
                    break
                
                # Aspiration window로 탐색
                try:
                    score, move = self.ultra_negamax(board, depth, alpha, beta, self.color, 0)
                    
                    # Window 밖의 결과가 나오면 전체 범위로 재탐색
                    if score <= alpha:
                        logger.debug(f"깊이 {depth}: alpha cutoff, 재탐색")
                        alpha = float('-inf')
                        score, move = self.ultra_negamax(board, depth, alpha, beta, self.color, 0)
                    elif score >= beta:
                        logger.debug(f"깊이 {depth}: beta cutoff, 재탐색")
                        beta = float('inf')
                        score, move = self.ultra_negamax(board, depth, alpha, beta, self.color, 0)
                    
                    if move and time.time() <= self.end_time:
                        best_move = move
                        best_score = score
                        max_depth_reached = depth
                        
                        # Principal Variation 수집
                        pv = self.extract_pv(board, best_move, depth)
                        
                        # 다음 반복을 위한 aspiration window 업데이트
                        alpha = score - aspiration_window
                        beta = score + aspiration_window
                        
                        depth_time = time.time() - depth_start_time
                        logger.info(f"깊이 {depth} 완료: score={score}, move={best_move}, time={depth_time:.3f}s")
                    
                    # 완전 탐색 달성 시 중단
                    if depth >= board.get_empty_count():
                        logger.info(f"완전 탐색 달성 (깊이 {depth})")
                        break
                    
                    # 시간 관리 - 더 보수적으로
                    elapsed = time.time() - self.start_time
                    if elapsed > self.time_limit * 0.85:  # 85%에서 중단
                        logger.info(f"시간 제한 85% 도달, 탐색 중단")
                        break
                        
                except SearchTimeoutException:
                    logger.info(f"깊이 {depth}에서 시간 초과")
                    break
                    
        except Exception as e:
            logger.error(f"탐색 중 오류 발생: {e}")
        
        elapsed_ms = int((time.time() - self.start_time) * 1000)
        
        # 평가 분석
        if best_move:
            final_board = board.apply_move(*best_move, self.color)
            final_eval = self.ultra_evaluate_position(final_board)
            eval_breakdown = {'final_eval': final_eval}
        
        logger.info(f"탐색 완료: depth={max_depth_reached}, nodes={self.nodes_searched}, time={elapsed_ms}ms")
        
        return UltraSearchResult(
            score=best_score,
            best_move=best_move,
            depth=max_depth_reached,
            nodes=self.nodes_searched,
            time_ms=elapsed_ms,
            is_exact=(max_depth_reached >= board.get_empty_count()),
            pv=pv,
            eval_breakdown=eval_breakdown
        )
    
    def extract_pv(self, board, first_move, max_depth):
        """Principal Variation 추출"""
        pv = [first_move]
        current_board = board.apply_move(*first_move, self.color)
        current_player = opponent(self.color)
        depth = 1
        
        while depth < max_depth and depth < 6:  # PV는 너무 길지 않게
            board_hash = self.get_board_hash(current_board)
            if board_hash in self.tt and self.tt[board_hash].get('best_move'):
                next_move = self.tt[board_hash]['best_move']
                if current_board.is_valid_move(*next_move, current_player):
                    pv.append(next_move)
                    current_board = current_board.apply_move(*next_move, current_player)
                    current_player = opponent(current_player)
                    depth += 1
                else:
                    break
            else:
                break
        
        return pv
    
    def get_move(self, board, use_mtdf=True):
        """최고의 수 반환"""
        self.nodes_searched = 0
        self.tt_hits = 0
        self.cutoffs = 0
        self.perfect_searches = 0
        self.multi_cut_prunes = 0
        self.tt_age += 1
        
        logger.info(f"AI 수 계산 시작 - 색깔: {'Black' if self.color == BLACK else 'White'}")
        
        # 오프닝북 먼저 시도
        if board.get_empty_count() > 54:
            opening_move = self.get_opening_move(board)
            if opening_move:
                logger.info(f"오프닝북 수 선택: {chr(opening_move[1] + ord('a'))}{opening_move[0] + 1}")
                return opening_move
        
        # MTD(f) 또는 기존 방식 선택
        if use_mtdf:
            result = self.ultra_iterative_deepening_mtdf(board)
            search_type = "MTD(f)"
        else:
            result = self.ultra_iterative_deepening(board)
            search_type = "Alpha-Beta"
        
        
        # 상세 통계 출력
        if result.time_ms > 100:
            nps = result.nodes / (result.time_ms / 1000) if result.time_ms > 0 else 0
            
            # 콘솔 출력 (간단하게)
            print(f"🧠 Ultra AI Analysis:")
            print(f"   Best move: {chr(result.best_move[1] + ord('a'))}{result.best_move[0] + 1}")
            print(f"   Score: {result.score}")
            print(f"   Depth: {result.depth}")
            print(f"   Time: {result.time_ms}ms")
            
            # 로그 출력 (상세하게)
            logger.info(f"탐색 결과 ({search_type}):")  # ← search_type 추가
            logger.info(f"  최고 수: {chr(result.best_move[1] + ord('a'))}{result.best_move[0] + 1}")
            logger.info(f"  점수: {result.score}")
            logger.info(f"  깊이: {result.depth}")
            logger.info(f"  노드: {result.nodes:,}")
            logger.info(f"  시간: {result.time_ms}ms")
            logger.info(f"  NPS: {nps:,.0f}")
            logger.info(f"  TT 히트: {self.tt_hits:,}")
            logger.info(f"  컷오프: {self.cutoffs:,}")
            logger.info(f"  Multi-Cut 가지치기: {self.multi_cut_prunes:,}")  # 추가
            if self.perfect_searches > 0:
                logger.info(f"  완벽 탐색: {self.perfect_searches}")
            logger.info(f"  정확성: {'예' if result.is_exact else '아니오'}")
            if result.pv and len(result.pv) > 1:
                pv_str = " ".join([f"{chr(move[1] + ord('a'))}{move[0] + 1}" for move in result.pv[:5]])
                logger.info(f"  주변이: {pv_str}")
            logger.info(f"  평가 분석: {result.eval_breakdown}")
            logger.info('' + '-' * 50)
        
        return result.best_move
    def get_move_traditional(self, board):
        """기존 방식의 수 선택 (신경망 없이)"""
        self.nodes_searched = 0
        self.tt_hits = 0
        self.cutoffs = 0
        self.perfect_searches = 0
        self.multi_cut_prunes = 0
        self.tt_age += 1
    
        logger.info(f"AI 수 계산 시작 - 색깔: {'Black' if self.color == BLACK else 'White'}")
    
        # 오프닝북 먼저 시도
        if board.get_empty_count() > 54:
            opening_move = self.get_opening_move(board)
            if opening_move:
                logger.info(f"오프닝북 수 선택: {chr(opening_move[1] + ord('a'))}{opening_move[0] + 1}")
                return opening_move
    
        # 기존 Alpha-Beta 방식 사용
        result = self.ultra_iterative_deepening(board)
        
        # 상세 통계 출력
        if result.time_ms > 100:
            nps = result.nodes / (result.time_ms / 1000) if result.time_ms > 0 else 0
            
            print(f"🧠 Ultra AI Analysis:")
            print(f" Best move: {chr(result.best_move[1] + ord('a'))}{result.best_move[0] + 1}")
            print(f" Score: {result.score}")
            print(f" Depth: {result.depth}")
            print(f" Time: {result.time_ms}ms")
    
            logger.info(f"탐색 결과 (Alpha-Beta):")
            logger.info(f" 최고 수: {chr(result.best_move[1] + ord('a'))}{result.best_move[0] + 1}")
            logger.info(f" 점수: {result.score}")
            logger.info(f" 깊이: {result.depth}")
            logger.info(f" 노드: {result.nodes:,}")
            logger.info(f" 시간: {result.time_ms}ms")
            logger.info(f" NPS: {nps:,.0f}")
            logger.info(f" TT 히트: {self.tt_hits:,}")
            logger.info(f" 컷오프: {self.cutoffs:,}")
            logger.info(f" Multi-Cut 가지치기: {self.multi_cut_prunes:,}")
    
            if self.perfect_searches > 0:
                logger.info(f" 완벽 탐색: {self.perfect_searches}")
            logger.info(f" 정확성: {'예' if result.is_exact else '아니오'}")
    
            if result.pv and len(result.pv) > 1:
                pv_str = " ".join([f"{chr(move[1] + ord('a'))}{move[0] + 1}" for move in result.pv[:5]])
                logger.info(f" 주변이: {pv_str}")
    
            logger.info(f" 평가 분석: {result.eval_breakdown}")
            logger.info('-' * 50)
    
# 사용 예시
def demo_game():
    """데모 게임"""
    board = Board()
    
    # AI 생성 (흑: Ultra, 백: Hard)
    black_ai = UltraStrongAI(BLACK, difficulty='ultra', time_limit=5.0)
    white_ai = UltraStrongAI(WHITE, difficulty='hard', time_limit=3.0)
    
    current_player = BLACK
    pass_count = 0
    
    logger.info("게임 시작")
    print("🎮 Ultra Strong Othello AI Demo")
    print("=" * 50)
    
    while pass_count < 2:
        moves = board.get_valid_moves(current_player)
        
        if not moves:
            logger.info(f"{'Black' if current_player == BLACK else 'White'} 패스")
            print(f"{'Black' if current_player == BLACK else 'White'} passes")
            pass_count += 1
            current_player = opponent(current_player)
            continue
        
        pass_count = 0
        
        # AI 수 선택
        if current_player == BLACK:
            move = black_ai.get_move(board, use_mtdf=True)  # MTD(f) 사용
            player_name = "Black (Ultra)"
        else:
            move = white_ai.get_move(board, use_mtdf=False)  # 기존 방식
            player_name = "White (Hard)"
        
        if move:
            board = board.apply_move(*move, current_player)
            move_str = f"{chr(move[1] + ord('a'))}{move[0] + 1}"
            logger.info(f"{player_name} 수: {move_str}")
            print(f"{player_name} plays: {move_str}")
            
            # 보드 상태 출력 (간단히)
            b, w = board.count_stones()
            print(f"Score - Black: {b}, White: {w}")
            print("-" * 30)
        
        current_player = opponent(current_player)
    
    # 최종 결과
    b, w = board.count_stones()
    logger.info(f"게임 종료 - Black: {b}, White: {w}")
    print("\n🏆 Game Over!")
    print(f"Final Score - Black: {b}, White: {w}")
    if b > w:
        print("Black (Ultra AI) Wins!")
        logger.info("Black (Ultra AI) 승리!")
    elif w > b:
        print("White (Hard AI) Wins!")
        logger.info("White (Hard AI) 승리!")
    else:
        print("Draw!")
        logger.info("무승부!")

if __name__ == "__main__":
    demo_game()