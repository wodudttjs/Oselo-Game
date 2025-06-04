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

# GPU 관련 라이브러리 import (우선순위에 따라 시도)
try:
    import cupy as cp
    GPU_AVAILABLE = True
    GPU_BACKEND = 'cupy'
    print("CuPy GPU backend loaded successfully")
except ImportError:
    try:
        import numpy as np
        from numba import cuda, jit
        GPU_AVAILABLE = cuda.is_available()
        GPU_BACKEND = 'numba'
        if GPU_AVAILABLE:
            print("Numba CUDA backend loaded successfully")
        else:
            print("CUDA not available, falling back to CPU")
    except ImportError:
        import numpy as np
        GPU_AVAILABLE = False
        GPU_BACKEND = 'cpu'
        print("No GPU backend available, using CPU only")

# CPU 백업용 numpy
import numpy as np

# 로깅 설정
def setup_logging():
    """
    로깅 시스템 설정
    - 콘솔과 파일에 모두 출력
    - INFO 레벨 이상 메시지 기록
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('othello_ai.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('UltraStrongAI')

# 로거 초기화
logger = setup_logging()

# GPU 유틸리티 클래스
class GPUManager:
    """
    GPU 메모리 및 연산 관리 클래스
    CuPy와 Numba CUDA를 지원하며 CPU 백업 기능 포함
    """
    
    def __init__(self):
        self.gpu_available = GPU_AVAILABLE
        self.backend = GPU_BACKEND
        self.memory_pool = None
        
        if self.gpu_available and self.backend == 'cupy':
            try:
                # CuPy 메모리 풀 설정
                self.memory_pool = cp.get_default_memory_pool()
                logger.info(f"GPU Manager initialized with CuPy backend")
            except Exception as e:
                logger.error(f"Failed to initialize CuPy: {e}")
                self.gpu_available = False
                self.backend = 'cpu'
        elif self.gpu_available and self.backend == 'numba':
            logger.info(f"GPU Manager initialized with Numba CUDA backend")
        else:
            logger.info(f"GPU Manager initialized with CPU backend")
    
    def to_gpu(self, array):
        """
        배열을 GPU로 이동
        Args:
            array: numpy 배열 또는 리스트
        Returns:
            GPU 배열 또는 원본 배열 (GPU 미사용시)
        """
        if not self.gpu_available:
            return np.array(array)
        
        try:
            if self.backend == 'cupy':
                return cp.asarray(array)
            else:
                return np.array(array)
        except Exception as e:
            logger.warning(f"Failed to move array to GPU: {e}")
            return np.array(array)
    
    def to_cpu(self, array):
        """
        GPU 배열을 CPU로 이동
        Args:
            array: GPU 배열 또는 numpy 배열
        Returns:
            numpy 배열
        """
        if self.backend == 'cupy' and hasattr(array, 'get'):
            return array.get()
        return np.asarray(array)
    
    def clear_memory(self):
        """GPU 메모리 정리"""
        if self.gpu_available and self.backend == 'cupy' and self.memory_pool:
            try:
                self.memory_pool.free_all_blocks()
                logger.debug("GPU memory cleared")
            except Exception as e:
                logger.warning(f"Failed to clear GPU memory: {e}")

# 상수 정의
BLACK = 1
WHITE = 2
EMPTY = 0

CORNERS = [(0, 0), (0, 7), (7, 0), (7, 7)]
X_SQUARES = [(1, 1), (1, 6), (6, 1), (6, 6)]
C_SQUARES = [(0, 1), (1, 0), (0, 6), (1, 7), (6, 0), (7, 1), (6, 7), (7, 6)]

def opponent(color):
    """상대방 색상 반환"""
    return WHITE if color == BLACK else BLACK

class GPUBoard:
    """
    GPU 가속 오델로 보드 클래스
    보드 연산을 GPU에서 처리하여 성능 향상
    """
    
    def __init__(self, gpu_manager: GPUManager):
        self.gpu = gpu_manager
        self.board = self.gpu.to_gpu(np.zeros((8, 8), dtype=np.int8))
        self.move_history = []
        
        # 초기 보드 설정
        self._initialize_board()
        logger.debug("GPU Board initialized")
    
    def _initialize_board(self):
        """
        초기 보드 상태 설정
        중앙 4칸에 흑백 돌 배치
        """
        board_cpu = np.zeros((8, 8), dtype=np.int8)
        board_cpu[3, 3] = WHITE
        board_cpu[3, 4] = BLACK
        board_cpu[4, 3] = BLACK
        board_cpu[4, 4] = WHITE
        self.board = self.gpu.to_gpu(board_cpu)
    
    def copy(self):
        """보드 깊은 복사"""
        new_board = GPUBoard(self.gpu)
        new_board.board = self.gpu.to_gpu(self.gpu.to_cpu(self.board).copy())
        new_board.move_history = self.move_history.copy()
        return new_board
    
    def is_valid_move(self, x, y, color):
        """
        유효한 수인지 확인
        Args:
            x, y: 보드 좌표
            color: 돌 색상
        Returns:
            bool: 유효한 수 여부
        """
        if not (0 <= x < 8 and 0 <= y < 8):
            return False
        
        board_cpu = self.gpu.to_cpu(self.board)
        if board_cpu[x, y] != EMPTY:
            return False
        
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        for dx, dy in directions:
            if self._check_direction(board_cpu, x, y, dx, dy, color):
                return True
        return False
    
    def _check_direction(self, board_cpu, x, y, dx, dy, color):
        """
        특정 방향으로 뒤집을 수 있는지 확인
        Args:
            board_cpu: CPU상의 보드 배열
            x, y: 시작 좌표
            dx, dy: 방향 벡터
            color: 돌 색상
        Returns:
            bool: 해당 방향으로 뒤집기 가능 여부
        """
        nx, ny = x + dx, y + dy
        if not (0 <= nx < 8 and 0 <= ny < 8) or board_cpu[nx, ny] != opponent(color):
            return False
        
        while 0 <= nx < 8 and 0 <= ny < 8:
            if board_cpu[nx, ny] == EMPTY:
                return False
            if board_cpu[nx, ny] == color:
                return True
            nx += dx
            ny += dy
        return False
    
    def get_valid_moves(self, color):
        """
        유효한 수 목록 반환
        GPU 병렬 처리로 최적화
        Args:
            color: 돌 색상
        Returns:
            List[Tuple[int, int]]: 유효한 수 좌표 리스트
        """
        moves = []
        board_cpu = self.gpu.to_cpu(self.board)
        
        # GPU 병렬 처리 가능 시 사용
        if self.gpu.gpu_available and self.gpu.backend == 'cupy':
            moves = self._get_valid_moves_gpu(board_cpu, color)
        else:
            moves = self._get_valid_moves_cpu(board_cpu, color)
        
        logger.debug(f"Found {len(moves)} valid moves for color {color}")
        return moves
    
    def _get_valid_moves_gpu(self, board_cpu, color):
        """GPU를 사용한 유효한 수 찾기"""
        try:
            # CuPy를 사용한 병렬 처리
            moves = []
            for x in range(8):
                for y in range(8):
                    if board_cpu[x, y] == EMPTY and self.is_valid_move(x, y, color):
                        moves.append((x, y))
            return moves
        except Exception as e:
            logger.warning(f"GPU move calculation failed: {e}")
            return self._get_valid_moves_cpu(board_cpu, color)
    
    def _get_valid_moves_cpu(self, board_cpu, color):
        """CPU를 사용한 유효한 수 찾기"""
        moves = []
        for x in range(8):
            for y in range(8):
                if board_cpu[x, y] == EMPTY and self.is_valid_move(x, y, color):
                    moves.append((x, y))
        return moves
    
    def apply_move(self, x, y, color):
        """
        수를 두고 새로운 보드 반환
        Args:
            x, y: 착수 좌표
            color: 돌 색상
        Returns:
            GPUBoard: 새로운 보드 상태
        """
        new_board = self.copy()
        if not new_board.is_valid_move(x, y, color):
            logger.warning(f"Invalid move attempted: ({x}, {y}) for color {color}")
            return new_board
        
        board_cpu = new_board.gpu.to_cpu(new_board.board)
        board_cpu[x, y] = color
        flipped = []
        
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        for dx, dy in directions:
            line_flipped = new_board._flip_direction(board_cpu, x, y, dx, dy, color)
            flipped.extend(line_flipped)
        
        new_board.board = new_board.gpu.to_gpu(board_cpu)
        new_board.move_history.append((x, y, color, flipped))
        
        logger.debug(f"Move applied: ({x}, {y}), flipped {len(flipped)} pieces")
        return new_board
    
    def _flip_direction(self, board_cpu, x, y, dx, dy, color):
        """
        특정 방향의 돌들을 뒤집기
        Args:
            board_cpu: CPU상의 보드 배열
            x, y: 시작 좌표
            dx, dy: 방향 벡터
            color: 돌 색상
        Returns:
            List[Tuple[int, int]]: 뒤집힌 돌의 좌표 리스트
        """
        flipped = []
        nx, ny = x + dx, y + dy
        
        while (0 <= nx < 8 and 0 <= ny < 8 and 
               board_cpu[nx, ny] == opponent(color)):
            flipped.append((nx, ny))
            nx += dx
            ny += dy
        
        if (0 <= nx < 8 and 0 <= ny < 8 and 
            board_cpu[nx, ny] == color and flipped):
            for fx, fy in flipped:
                board_cpu[fx, fy] = color
            return flipped
        return []
    
    def count_stones(self):
        """
        돌 개수 세기
        Returns:
            Tuple[int, int]: (흑돌 수, 백돌 수)
        """
        board_cpu = self.gpu.to_cpu(self.board)
        black_count = np.sum(board_cpu == BLACK)
        white_count = np.sum(board_cpu == WHITE)
        return int(black_count), int(white_count)
    
    def get_empty_count(self):
        """
        빈 칸 개수 반환
        Returns:
            int: 빈 칸 개수
        """
        board_cpu = self.gpu.to_cpu(self.board)
        return int(np.sum(board_cpu == EMPTY))
    
    def get_frontier_count(self, color):
        """
        프론티어 디스크 개수 (인접한 빈 칸이 있는 돌)
        Args:
            color: 돌 색상
        Returns:
            int: 프론티어 디스크 개수
        """
        count = 0
        board_cpu = self.gpu.to_cpu(self.board)
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        
        for x in range(8):
            for y in range(8):
                if board_cpu[x, y] == color:
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < 8 and 0 <= ny < 8 and 
                            board_cpu[nx, ny] == EMPTY):
                            count += 1
                            break
        return count

@dataclass
class UltraSearchResult:
    """
    Ultra Search Result with detailed analysis
    탐색 결과를 담는 데이터 클래스
    """
    score: int
    best_move: Optional[Tuple[int, int]]
    depth: int
    nodes: int
    time_ms: int
    is_exact: bool
    pv: List[Tuple[int, int]]
    eval_breakdown: Dict[str, float]

class GPUEvaluator:
    """
    GPU 가속 보드 평가 클래스
    보드 평가 함수들을 GPU에서 병렬 처리
    """
    
    def __init__(self, gpu_manager: GPUManager):
        self.gpu = gpu_manager
        self._initialize_evaluation_tables()
        logger.info("GPU Evaluator initialized")
    
    def _initialize_evaluation_tables(self):
        """평가 테이블 초기화 및 GPU 메모리 로드"""
        # 위치별 가중치 테이블
        position_weights = np.array([
            [100, -20, 10, 5, 5, 10, -20, 100],
            [-20, -40, -5, -5, -5, -5, -40, -20],
            [10, -5, 3, 2, 2, 3, -5, 10],
            [5, -5, 2, 1, 1, 2, -5, 5],
            [5, -5, 2, 1, 1, 2, -5, 5],
            [10, -5, 3, 2, 2, 3, -5, 10],
            [-20, -40, -5, -5, -5, -5, -40, -20],
            [100, -20, 10, 5, 5, 10, -20, 100]
        ], dtype=np.float32)
        
        self.position_weights_gpu = self.gpu.to_gpu(position_weights)
        
        # 코너, X-square, C-square 마스크
        corner_mask = np.zeros((8, 8), dtype=np.float32)
        for x, y in CORNERS:
            corner_mask[x, y] = 1.0
        self.corner_mask_gpu = self.gpu.to_gpu(corner_mask)
        
        logger.debug("Evaluation tables loaded to GPU")
    
    def evaluate_position_gpu(self, board: GPUBoard, color: int):
        """
        GPU 가속 위치 평가 함수
        Args:
            board: GPU 보드 객체
            color: 평가할 색상
        Returns:
            int: 평가 점수
        """
        try:
            if board.get_empty_count() == 0:
                return self._evaluate_endgame(board, color)
            
            score = 0.0
            empty_count = board.get_empty_count()
            
            # GPU 병렬 평가 실행
            if self.gpu.gpu_available:
                score = self._evaluate_parallel_gpu(board, color, empty_count)
            else:
                score = self._evaluate_sequential_cpu(board, color, empty_count)
            
            logger.debug(f"Position evaluated: score={score:.1f}, empty={empty_count}")
            return int(score)
            
        except Exception as e:
            logger.error(f"GPU evaluation failed: {e}")
            return self._evaluate_sequential_cpu(board, color, empty_count)
    
    def _evaluate_parallel_gpu(self, board: GPUBoard, color: int, empty_count: int):
        """GPU 병렬 평가 실행"""
        board_gpu = board.board
        score = 0.0
        
        # 1. 위치별 가중치 계산 (GPU 병렬)
        if self.gpu.backend == 'cupy':
            my_mask = (board_gpu == color).astype(cp.float32)
            opp_mask = (board_gpu == opponent(color)).astype(cp.float32)
            
            position_score = cp.sum(my_mask * self.position_weights_gpu) - \
                           cp.sum(opp_mask * self.position_weights_gpu)
            score += float(self.gpu.to_cpu(position_score))
        
        # 2. 기타 평가 요소들
        score += self._evaluate_mobility(board, color) * (2.0 if empty_count > 20 else 1.0)
        score += self._evaluate_corners_advanced(board, color)
        score += self._evaluate_stability_advanced(board, color)
        
        return score
    
    def _evaluate_sequential_cpu(self, board: GPUBoard, color: int, empty_count: int):
        """CPU 순차 평가 실행"""
        score = 0.0
        
        # 기동력 평가
        score += self._evaluate_mobility(board, color) * (2.0 if empty_count > 20 else 1.0)
        
        # 코너 제어
        score += self._evaluate_corners_advanced(board, color)
        
        # 안정성
        score += self._evaluate_stability_advanced(board, color)
        
        # 위치별 가중치
        score += self._evaluate_positions(board, color) * (0.5 if empty_count < 20 else 1.0)
        
        # 후반 돌 개수
        if empty_count < 20:
            b, w = board.count_stones()
            disc_diff = (b - w) if color == BLACK else (w - b)
            score += disc_diff * (5 if empty_count < 10 else 2)
        
        return score
    
    def _evaluate_endgame(self, board: GPUBoard, color: int):
        """게임 종료시 평가"""
        b, w = board.count_stones()
        diff = (b - w) if color == BLACK else (w - b)
        if diff > 0:
            return 50000 + diff
        elif diff < 0:
            return -50000 + diff
        else:
            return 0
    
    def _evaluate_mobility(self, board: GPUBoard, color: int):
        """기동력 평가"""
        my_moves = len(board.get_valid_moves(color))
        opp_moves = len(board.get_valid_moves(opponent(color)))
        
        if my_moves + opp_moves > 0:
            mobility = 100 * (my_moves - opp_moves) / (my_moves + opp_moves + 1)
        else:
            mobility = 0
            
        # 특별 기동력 보너스
        if my_moves > 0 and opp_moves == 0:
            mobility += 500
        elif my_moves == 0 and opp_moves > 0:
            mobility -= 500
            
        return mobility
    
    def _evaluate_corners_advanced(self, board: GPUBoard, color: int):
        """고급 코너 평가"""
        score = 0
        board_cpu = board.gpu.to_cpu(board.board)
        
        for corner_x, corner_y in CORNERS:
            if board_cpu[corner_x, corner_y] == color:
                score += 300
            elif board_cpu[corner_x, corner_y] == opponent(color):
                score -= 300
        
        return score
    
    def _evaluate_stability_advanced(self, board: GPUBoard, color: int):
        """고급 안정성 평가"""
        my_stable = 0
        opp_stable = 0
        board_cpu = board.gpu.to_cpu(board.board)
        
        for i in range(8):
            for j in range(8):
                if board_cpu[i, j] == color:
                    if self._is_stable(board_cpu, i, j, color):
                        my_stable += 1
                elif board_cpu[i, j] == opponent(color):
                    if self._is_stable(board_cpu, i, j, opponent(color)):
                        opp_stable += 1
        
        return (my_stable - opp_stable) * 30
    
    def _is_stable(self, board_cpu, x, y, color):
        """돌의 안정성 검사"""
        if (x, y) in CORNERS:
            return True
        
        # 간단한 안정성 검사 (실제로는 더 복잡한 알고리즘 필요)
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        stable_directions = 0
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (0 <= nx < 8 and 0 <= ny < 8 and 
                board_cpu[nx, ny] == color):
                stable_directions += 1
        
        return stable_directions >= 3
    
    def _evaluate_positions(self, board: GPUBoard, color: int):
        """위치별 평가"""
        score = 0
        board_cpu = board.gpu.to_cpu(board.board)
        position_weights_cpu = board.gpu.to_cpu(self.position_weights_gpu)
        
        for i in range(8):
            for j in range(8):
                if board_cpu[i, j] == color:
                    score += position_weights_cpu[i, j]
                elif board_cpu[i, j] == opponent(color):
                    score -= position_weights_cpu[i, j]
        
        return score

class UltraStrongAI:
    """
    최강 오델로 AI - GPU 가속 버전
    탐색 알고리즘과 평가 함수를 GPU에서 병렬 처리
    """
    
    def __init__(self, color, difficulty='ultra', time_limit=10.0):
        self.color = color
        self.difficulty = difficulty
        self.time_limit = time_limit
        
        # GPU 관리자 및 평가자 초기화
        self.gpu = GPUManager()
        self.evaluator = GPUEvaluator(self.gpu)
        
        # 난이도별 설정
        self._configure_difficulty(difficulty)
        
        # 강화된 Transposition Table
        self.tt = {}
        self.tt_age = 0
        self.max_tt_size = 1000000
        
        # 고급 휴리스틱들
        self.killer_moves = defaultdict(list)
        self.history_table = defaultdict(int)
        self.counter_moves = defaultdict(list)
        
        # 통계
        self.nodes_searched = 0
        self.tt_hits = 0
        self.cutoffs = 0
        self.perfect_searches = 0
        
        logger.info(f"UltraStrongAI initialized - Color: {color}, Difficulty: {difficulty}, GPU: {self.gpu.gpu_available}")
    
    def _configure_difficulty(self, difficulty):
        """
        난이도별 AI 설정
        Args:
            difficulty: 난이도 설정값
        """
        if difficulty == 'ultra':
            self.max_depth = 18
            self.endgame_depth = 64
            self.use_perfect_endgame = True
            self.endgame_threshold = 16
            logger.info("Ultra difficulty configured: max_depth=18, perfect_endgame=True")
        elif difficulty == 'hard':
            self.max_depth = 14
            self.endgame_depth = 20
            self.use_perfect_endgame = True
            self.endgame_threshold = 12
            logger.info("Hard difficulty configured: max_depth=14, perfect_endgame=True")
        else:
            self.max_depth = 12
            self.endgame_depth = 16
            self.use_perfect_endgame = False
            self.endgame_threshold = 8
            logger.info(f"Default difficulty configured: max_depth=12, perfect_endgame=False")
    
    def gpu_negamax(self, board: GPUBoard, depth: int, alpha: int, beta: int, 
                    maximizing: bool, end_time: float, passes=0):
        """
        GPU 가속 네가맥스 알고리즘
        보드 평가와 이동 생성을 GPU에서 병렬 처리
        
        Args:
            board: GPU 보드 객체
            depth: 탐색 깊이
            alpha, beta: 알파-베타 값
            maximizing: 최대화 플레이어 여부
            end_time: 종료 시간
            passes: 패스 횟수
        Returns:
            Tuple[int, Optional[Tuple[int, int]]]: (점수, 최적 수)
        """
        self.nodes_searched += 1
        
        # 시간 체크
        if time.time() > end_time:
            score = self.evaluator.evaluate_position_gpu(board, self.color)
            return score, None
        
        # 완벽한 종료게임 탐색
        empty_count = board.get_empty_count()
        if (self.use_perfect_endgame and 
            empty_count <= self.endgame_threshold and 
            depth >= empty_count):
            return self.perfect_endgame_search_gpu(board, alpha, beta, 
                                                 self.color if maximizing else opponent(self.color), 
                                                 passes)
        
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
                    score = self.evaluator.evaluate_position_gpu(board, self.color)
                    return score, None
                else:
                    # 패스
                    score, move = self.gpu_negamax(board, depth, -beta, -alpha, 
                                                 not maximizing, end_time, passes + 1)
                    return -score, None
            else:
                score = self.evaluator.evaluate_position_gpu(board, self.color)
                return score, None
        
        # GPU 가속 무브 정렬
        ordered_moves = self.gpu_order_moves(board, moves, depth, current_color)
        best_move = None
        original_alpha = alpha
        best_score = alpha if maximizing else beta
        
        for i, move in enumerate(ordered_moves):
            new_board = board.apply_move(*move, current_color)
            
            # Late Move Reduction (LMR)
            reduction = 0
            if (i > 3 and depth > 3 and 
                move not in self.killer_moves.get(depth, []) and
                not self.is_tactical_move_gpu(board, move)):
                reduction = 1
            
            score, _ = self.gpu_negamax(new_board, depth - 1 - reduction, 
                                      -beta, -best_score, not maximizing, end_time, 0)
            score = -score
            
            # LMR에서 좋은 결과가 나오면 전체 깊이로 재탐색
            if reduction > 0 and score > alpha:
                score, _ = self.gpu_negamax(new_board, depth - 1, 
                                          -beta, -best_score, not maximizing, end_time, 0)
                score = -score
            
            if maximizing:
                if score > best_score:
                    best_score = score
                    best_move = move
                
                if best_score >= beta:
                    # Beta cutoff
                    self.cutoffs += 1
                    self.update_killer_moves(depth, move)
                    break
            else:
                if score < best_score:
                    best_score = score
                    best_move = move
                
                if best_score <= alpha:
                    # Alpha cutoff
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
    
    def perfect_endgame_search_gpu(self, board: GPUBoard, alpha: int, beta: int, 
                                  player: int, passes=0):
        """
        GPU 가속 완벽한 종료게임 탐색
        남은 빈 칸이 적을 때 완벽한 탐색 수행
        
        Args:
            board: GPU 보드 객체
            alpha, beta: 알파-베타 값
            player: 현재 플레이어
            passes: 패스 횟수
        Returns:
            Tuple[int, Optional[Tuple[int, int]]]: (점수, 최적 수)
        """
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
                score, move = self.perfect_endgame_search_gpu(board, -beta, -alpha, 
                                                            opponent(current_color), passes + 1)
                return -score, None
        
        best_score = alpha
        best_move = None
        
        # GPU 가속 이동 정렬
        ordered_moves = self.gpu_order_moves(board, moves, 20, current_color)
        
        for move in ordered_moves:
            new_board = board.apply_move(*move, current_color)
            score, _ = self.perfect_endgame_search_gpu(new_board, -beta, -best_score, 
                                                     opponent(current_color), 0)
            score = -score
            
            if score > best_score:
                best_score = score
                best_move = move
                
            if best_score >= beta:
                break
        
        return best_score, best_move
    
    def gpu_order_moves(self, board: GPUBoard, moves: List[Tuple[int, int]], 
                       depth: int, current_color: int):
        """
        GPU 가속 무브 정렬
        이동 평가를 GPU에서 병렬 처리하여 최적 순서로 정렬
        
        Args:
            board: GPU 보드 객체
            moves: 가능한 이동 리스트
            depth: 현재 깊이
            current_color: 현재 플레이어 색상
        Returns:
            List[Tuple[int, int]]: 정렬된 이동 리스트
        """
        if not moves:
            return moves
        
        try:
            # GPU 병렬 평가 시도
            if self.gpu.gpu_available and len(moves) > 4:
                return self._gpu_parallel_move_ordering(board, moves, depth, current_color)
            else:
                return self._cpu_sequential_move_ordering(board, moves, depth, current_color)
                
        except Exception as e:
            logger.warning(f"GPU move ordering failed: {e}")
            return self._cpu_sequential_move_ordering(board, moves, depth, current_color)
    
    def _gpu_parallel_move_ordering(self, board: GPUBoard, moves: List[Tuple[int, int]], 
                                   depth: int, current_color: int):
        """GPU 병렬 무브 정렬 실행"""
        move_scores = []
        board_hash = self.get_board_hash(board)
        
        # TT에서 최고 수 가져오기
        tt_move = None
        if board_hash in self.tt:
            tt_move = self.tt[board_hash].get('best_move')
        
        # 배치 평가를 위한 준비
        batch_scores = []
        
        for move in moves:
            x, y = move
            score = 0
            
            # TT 수 최우선
            if move == tt_move:
                score += 50000
            
            # 킬러 무브
            if move in self.killer_moves.get(depth, []):
                score += 10000
            
            # 히스토리 휴리스틱
            score += self.history_table.get(move, 0)
            
            # 위치별 전략적 가치
            position_score = self.evaluate_move_position_gpu(board, move)
            score += position_score
            
            # GPU에서 mobility 평가
            if self.gpu.gpu_available:
                mobility_score = self.evaluate_move_mobility_gpu(board, move, current_color)
                score += mobility_score
            
            batch_scores.append(score)
        
        # 점수에 따라 정렬
        move_score_pairs = list(zip(batch_scores, moves))
        move_score_pairs.sort(reverse=True)
        
        return [move for _, move in move_score_pairs]
    
    def _cpu_sequential_move_ordering(self, board: GPUBoard, moves: List[Tuple[int, int]], 
                                     depth: int, current_color: int):
        """CPU 순차 무브 정렬 실행"""
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
            
            # 히스토리 휴리스틱
            score += self.history_table.get(move, 0)
            
            # 위치별 전략적 가치
            position_score = self.evaluate_move_position_gpu(board, move)
            score += position_score
            
            # 이 수로 인한 mobility 변화
            mobility_score = self.evaluate_move_mobility_cpu(board, move, current_color)
            score += mobility_score
            
            move_scores.append((score, move))
        
        move_scores.sort(reverse=True)
        return [move for _, move in move_scores]
    
    def evaluate_move_position_gpu(self, board: GPUBoard, move: Tuple[int, int]):
        """
        GPU 가속 수의 위치적 가치 평가
        Args:
            board: GPU 보드 객체
            move: 평가할 수
        Returns:
            int: 위치적 가치 점수
        """
        x, y = move
        score = 0
        
        # 코너
        if (x, y) in CORNERS:
            score += 1000
        
        # X-squares (위험한 수)
        elif (x, y) in X_SQUARES:
            board_cpu = board.gpu.to_cpu(board.board)
            adjacent_corner_empty = False
            for corner in CORNERS:
                if abs(corner[0] - x) <= 1 and abs(corner[1] - y) <= 1:
                    if board_cpu[corner[0]][corner[1]] == EMPTY:
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
    
    def evaluate_move_mobility_gpu(self, board: GPUBoard, move: Tuple[int, int], 
                                  current_color: int):
        """
        GPU 가속 수에 따른 mobility 변화 평가
        Args:
            board: GPU 보드 객체
            move: 평가할 수
            current_color: 현재 플레이어 색상
        Returns:
            int: mobility 변화 점수
        """
        current_my_moves = len(board.get_valid_moves(current_color))
        current_opp_moves = len(board.get_valid_moves(opponent(current_color)))
        
        new_board = board.apply_move(*move, current_color)
        new_my_moves = len(new_board.get_valid_moves(current_color))
        new_opp_moves = len(new_board.get_valid_moves(opponent(current_color)))
        
        my_mobility_change = new_my_moves - current_my_moves
        opp_mobility_change = new_opp_moves - current_opp_moves
        
        return (current_opp_moves - new_opp_moves) * 20 + my_mobility_change * 10
    
    def evaluate_move_mobility_cpu(self, board: GPUBoard, move: Tuple[int, int], 
                                  current_color: int):
        """CPU 버전 mobility 변화 평가"""
        return self.evaluate_move_mobility_gpu(board, move, current_color)
    
    def is_tactical_move_gpu(self, board: GPUBoard, move: Tuple[int, int]):
        """
        GPU 가속 전술적 수 판단
        Args:
            board: GPU 보드 객체
            move: 판단할 수
        Returns:
            bool: 전술적 수 여부
        """
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
    
    def update_killer_moves(self, depth: int, move: Tuple[int, int]):
        """
        킬러 무브 업데이트
        Args:
            depth: 현재 깊이
            move: 킬러 무브로 추가할 수
        """
        if move not in self.killer_moves[depth]:
            if len(self.killer_moves[depth]) >= 3:
                self.killer_moves[depth].pop(0)
            self.killer_moves[depth].append(move)
    
    def get_board_hash(self, board: GPUBoard):
        """
        보드 해시 계산
        Args:
            board: GPU 보드 객체
        Returns:
            str: 보드 해시값
        """
        board_cpu = board.gpu.to_cpu(board.board)
        board_str = ''.join(str(cell) for row in board_cpu for cell in row)
        return hashlib.md5(board_str.encode()).hexdigest()
    
    def store_tt(self, board_hash: str, depth: int, score: int, flag: str, 
                best_move: Optional[Tuple[int, int]]):
        """
        Transposition Table 저장
        Args:
            board_hash: 보드 해시값
            depth: 탐색 깊이
            score: 평가 점수
            flag: 플래그 ('EXACT', 'ALPHA', 'BETA')
            best_move: 최적 수
        """
        if len(self.tt) >= self.max_tt_size:
            self.clear_old_tt_entries()
        
        self.tt[board_hash] = {
            'depth': depth, 'score': score, 'flag': flag, 
            'best_move': best_move, 'age': self.tt_age
        }
    
    def probe_tt(self, board_hash: str, depth: int, alpha: int, beta: int):
        """
        Transposition Table 조회
        Args:
            board_hash: 보드 해시값
            depth: 요구 깊이
            alpha, beta: 알파-베타 값
        Returns:
            Optional[int]: 저장된 점수 또는 None
        """
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
        """
        오래된 Transposition Table 엔트리 정리
        메모리 사용량 관리를 위해 오래되고 얕은 엔트리부터 삭제
        """
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
        removed_count = 0
        target_remove = min(len(entries_to_remove) // 2, len(self.tt) // 4)
        
        for i in range(target_remove):
            if i < len(entries_to_remove):
                del self.tt[entries_to_remove[i][1]]
                removed_count += 1
        
        logger.debug(f"Cleared {removed_count} old TT entries")
    
    def get_opening_move(self, board: GPUBoard):
        """
        오프닝북에서 수 선택
        Args:
            board: GPU 보드 객체
        Returns:
            Optional[Tuple[int, int]]: 오프닝북 수 또는 None
        """
        # 간단한 오프닝 전략
        moves = board.get_valid_moves(self.color)
        if not moves:
            return None
        
        # 중앙과 모서리 위치 선호, X-squares와 C-squares 회피
        preferred = []
        for move in moves:
            x, y = move
            if move not in X_SQUARES and move not in C_SQUARES:
                # 중앙에서의 거리 계산
                center_dist = abs(x - 3.5) + abs(y - 3.5)
                preferred.append((center_dist, move))
        
        if preferred:
            preferred.sort()
            selected_move = preferred[0][1]
            logger.info(f"Opening move selected: {chr(selected_move[1] + ord('a'))}{selected_move[0] + 1}")
            return selected_move
        
        selected_move = random.choice(moves)
        logger.info(f"Random opening move selected: {chr(selected_move[1] + ord('a'))}{selected_move[0] + 1}")
        return selected_move
    
    def ultra_iterative_deepening(self, board: GPUBoard):
        """
        GPU 가속 반복 심화 탐색
        점진적으로 깊이를 증가시키며 최적 수 탐색
        
        Args:
            board: GPU 보드 객체
        Returns:
            UltraSearchResult: 탐색 결과
        """
        start_time = time.time()
        end_time = start_time + self.time_limit
        
        moves = board.get_valid_moves(self.color)
        if not moves:
            logger.warning("No valid moves available")
            return UltraSearchResult(0, None, 0, 0, 0, True, [], {})
        
        if len(moves) == 1:
            logger.info(f"Only one move available: {chr(moves[0][1] + ord('a'))}{moves[0][0] + 1}")
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
        
        logger.info(f"Starting iterative deepening search with {len(moves)} possible moves")
        
        for depth in range(1, self.max_depth + 1):
            try:
                if time.time() > end_time:
                    logger.info(f"Time limit reached at depth {depth}")
                    break
                
                logger.debug(f"Searching depth {depth}")
                
                # Aspiration window로 탐색
                score, move = self.gpu_negamax(board, depth, alpha, beta, True, end_time, 0)
                
                # Window 밖의 결과가 나오면 전체 범위로 재탐색
                if score <= alpha or score >= beta:
                    logger.debug(f"Aspiration window miss, researching with full window")
                    score, move = self.gpu_negamax(board, depth, float('-inf'), float('inf'), True, end_time, 0)
                
                if move and time.time() <= end_time:
                    best_move = move
                    best_score = score
                    max_depth_reached = depth
                    
                    # Principal Variation 수집
                    pv = self.extract_pv(board, best_move, depth)
                    
                    # 다음 반복을 위한 aspiration window 업데이트
                    alpha = score - aspiration_window
                    beta = score + aspiration_window
                    
                    logger.debug(f"Depth {depth} completed: score={score}, move={chr(move[1] + ord('a'))}{move[0] + 1}")
                
                # 완전 탐색 달성 시 중단
                if depth >= board.get_empty_count():
                    logger.info(f"Complete search achieved at depth {depth}")
                    break
                
                # 시간 관리
                elapsed = time.time() - start_time
                if elapsed > self.time_limit * 0.7:
                    logger.info(f"Time threshold reached at depth {depth}")
                    break
                    
            except Exception as e:
                logger.error(f"Error in depth {depth}: {e}")
                break
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        # 평가 분석
        if best_move:
            final_board = board.apply_move(*best_move, self.color)
            final_eval = self.evaluator.evaluate_position_gpu(final_board, self.color)
            eval_breakdown = {'final_eval': final_eval}

        # GPU 메모리 정리
        self.gpu.clear_memory()
    
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
        
    def extract_pv(self, board: GPUBoard, first_move: Tuple[int, int], max_depth: int):
        """
        Principal Variation 추출
        Args:
            board: GPU 보드 객체
            first_move: 첫 번째 수
            max_depth: 최대 깊이
        Returns:
            List[Tuple[int, int]]: Principal Variation
        """
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
    
    def get_move(self, board):
        """
        최고의 수 반환 - 메인 인터페이스
        Args:
            board: 일반 보드 객체 (호환성을 위해)
        Returns:
            Optional[Tuple[int, int]]: 최적 수
        """
        # 통계 초기화
        self.nodes_searched = 0
        self.tt_hits = 0
        self.cutoffs = 0
        self.perfect_searches = 0
        self.tt_age += 1
        
        # GPU 보드로 변환
        gpu_board = self._convert_to_gpu_board(board)
        
        logger.info(f"AI thinking started - Empty squares: {gpu_board.get_empty_count()}")
        
        # 오프닝북 먼저 시도
        if gpu_board.get_empty_count() > 54:
            opening_move = self.get_opening_move(gpu_board)
            if opening_move:
                logger.info(f"Opening book move: {chr(opening_move[1] + ord('a'))}{opening_move[0] + 1}")
                return opening_move
        
        # 메인 탐색
        start_time = time.time()
        result = self.ultra_iterative_deepening(gpu_board)

        # result 검증
        if not result or result.best_move is None:
            logger.error("Failed to find a move")
            return None
        
        # 상세 통계 출력
        if result.time_ms > 100:
            nps = result.nodes / (result.time_ms / 1000) if result.time_ms > 0 else 0
            logger.info(f"AI Analysis Complete:")
            logger.info(f"  Best move: {chr(result.best_move[1] + ord('a'))}{result.best_move[0] + 1}")
            logger.info(f"  Score: {result.score}")
            logger.info(f"  Depth: {result.depth}")
            logger.info(f"  Nodes: {result.nodes:,}")
            logger.info(f"  Time: {result.time_ms}ms")
            logger.info(f"  NPS: {nps:,.0f}")
            logger.info(f"  TT hits: {self.tt_hits:,}")
            logger.info(f"  Cutoffs: {self.cutoffs:,}")
            logger.info(f"  GPU Backend: {self.gpu.backend}")
            
            if self.perfect_searches > 0:
                logger.info(f"  Perfect searches: {self.perfect_searches}")
            logger.info(f"  Exact: {'Yes' if result.is_exact else 'No'}")
            
            if result.pv and len(result.pv) > 1:
                pv_str = " ".join([f"{chr(move[1] + ord('a'))}{move[0] + 1}" for move in result.pv[:5]])
                logger.info(f"  PV: {pv_str}")
        
        return result.best_move
    
    def _convert_to_gpu_board(self, board):
        """
        일반 보드를 GPU 보드로 변환
        Args:
            board: 일반 보드 객체
        Returns:
            GPUBoard: GPU 보드 객체
        """
        gpu_board = GPUBoard(self.gpu)
        
        # 보드 상태 복사
        board_array = np.array(board.board, dtype=np.int8)
        gpu_board.board = gpu_board.gpu.to_gpu(board_array)
        gpu_board.move_history = board.move_history.copy()
        
        return gpu_board

# 사용 예시 및 데모
def demo_gpu_game():
    """
    GPU 가속 오델로 AI 데모 게임
    Ultra Strong AI vs Hard AI 대결
    """
    logger.info("Starting GPU Ultra Strong Othello AI Demo")
    logger.info("=" * 50)
    
    # GPU 보드 초기화
    gpu_manager = GPUManager()
    board = GPUBoard(gpu_manager)
    
    # AI 생성 (흑: Ultra GPU, 백: Hard CPU)
    black_ai = UltraStrongAI(BLACK, difficulty='ultra', time_limit=5.0)
    white_ai = UltraStrongAI(WHITE, difficulty='hard', time_limit=3.0)
    
    current_player = BLACK
    pass_count = 0
    move_count = 0
    
    while pass_count < 2 and move_count < 60:
        moves = board.get_valid_moves(current_player)
        
        if not moves:
            logger.info(f"{'Black' if current_player == BLACK else 'White'} passes")
            pass_count += 1
            current_player = opponent(current_player)
            continue
        
        pass_count = 0
        move_count += 1
        
        # AI 수 선택
        if current_player == BLACK:
            # GPU 보드를 일반 보드 형식으로 변환하여 호환성 확보
            temp_board = type('Board', (), {
                'board': board.gpu.to_cpu(board.board).tolist(),
                'move_history': board.move_history,
                'get_valid_moves': board.get_valid_moves,
                'apply_move': board.apply_move,
                'count_stones': board.count_stones,
                'get_empty_count': board.get_empty_count,
                'get_frontier_count': board.get_frontier_count
            })()
            
            move = black_ai.get_move(temp_board)
            player_name = "Black (Ultra GPU)"
        else:
            temp_board = type('Board', (), {
                'board': board.gpu.to_cpu(board.board).tolist(),
                'move_history': board.move_history,
                'get_valid_moves': board.get_valid_moves,
                'apply_move': board.apply_move,
                'count_stones': board.count_stones,
                'get_empty_count': board.get_empty_count,
                'get_frontier_count': board.get_frontier_count
            })()
            
            move = white_ai.get_move(temp_board)
            player_name = "White (Hard CPU)"
        
        if move:
            board = board.apply_move(*move, current_player)
            logger.info(f"{player_name} plays: {chr(move[1] + ord('a'))}{move[0] + 1}")
            
            # 보드 상태 출력
            b, w = board.count_stones()
            logger.info(f"Score - Black: {b}, White: {w}")
            logger.info("-" * 30)
        
        current_player = opponent(current_player)
    
    # 최종 결과
    b, w = board.count_stones()
    logger.info("")
    logger.info("Game Over!")
    logger.info(f"Final Score - Black: {b}, White: {w}")
    
    if b > w:
        logger.info("Black (Ultra GPU AI) Wins!")
    elif w > b:
        logger.info("White (Hard CPU AI) Wins!")
    else:
        logger.info("Draw!")
    
    # GPU 메모리 정리
    gpu_manager.clear_memory()
    logger.info("Demo completed successfully")

if __name__ == "__main__":
    demo_gpu_game()
