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
from collections import deque
import os


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    TORCH_AVAILABLE = True
    print("PyTorch loaded successfully for neural network training")
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available, neural network features disabled")

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

def setup_ai_logger():
    """Training Pipeline 전용 로거 설정 - 개선된 버전"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger('AI_Logger')
    logger.setLevel(logging.DEBUG)  # DEBUG 레벨로 변경
    
    # 기존 핸들러 제거
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 파일 핸들러 추가
    file_handler = logging.FileHandler(
        os.path.join(log_dir, 'AI.log'),
        mode='a',
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    
    # 콘솔 핸들러 추가 (통계 확인용)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 더 상세한 포맷터
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(funcName)s - %(message)s'
    )
    
    file_handler.setFormatter(detailed_formatter)
    console_handler.setFormatter(detailed_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # 콘솔 출력 활성화
    logger.propagate = False
    
    return logger

logger = setup_ai_logger()

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
                #logger.info(f"GPU Manager initialized with CuPy backend")
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
    
# GPUEvaluator 클래스 다음에 추가
class GPUMCTSNode:
    """GPU 최적화된 MCTS 노드"""
    
    def __init__(self, board, color, parent=None, action=None, prior=0):
        self.board = board
        self.color = color
        self.parent = parent
        self.action = action
        self.prior = prior
        
        self.visits = 0
        self.value_sum = 0
        self.children = {}
        self.is_expanded = False
    
    def is_leaf(self):
        return not self.is_expanded
    
    def select_child(self, c_puct=1.0):
        """UCB 공식으로 자식 노드 선택"""
        best_score = float('-inf')
        best_action = None
        
        for action, child in self.children.items():
            if child.visits == 0:
                ucb_score = float('inf')
            else:
                q_value = child.value_sum / child.visits
                u_value = (c_puct * child.prior * 
                          math.sqrt(self.visits) / (1 + child.visits))
                ucb_score = q_value + u_value
            
            if ucb_score > best_score:
                best_score = ucb_score
                best_action = action
        
        return self.children[best_action]
    
    def expand(self, action_probs):
        """노드 확장"""
        valid_moves = self.board.get_valid_moves(self.color)
        
        for move in valid_moves:
            action_idx = move[0] * 8 + move[1]
            prior = action_probs[action_idx] if action_idx < len(action_probs) else 0.1
            
            new_board = self.board.apply_move(*move, self.color)
            child = GPUMCTSNode(new_board, opponent(self.color), 
                               parent=self, action=move, prior=prior)
            self.children[move] = child
        
        self.is_expanded = True
    
    def backup(self, value):
        """백업 (역전파)"""
        self.visits += 1
        self.value_sum += value
        
        if self.parent:
            self.parent.backup(-value)

class GPUAlphaZeroMCTS:
    """GPU 가속 AlphaZero MCTS"""
    
    def __init__(self, neural_net, gpu_manager, c_puct=1.0, num_simulations=800):
        self.neural_net = neural_net
        self.gpu = gpu_manager
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if self.neural_net:
            self.neural_net.to(self.device)
    
    def search(self, board, color):
        """MCTS 탐색 실행"""
        root = GPUMCTSNode(board, color)
        
        # 배치 처리를 위한 준비
        batch_size = min(32, self.num_simulations // 4)
        
        for simulation_batch in range(0, self.num_simulations, batch_size):
            current_batch_size = min(batch_size, self.num_simulations - simulation_batch)
            
            # 배치 단위로 시뮬레이션 실행
            self._batch_simulate(root, current_batch_size)
        
        return self.get_action_probabilities(root)
    
    def _batch_simulate(self, root, batch_size):
        """배치 단위 시뮬레이션"""
        for _ in range(batch_size):
            node = root
            path = []
            
            # Selection
            while not node.is_leaf() and node.children:
                path.append(node)
                node = node.select_child(self.c_puct)
            
            # Expansion & Evaluation
            if not node.board.get_valid_moves(node.color):
                value = self.evaluate_terminal(node.board, root.color)
            else:
                policy, value = self.neural_net_predict(node.board, node.color)
                
                if not node.is_expanded:
                    node.expand(policy)
            
            # Backup
            node.backup(value if node.color == root.color else -value)
    
    def neural_net_predict(self, board, color):
        """신경망 예측"""
        if not self.neural_net:
            # 신경망이 없으면 랜덤 정책과 0 가치 반환
            return np.ones(64) / 64, 0.0
        
        board_tensor = self.board_to_tensor(board, color)
        
        with torch.no_grad():
            policy_logits, value = self.neural_net(board_tensor.unsqueeze(0))
            policy = torch.exp(policy_logits).squeeze().cpu().numpy()
            value = value.item()
        
        return policy, value
    
    def board_to_tensor(self, board, color):
        """보드를 텐서로 변환"""
        tensor = torch.zeros(3, 8, 8, device=self.device)
        
        board_cpu = board.gpu.to_cpu(board.board)
        
        for i in range(8):
            for j in range(8):
                if board_cpu[i][j] == color:
                    tensor[0][i][j] = 1
                elif board_cpu[i][j] == opponent(color):
                    tensor[1][i][j] = 1
        
        # 현재 플레이어 정보
        if color == BLACK:
            tensor[2] = torch.ones(8, 8, device=self.device)
        
        return tensor
    
    def get_action_probabilities(self, root):
        """액션 확률 분포 반환"""
        visits = np.zeros(64)
        
        for action, child in root.children.items():
            action_idx = action[0] * 8 + action[1]
            visits[action_idx] = child.visits
        
        if visits.sum() == 0:
            return visits
        
        return visits / visits.sum()
    
    def evaluate_terminal(self, board, color):
        """터미널 노드 평가"""
        b, w = board.count_stones()
        if color == BLACK:
            return 1 if b > w else (-1 if b < w else 0)
        else:
            return 1 if w > b else (-1 if w < b else 0)
        
# GPUAlphaZeroMCTS 클래스 다음에 추가
class GPUSelfPlayTrainer:
    """GPU 가속 자가 학습 트레이너"""
    
    def __init__(self, neural_net=None, gpu_manager=None, learning_rate=0.001):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gpu = gpu_manager
        
        if neural_net is None:
            self.neural_net = GPUOthelloNet().to(self.device)
        else:
            self.neural_net = neural_net.to(self.device)
        
        self.optimizer = optim.Adam(self.neural_net.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)
        
        # 훈련 데이터 저장
        from collections import deque
        self.training_data = deque(maxlen=100000)
        self.training_stats = {
            'games_played': 0,
            'training_iterations': 0,
            'avg_policy_loss': 0,
            'avg_value_loss': 0
        }
        
        logger.info(f"GPU Self-Play Trainer initialized on device: {self.device}")
    
    def self_play_game(self, temperature=1.0, add_noise=True):
        """자가 대국 한 게임"""
        board = GPUBoard(self.gpu)
        game_data = []
        current_player = BLACK
        move_count = 0
        
        mcts = GPUAlphaZeroMCTS(self.neural_net, self.gpu, num_simulations=400)
        
        while True:
            moves = board.get_valid_moves(current_player)
            if not moves:
                current_player = opponent(current_player)
                moves = board.get_valid_moves(current_player)
                if not moves:
                    break
            
            # MCTS로 수 선택
            action_probs = mcts.search(board, current_player)
            
            # 노이즈 추가 (탐험 증진)
            if add_noise and move_count < 30:
                noise = np.random.dirichlet([0.3] * 64)
                action_probs = 0.75 * action_probs + 0.25 * noise
            
            # 데이터 저장
            board_tensor = mcts.board_to_tensor(board, current_player)
            game_data.append((board_tensor.cpu(), action_probs, current_player))
            
            # 수 실행
            best_move = self.select_move_from_probs(moves, action_probs, temperature)
            board = board.apply_move(*best_move, current_player)
            current_player = opponent(current_player)
            move_count += 1
        
        # 게임 결과로 라벨링
        b, w = board.count_stones()
        if b > w:
            winner = BLACK
        elif w > b:
            winner = WHITE
        else:
            winner = None
        
        labeled_data = []
        for board_state, probs, player in game_data:
            if winner is None:
                value = 0
            elif winner == player:
                value = 1
            else:
                value = -1
            labeled_data.append((board_state, probs, value))
        
        self.training_stats['games_played'] += 1
        return labeled_data
    
    def select_move_from_probs(self, valid_moves, action_probs, temperature):
        """확률 분포에서 수 선택"""
        if temperature == 0:
            best_prob = 0
            best_move = valid_moves[0]
            
            for move in valid_moves:
                action_idx = move[0] * 8 + move[1]
                if action_probs[action_idx] > best_prob:
                    best_prob = action_probs[action_idx]
                    best_move = move
            
            return best_move
        else:
            move_probs = []
            for move in valid_moves:
                action_idx = move[0] * 8 + move[1]
                prob = action_probs[action_idx] ** (1 / temperature)
                move_probs.append(prob)
            
            total_prob = sum(move_probs)
            if total_prob > 0:
                move_probs = [p / total_prob for p in move_probs]
                return np.random.choice(valid_moves, p=move_probs)
            else:
                return random.choice(valid_moves)
    
    def train_iteration(self, num_games=50, batch_size=32, epochs=10):
        """훈련 반복"""
        logger.info(f"GPU 자가 대국 {num_games}게임 시작...")
        
        # 자가 대국으로 데이터 생성
        for i in range(num_games):
            temperature = max(0.1, 1.0 - (i / num_games) * 0.9)
            
            game_data = self.self_play_game(temperature=temperature)
            self.training_data.extend(game_data)
            
            if (i + 1) % 10 == 0:
                logger.info(f"  {i + 1}/{num_games} 게임 완료")
        
        logger.info(f"GPU 신경망 훈련 시작 (데이터: {len(self.training_data)}개)...")
        
        # 신경망 훈련
        self.train_neural_net(batch_size=batch_size, epochs=epochs)
        
        self.training_stats['training_iterations'] += 1
        logger.info(f"GPU 훈련 반복 {self.training_stats['training_iterations']} 완료")
    
    def train_neural_net(self, batch_size=32, epochs=10):
        """GPU 가속 신경망 훈련"""
        if len(self.training_data) < batch_size:
            logger.warning("훈련 데이터 부족")
            return
        
        self.neural_net.train()
        
        total_policy_loss = 0
        total_value_loss = 0
        num_batches = 0
        
        for epoch in range(epochs):
            training_list = list(self.training_data)
            random.shuffle(training_list)
            
            epoch_policy_loss = 0
            epoch_value_loss = 0
            epoch_batches = 0
            
            for i in range(0, len(training_list), batch_size):
                batch = training_list[i:i + batch_size]
                if len(batch) < batch_size:
                    continue
                
                # 배치 데이터 준비 (GPU로 이동)
                boards = torch.stack([item[0] for item in batch]).to(self.device)
                target_policies = torch.tensor([item[1] for item in batch]).to(self.device)
                target_values = torch.tensor([[item[2]] for item in batch], dtype=torch.float32).to(self.device)
                
                # 순전파
                pred_policies, pred_values = self.neural_net(boards)
                
                # 손실 계산
                policy_loss = F.kl_div(pred_policies, target_policies, reduction='batchmean')
                value_loss = F.mse_loss(pred_values, target_values)
                total_loss = policy_loss + value_loss
                
                # 역전파
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.neural_net.parameters(), 1.0)
                self.optimizer.step()
                
                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()
                epoch_batches += 1
            
            if epoch_batches > 0:
                total_policy_loss += epoch_policy_loss / epoch_batches
                total_value_loss += epoch_value_loss / epoch_batches
                num_batches += 1
        
        self.scheduler.step()
        
        if num_batches > 0:
            self.training_stats['avg_policy_loss'] = total_policy_loss / num_batches
            self.training_stats['avg_value_loss'] = total_value_loss / num_batches
            
            logger.info(f"  평균 Policy Loss: {self.training_stats['avg_policy_loss']:.4f}")
            logger.info(f"  평균 Value Loss: {self.training_stats['avg_value_loss']:.4f}")
    
    def save_model(self, filepath):
        """모델 저장"""
        torch.save({
            'model_state_dict': self.neural_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats
        }, filepath)
        logger.info(f"GPU 모델 저장 완료: {filepath}")
    
    def load_model(self, filepath):
        """모델 로드"""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.neural_net.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.training_stats = checkpoint.get('training_stats', self.training_stats)
            logger.info(f"GPU 모델 로드 완료: {filepath}")
            return True
        return False


# GPUEvaluator 클래스 다음에 추가

class GPUOthelloNet(nn.Module):
    """GPU 최적화된 오델로 신경망"""
    
    def __init__(self, board_size=8, num_channels=256, num_res_blocks=10):
        super(GPUOthelloNet, self).__init__()
        self.board_size = board_size
        self.num_channels = num_channels
        
        # 입력 레이어 (3채널: 내 돌, 상대 돌, 현재 플레이어)
        self.conv_input = nn.Conv2d(3, num_channels, 3, stride=1, padding=1)
        self.bn_input = nn.BatchNorm2d(num_channels)
        
        # 잔차 블록들
        self.res_blocks = nn.ModuleList([
            GPUResidualBlock(num_channels) for _ in range(num_res_blocks)
        ])
        
        # 정책 헤드
        self.policy_conv = nn.Conv2d(num_channels, 32, 1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * board_size * board_size, board_size * board_size)
        
        # 가치 헤드
        self.value_conv = nn.Conv2d(num_channels, 3, 1)
        self.value_bn = nn.BatchNorm2d(3)
        self.value_fc1 = nn.Linear(3 * board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0, 0.01)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # 공통 특징 추출
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        # 잔차 블록들 통과
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # 정책 헤드
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = F.log_softmax(self.policy_fc(policy), dim=1)
        
        # 가치 헤드
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value

class GPUResidualBlock(nn.Module):
    """GPU 최적화된 잔차 블록"""
    
    def __init__(self, num_channels):
        super(GPUResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class UltraStrongAI:
    """
    최강 오델로 AI - GPU 가속 버전
    탐색 알고리즘과 평가 함수를 GPU에서 병렬 처리
    """
    
    def __init__(self, color, difficulty='ultra', time_limit=10.0,use_neural_net=True):
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

        # 신경망 관련 추가
        # 기본적으로 신경망 사용
        self.use_neural_net = use_neural_net and TORCH_AVAILABLE
        
        # 항상 학습 모드 활성화
        self.continuous_learning = True
        self.learning_buffer = deque(maxlen=10000)
        
        # 자동 학습 스케줄러
        self.games_since_training = 0
        self.training_interval = 10  # 10게임마다 학습

        if self.use_neural_net:
            self.neural_net = GPUOthelloNet().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            self.mcts = GPUAlphaZeroMCTS(self.neural_net, self.gpu, num_simulations=800)
            self.trainer = GPUSelfPlayTrainer(self.neural_net, self.gpu)
            self.load_model()
        
        logger.info(f"UltraStrongAI initialized - Color: {color}, Difficulty: {difficulty}, GPU: {self.gpu.gpu_available}")

    def _collect_game_data(self, board, move):
        """게임 데이터 수집 및 학습 트리거"""
        # 현재 보드 상태와 선택한 수를 버퍼에 저장
        board_tensor = self.mcts.board_to_tensor(board, self.color)
        action_probs = self.mcts.search(board, self.color)
        
        self.learning_buffer.append({
            'board': board_tensor,
            'action_probs': action_probs,
            'color': self.color
        })
        
        self.games_since_training += 1
        
        # 주기적 학습 실행
        if self.games_since_training >= self.training_interval:
            self._trigger_background_learning()
            self.games_since_training = 0
    
    def _trigger_background_learning(self):
        """백그라운드에서 학습 실행"""
        if len(self.learning_buffer) < 100:
            return
            
        # 별도 스레드에서 학습 실행
        import threading
        learning_thread = threading.Thread(
            target=self._background_training,
            daemon=True
        )
        learning_thread.start()
    
    def _background_training(self):
        """백그라운드 학습 실행"""
        try:
            # 최근 데이터로 빠른 학습
            self.trainer.train_neural_net(
                batch_size=32,
                epochs=3  # 빠른 학습
            )
            logger.info("백그라운드 학습 완료")
        except Exception as e:
            logger.error(f"백그라운드 학습 실패: {e}")
    
    def load_model(self, model_path='models/gpu_best_model.pth'):
        """훈련된 모델 로드"""
        if not self.use_neural_net or not self.trainer:
            return
        
        try:
            if self.trainer.load_model(model_path):
                logger.info("GPU 신경망 모델 로드 완료")
            else:
                logger.warning("GPU 모델 파일을 찾을 수 없음. 랜덤 가중치 사용")
        except Exception as e:
            logger.error(f"GPU 모델 로드 중 오류: {e}")

    def get_move_with_neural_net(self, board):
        """신경망 기반 수 선택"""
        if not self.mcts:
            return self.get_move_traditional(board)
        
        # GPU 보드로 변환
        gpu_board = self._convert_to_gpu_board(board)
        
        # MCTS 탐색
        action_probs = self.mcts.search(gpu_board, self.color)
        
        # 가장 높은 확률의 수 선택
        valid_moves = gpu_board.get_valid_moves(self.color)
        if not valid_moves:
            return None
        
        best_prob = 0
        best_move = valid_moves[0]
        
        for move in valid_moves:
            action_idx = move[0] * 8 + move[1]
            if action_probs[action_idx] > best_prob:
                best_prob = action_probs[action_idx]
                best_move = move
        
        logger.info(f"신경망 AI 수: {chr(best_move[1] + ord('a'))}{best_move[0] + 1}")
        return best_move

    def start_self_play_training(self, iterations=100, games_per_iteration=50):
        """자가 학습 시작"""
        if not self.use_neural_net or not self.trainer:
            logger.error("신경망이 활성화되지 않아 자가 학습을 시작할 수 없습니다")
            return
        
        logger.info(f"GPU 자가 학습 시작: {iterations}회 반복, 반복당 {games_per_iteration}게임")
        
        for iteration in range(iterations):
            logger.info(f"\n=== GPU 자가 학습 반복 {iteration + 1}/{iterations} ===")
            
            # 자가 학습 실행
            self.trainer.train_iteration(
                num_games=games_per_iteration,
                batch_size=64,
                epochs=10
            )
            
            # 주기적 모델 저장
            if (iteration + 1) % 10 == 0:
                model_path = f'models/gpu_checkpoint_{iteration + 1}.pth'
                import os
                os.makedirs('models', exist_ok=True)
                self.trainer.save_model(model_path)
                logger.info(f"체크포인트 저장: {model_path}")
            
            # 통계 출력
            stats = self.trainer.training_stats
            logger.info(f"총 게임: {stats['games_played']}, "
                    f"Policy Loss: {stats['avg_policy_loss']:.4f}, "
                    f"Value Loss: {stats['avg_value_loss']:.4f}")
        
        # 최종 모델 저장
        final_model_path = 'models/gpu_final_model.pth'
        import os
        os.makedirs('models', exist_ok=True)
        self.trainer.save_model(final_model_path)
        logger.info(f"최종 모델 저장: {final_model_path}")

    def get_move_traditional(self, board):
        """기존 방식의 수 선택"""
        # 기존 get_move 로직을 여기로 이동
        # GPU 보드로 변환
        gpu_board = self._convert_to_gpu_board(board)
        
        # 기존 탐색 알고리즘 사용
        result = self.ultra_iterative_deepening(gpu_board)
        
        if result and result.best_move:
            logger.info(f"전통적 AI 수: {chr(result.best_move[1] + ord('a'))}{result.best_move[0] + 1}")
            return result.best_move
        
        return None

    
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
        """최고의 수 반환 - 통계 문제 해결 버전"""
        # 통계 변수 강제 초기화
        if not hasattr(self, 'nodes_searched'):
            self.nodes_searched = 0
        if not hasattr(self, 'tt_hits'):
            self.tt_hits = 0
        if not hasattr(self, 'cutoffs'):
            self.cutoffs = 0
        if not hasattr(self, 'perfect_searches'):
            self.perfect_searches = 0
        if not hasattr(self, 'tt_age'):
            self.tt_age = 0
        
        # 통계 초기화
        self.nodes_searched = 0
        self.tt_hits = 0
        self.cutoffs = 0
        self.perfect_searches = 0
        self.tt_age += 1
        
        # GPU 보드로 변환 (안전하게)
        try:
            gpu_board = self._convert_to_gpu_board(board)
            if gpu_board is None:
                logger.error("GPU 보드 변환 실패")
                return None
        except Exception as e:
            logger.error(f"GPU 보드 변환 중 오류: {e}")
            return None
        
        logger.info(f"=== AI 분석 시작 ===")
        logger.info(f"빈 칸 수: {gpu_board.get_empty_count()}")
        logger.info(f"현재 플레이어: {'흑' if self.color == BLACK else '백'}")
        logger.info(f"GPU 백엔드: {getattr(self.gpu, 'backend', 'Unknown')}")
        logger.info(f"신경망 사용: {self.use_neural_net}")
        
        # 현재 보드 상황 로깅
        try:
            b, w = gpu_board.count_stones()
            logger.info(f"현재 점수 - 흑: {b}, 백: {w}")
        except Exception as e:
            logger.warning(f"점수 계산 오류: {e}")
        
        # 메인 탐색
        start_time = time.time()
        result = None
        best_move = None
        search_stats = {
            'nodes': 0,
            'depth': 0,
            'score': 0,
            'time_ms': 0,
            'is_exact': False,
            'pv': []
        }
        
        try:
            if self.use_neural_net and hasattr(self, 'mcts') and self.mcts:
                logger.info("신경망 기반 MCTS 탐색 시작")
                best_move, mcts_stats = self.get_move_with_neural_net_enhanced(gpu_board)
                search_stats.update(mcts_stats)
            else:
                logger.info("전통적 알파베타 탐색 시작")
                result = self.ultra_iterative_deepening(gpu_board)
                
                if result and result.best_move:
                    best_move = result.best_move
                    search_stats = {
                        'nodes': getattr(result, 'nodes', 0),
                        'depth': getattr(result, 'depth', 0),
                        'score': getattr(result, 'score', 0),
                        'time_ms': getattr(result, 'time_ms', 0),
                        'is_exact': getattr(result, 'is_exact', False),
                        'pv': getattr(result, 'pv', [])
                    }
                else:
                    logger.error("탐색 결과가 None이거나 best_move가 없습니다")
                    # 백업 탐색 시도
                    valid_moves = gpu_board.get_valid_moves(self.color)
                    if valid_moves:
                        best_move = valid_moves[0]
                        logger.warning(f"백업 수 선택: {chr(best_move[1] + ord('a'))}{best_move[0] + 1}")
                    
        except Exception as e:
            logger.error(f"탐색 중 오류 발생: {e}")
            # 긴급 백업
            valid_moves = gpu_board.get_valid_moves(self.color)
            if valid_moves:
                best_move = valid_moves[0]
        
        if not best_move:
            logger.error("수를 찾지 못했습니다!")
            return None
        
        # 상세 통계 출력
        elapsed_time = time.time() - start_time
        elapsed_ms = elapsed_time * 1000
        
        logger.info(f"=== AI 분석 완료 ===")
        logger.info(f"최적 수: {chr(best_move[1] + ord('a'))}{best_move[0] + 1}")
        logger.info(f"평가 점수: {search_stats['score']}")
        logger.info(f"탐색 깊이: {search_stats['depth']}")
        logger.info(f"탐색 노드: {search_stats['nodes']:,}개")
        logger.info(f"소요 시간: {elapsed_ms:.1f}ms ({elapsed_time:.3f}초)")
        
        # NPS 계산 (안전하게)
        if elapsed_time > 0 and search_stats['nodes'] > 0:
            nps = search_stats['nodes'] / elapsed_time
            logger.info(f"초당 노드: {nps:,.0f} NPS")
        else:
            logger.info("초당 노드: 계산 불가")
        
        # TT 통계 (안전하게)
        if search_stats['nodes'] > 0:
            tt_hit_rate = (self.tt_hits / search_stats['nodes']) * 100
            logger.info(f"TT 히트: {self.tt_hits:,}개 ({tt_hit_rate:.1f}%)")
        else:
            logger.info(f"TT 히트: {self.tt_hits:,}개")
        
        logger.info(f"컷오프: {self.cutoffs:,}개")
        logger.info(f"완전 탐색: {self.perfect_searches}회")
        logger.info(f"정확도: {'완전' if search_stats['is_exact'] else '근사'}")
        
        # PV (주요 변화) 출력
        if search_stats['pv'] and len(search_stats['pv']) > 1:
            try:
                pv_str = " ".join([f"{chr(move[1] + ord('a'))}{move[0] + 1}" 
                                for move in search_stats['pv'][:5] if move])
                logger.info(f"주요 변화: {pv_str}")
            except Exception as e:
                logger.warning(f"PV 출력 오류: {e}")
        
        # 메모리 사용량 로깅
        if hasattr(self, 'tt'):
            logger.info(f"TT 크기: {len(self.tt):,}개 엔트리")
        
        # GPU 메모리 상태
        if hasattr(self, 'gpu') and getattr(self.gpu, 'gpu_available', False):
            logger.info(f"GPU 메모리 정리 완료")
        
        logger.info("=" * 40)
        
        return best_move


    def get_move_with_neural_net_enhanced(self, gpu_board):
        """신경망 기반 수 선택 - 통계 포함 버전"""
        if not hasattr(self, 'mcts') or not self.mcts:
            return None, {'nodes': 0, 'depth': 0, 'score': 0}
        
        try:
            # MCTS 탐색 실행
            action_probs = self.mcts.search(gpu_board, self.color)
            
            # MCTS 통계 수집
            mcts_stats = {
                'nodes': getattr(self.mcts, 'search_count', 0),
                'depth': getattr(self.mcts, 'max_depth', 0),
                'score': 0,
                'time_ms': 0,
                'is_exact': False,
                'pv': []
            }
            
            # 가장 높은 확률의 수 선택
            valid_moves = gpu_board.get_valid_moves(self.color)
            if not valid_moves:
                return None, mcts_stats
            
            best_prob = 0
            best_move = valid_moves[0]
            
            for move in valid_moves:
                action_idx = move[0] * 8 + move[1]
                if action_idx < len(action_probs) and action_probs[action_idx] > best_prob:
                    best_prob = action_probs[action_idx]
                    best_move = move
            
            mcts_stats['score'] = best_prob
            return best_move, mcts_stats
            
        except Exception as e:
            logger.error(f"신경망 탐색 오류: {e}")
            valid_moves = gpu_board.get_valid_moves(self.color)
            fallback_move = valid_moves[0] if valid_moves else None
            return fallback_move, {'nodes': 0, 'depth': 0, 'score': 0}
    
    def ultra_iterative_deepening(self, board):
        """반복 심화 탐색 - 통계 보장 버전"""
        try:
            # 기본 결과 객체 생성
            class SearchResult:
                def __init__(self):
                    self.best_move = None
                    self.score = 0
                    self.depth = 0
                    self.nodes = 0
                    self.time_ms = 0
                    self.is_exact = False
                    self.pv = []
            
            result = SearchResult()
            start_time = time.time()
            
            # 유효한 수 확인
            valid_moves = board.get_valid_moves(self.color)
            if not valid_moves:
                logger.warning("유효한 수가 없습니다")
                return result
            
            # 시간 제한 설정
            time_limit = min(5.0, max(0.1, board.get_empty_count() * 0.1))
            max_depth = min(20, board.get_empty_count())
            
            logger.info(f"탐색 제한: 깊이 {max_depth}, 시간 {time_limit:.1f}초")
            
            # 반복 심화 탐색
            for depth in range(1, max_depth + 1):
                if time.time() - start_time > time_limit:
                    logger.info(f"시간 제한으로 깊이 {depth-1}에서 탐색 종료")
                    break
                
                try:
                    # 알파베타 탐색 실행
                    alpha_beta_result = self.alpha_beta_search(
                        board, depth, -float('inf'), float('inf'), True
                    )
                    
                    if alpha_beta_result and alpha_beta_result[1]:  # (score, move)
                        result.score = alpha_beta_result[0]
                        result.best_move = alpha_beta_result[1]
                        result.depth = depth
                        result.nodes = self.nodes_searched
                        result.is_exact = (depth >= board.get_empty_count())
                        
                        logger.debug(f"깊이 {depth}: 점수 {result.score}, 수 {chr(result.best_move[1] + ord('a'))}{result.best_move[0] + 1}")
                    
                except Exception as e:
                    logger.warning(f"깊이 {depth} 탐색 중 오류: {e}")
                    break
            
            # 최종 통계 설정
            result.time_ms = (time.time() - start_time) * 1000
            
            if not result.best_move and valid_moves:
                result.best_move = valid_moves[0]
                logger.warning("탐색 실패, 첫 번째 유효한 수 선택")
            
            return result
            
        except Exception as e:
            logger.error(f"반복 심화 탐색 전체 오류: {e}")
            # 최소한의 결과 반환
            result = SearchResult()
            valid_moves = board.get_valid_moves(self.color)
            if valid_moves:
                result.best_move = valid_moves[0]
            return result
        
    # get_move 메서드 다음에 추가
    def load_model(self, model_path='models/gpu_best_model.pth'):
        """훈련된 모델 로드"""
        if not self.use_neural_net or not self.trainer:
            return

        try:
            if self.trainer.load_model(model_path):
                logger.info("GPU 신경망 모델 로드 완료")
            else:
                logger.warning("GPU 모델 파일을 찾을 수 없음. 랜덤 가중치 사용")
        except Exception as e:
            logger.error(f"GPU 모델 로드 중 오류: {e}")

    def get_move_with_neural_net(self, board):
        """신경망 기반 수 선택"""
        if not self.mcts:
            return self.get_move_traditional(board)
        
        # GPU 보드로 변환
        gpu_board = self._convert_to_gpu_board(board)
        
        # MCTS 탐색
        action_probs = self.mcts.search(gpu_board, self.color)
        
        # 가장 높은 확률의 수 선택
        valid_moves = gpu_board.get_valid_moves(self.color)
        if not valid_moves:
            return None
        
        best_prob = 0
        best_move = valid_moves[0]
        
        for move in valid_moves:
            action_idx = move[0] * 8 + move[1]
            if action_probs[action_idx] > best_prob:
                best_prob = action_probs[action_idx]
                best_move = move
        
        logger.info(f"신경망 AI 수: {chr(best_move[1] + ord('a'))}{best_move[0] + 1}")
        return best_move


    def start_self_play_training(self, iterations=100, games_per_iteration=50):
        """자가 학습 시작"""
        if not self.use_neural_net or not self.trainer:
            logger.error("신경망이 활성화되지 않아 자가 학습을 시작할 수 없습니다")
            return
        
        logger.info(f"GPU 자가 학습 시작: {iterations}회 반복, 반복당 {games_per_iteration}게임")
        
        for iteration in range(iterations):
            logger.info(f"\n=== GPU 자가 학습 반복 {iteration + 1}/{iterations} ===")
            
            # 자가 학습 실행
            self.trainer.train_iteration(
                num_games=games_per_iteration,
                batch_size=64,
                epochs=10
            )
            
            # 주기적 모델 저장
            if (iteration + 1) % 10 == 0:
                model_path = f'models/gpu_checkpoint_{iteration + 1}.pth'
                import os
                os.makedirs('models', exist_ok=True)
                self.trainer.save_model(model_path)
                logger.info(f"체크포인트 저장: {model_path}")
            
            # 통계 출력
            stats = self.trainer.training_stats
            logger.info(f"총 게임: {stats['games_played']}, "
                    f"Policy Loss: {stats['avg_policy_loss']:.4f}, "
                    f"Value Loss: {stats['avg_value_loss']:.4f}")
        
        # 최종 모델 저장
        final_model_path = 'models/gpu_final_model.pth'
        import os
        os.makedirs('models', exist_ok=True)
        self.trainer.save_model(final_model_path)
        logger.info(f"최종 모델 저장: {final_model_path}")

    def get_move_traditional(self, board):
        """기존 방식의 수 선택"""
        # 기존 get_move 로직을 여기로 이동
        # GPU 보드로 변환
        gpu_board = self._convert_to_gpu_board(board)
        
        # 기존 탐색 알고리즘 사용
        result = self.ultra_iterative_deepening(gpu_board)
        
        if result and result.best_move:
            logger.info(f"전통적 AI 수: {chr(result.best_move[1] + ord('a'))}{result.best_move[0] + 1}")
            return result.best_move
        
        return None
        
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
