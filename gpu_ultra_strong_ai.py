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
import datetime
from datetime import datetime

def setup_ai_logger():
    """세션별 AI 로거 설정 - INFO/DEBUG 분리 저장"""
    # 디렉토리 구조 생성
    base_log_dir = "logs/ai_sessions"
    info_dir = os.path.join(base_log_dir, 'info')
    debug_dir = os.path.join(base_log_dir, 'debug')
    
    # 디렉토리 생성
    os.makedirs(info_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)
    
    # 세션별 고유 타임스탬프 생성
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = f"session_{session_timestamp}"
    
    # 로그 파일명들 생성
    info_log_file = os.path.join(info_dir, f"AI_INFO_{session_id}.log")
    debug_log_file = os.path.join(debug_dir, f"AI_DEBUG_{session_id}.log")
    
    logger = logging.getLogger('AI_Logger')
    logger.setLevel(logging.DEBUG)
    
    # 기존 핸들러 제거 (새 세션 시작)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 포맷터 설정
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 1. INFO 전용 핸들러 (INFO, WARNING, ERROR)
    info_handler = logging.FileHandler(info_log_file, mode='w', encoding='utf-8')
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(detailed_formatter)
    logger.addHandler(info_handler)
    
    # 2. DEBUG 전용 핸들러 (DEBUG만)
    debug_handler = logging.FileHandler(debug_log_file, mode='w', encoding='utf-8')
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.addFilter(lambda record: record.levelno == logging.DEBUG)
    debug_handler.setFormatter(detailed_formatter)
    logger.addHandler(debug_handler)
    
    # 3. 콘솔 핸들러 (INFO 레벨 이상만)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(detailed_formatter)
    logger.addHandler(console_handler)
    
    logger.propagate = False
    
    # 세션 시작 로그
    logger.info("=" * 60)
    logger.info(f"🚀 NEW AI SESSION STARTED: {session_id}")
    logger.info(f"📁 INFO Log: {info_log_file}")
    logger.info(f"🔍 DEBUG Log: {debug_log_file}")
    logger.info(f"🕒 Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    
    # 테스트 로그
    logger.debug("🔍 DEBUG logging is working")
    logger.info("ℹ️ INFO logging is working")
    
    return logger

logger = setup_ai_logger()

class GPUManager:
    """
    GPU 메모리 및 연산 관리 클래스 - 조용한 폴백 지원
    """
    
    def __init__(self):
        self.gpu_available = False
        self.backend = 'cpu'
        self.memory_pool = None
        self.fallback_reason = None
        self._initialization_attempted = False
        self._silent_mode = False  # 조용한 모드 플래그
        
        # GPU 사용 가능성 확인
        self._initialize_gpu_backend()
    
    def _initialize_gpu_backend(self):
        """GPU 백엔드 초기화 - 조용한 버전"""
        if self._initialization_attempted:
            return
            
        self._initialization_attempted = True
        global GPU_AVAILABLE, GPU_BACKEND
        
        if GPU_AVAILABLE and GPU_BACKEND == 'cupy':
            try:
                # CuPy 초기화 및 테스트 (조용히)
                import cupy as cp
                self.memory_pool = cp.get_default_memory_pool()
                
                # 간단한 테스트 연산
                test_array = cp.array([1.0, 2.0, 3.0])
                result = cp.sum(test_array)
                _ = result.get()  # CPU로 결과 가져오기 테스트
                
                self.gpu_available = True
                self.backend = 'cupy'
                if not self._silent_mode:
                    logger.info("✅ CuPy GPU backend 초기화 성공")
                
            except Exception as e:
                self.fallback_reason = f"CuPy 초기화 실패: {type(e).__name__}"
                if not self._silent_mode:
                    logger.info(f"💻 GPU 사용 불가, CPU 모드로 진행: {self.fallback_reason}")
                self._try_numba_fallback()
                
        elif GPU_AVAILABLE and GPU_BACKEND == 'numba':
            try:
                from numba import cuda
                if cuda.is_available():
                    cuda.select_device(0)
                    self.gpu_available = True
                    self.backend = 'numba'
                    if not self._silent_mode:
                        logger.info("✅ Numba CUDA backend 초기화 성공")
                else:
                    self.fallback_reason = "Numba CUDA 디바이스 없음"
                    self._fallback_to_cpu()
            except Exception as e:
                self.fallback_reason = f"Numba 초기화 실패: {type(e).__name__}"
                self._fallback_to_cpu()
        else:
            self.fallback_reason = "GPU 백엔드 전역적으로 사용 불가"
            self._fallback_to_cpu()
    
    def _try_numba_fallback(self):
        """CuPy 실패시 Numba로 폴백 시도 (조용히)"""
        try:
            from numba import cuda
            if cuda.is_available():
                cuda.select_device(0)
                self.gpu_available = True
                self.backend = 'numba'
                if not self._silent_mode:
                    logger.info("✅ Numba CUDA 폴백 성공")
            else:
                self._fallback_to_cpu()
        except Exception as e:
            self.fallback_reason += f" + Numba 폴백 실패: {type(e).__name__}"
            self._fallback_to_cpu()
    
    def _fallback_to_cpu(self):
        """CPU로 완전 폴백 (조용히)"""
        self.gpu_available = False
        self.backend = 'cpu'
        if not self._silent_mode:
            logger.info(f"💻 CPU 모드 사용: {self.fallback_reason}")
    
    def enable_silent_mode(self):
        """조용한 모드 활성화 (오류 메시지 최소화)"""
        self._silent_mode = True
    
    def to_gpu(self, array):
        """
        배열을 GPU로 이동 (조용한 폴백)
        """
        if not self.gpu_available:
            return np.array(array, dtype=np.float32)
        
        try:
            if self.backend == 'cupy':
                import cupy as cp
                return cp.asarray(array, dtype=cp.float32)
            else:
                return np.array(array, dtype=np.float32)
        except Exception as e:
            # 조용히 폴백
            if not getattr(self, '_to_gpu_failed_logged', False):
                logger.debug(f"GPU 배열 이동 실패, CPU 사용: {type(e).__name__}")
                self._to_gpu_failed_logged = True
            self.gpu_available = False
            self.backend = 'cpu'
            return np.array(array, dtype=np.float32)
    
    def to_cpu(self, array):
        """
        GPU 배열을 CPU로 이동 (조용한 처리)
        """
        try:
            if self.backend == 'cupy' and hasattr(array, 'get'):
                return array.get()
            return np.asarray(array, dtype=np.float32)
        except Exception as e:
            # 조용히 처리
            try:
                return np.array(array, dtype=np.float32)
            except:
                return np.zeros((8, 8), dtype=np.float32)
    
    def clear_memory(self):
        """GPU 메모리 정리 (조용한 처리)"""
        if not self.gpu_available:
            return
            
        try:
            if self.backend == 'cupy' and self.memory_pool:
                self.memory_pool.free_all_blocks()
            elif self.backend == 'numba':
                from numba import cuda
                cuda.current_context().memory_manager.deallocations.clear()
        except Exception:
            # 조용히 무시
            pass

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
# gpu_ultra_strong_ai.py의 GPUBoard 클래스를 이것으로 교체하세요

class GPUBoard:
    """
    GPU 가속 오델로 보드 클래스 - 안전한 폴백 지원
    보드 연산을 GPU에서 처리하되, 실패시 CPU로 안전하게 폴백
    """
    
    def __init__(self, gpu_manager: GPUManager):
        self.gpu = gpu_manager
        self.board = None
        self.move_history = []
        self.cpu_fallback_active = False
        
        # 안전한 보드 초기화
        self._safe_initialize_board()
        logger.debug(f"GPU Board 초기화: GPU={self.gpu.gpu_available}, Fallback={self.cpu_fallback_active}")
    
    def _safe_initialize_board(self):
        """
        안전한 초기 보드 상태 설정
        GPU 실패시 CPU로 자동 폴백
        """
        try:
            # 먼저 CPU에서 보드 생성
            board_cpu = np.zeros((8, 8), dtype=np.int8)
            board_cpu[3, 3] = WHITE
            board_cpu[3, 4] = BLACK
            board_cpu[4, 3] = BLACK
            board_cpu[4, 4] = WHITE
            
            # GPU 사용 가능하면 GPU로 이동 시도
            if self.gpu.gpu_available:
                try:
                    self.board = self.gpu.to_gpu(board_cpu)
                    logger.debug("보드를 GPU로 초기화 성공")
                except Exception as e:
                    logger.warning(f"GPU 보드 초기화 실패, CPU 사용: {e}")
                    self.board = board_cpu
                    self.cpu_fallback_active = True
                    self.gpu.gpu_available = False
            else:
                self.board = board_cpu
                self.cpu_fallback_active = True
                
        except Exception as e:
            logger.error(f"보드 초기화 완전 실패: {e}")
            # 최후의 수단: 기본 numpy 배열
            self.board = np.array([
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, WHITE, BLACK, 0, 0, 0],
                [0, 0, 0, BLACK, WHITE, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0]
            ], dtype=np.int8)
            self.cpu_fallback_active = True
    
    def copy(self):
        """보드 깊은 복사 - 안전한 버전"""
        try:
            new_board = GPUBoard(self.gpu)
            
            # 현재 보드 상태 복사
            if self.cpu_fallback_active or not self.gpu.gpu_available:
                # CPU 모드에서 복사
                board_data = self.get_board_array()
                new_board.board = np.array(board_data, dtype=np.int8)
                new_board.cpu_fallback_active = True
            else:
                # GPU 모드에서 복사 시도
                try:
                    board_cpu = self.gpu.to_cpu(self.board)
                    new_board.board = self.gpu.to_gpu(board_cpu.copy())
                except Exception as e:
                    logger.warning(f"GPU 복사 실패, CPU로 폴백: {e}")
                    board_data = self.get_board_array()
                    new_board.board = np.array(board_data, dtype=np.int8)
                    new_board.cpu_fallback_active = True
            
            new_board.move_history = self.move_history.copy()
            return new_board
            
        except Exception as e:
            logger.error(f"보드 복사 실패: {e}")
            # 최후의 수단: 새 보드 반환
            return GPUBoard(self.gpu)
    
    def get_board_array(self):
        """보드 배열 반환 - 안전한 버전"""
        try:
            if self.cpu_fallback_active or not self.gpu.gpu_available:
                if isinstance(self.board, np.ndarray):
                    return self.board.tolist()
                else:
                    return self.board
            else:
                # GPU에서 CPU로 안전하게 이동
                board_cpu = self.gpu.to_cpu(self.board)
                return board_cpu.tolist()
        except Exception as e:
            logger.warning(f"보드 배열 반환 실패: {e}")
            # 기본 초기 보드 반환
            return [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, WHITE, BLACK, 0, 0, 0],
                [0, 0, 0, BLACK, WHITE, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0]
            ]
    
    def set_board_array(self, board_array):
        """보드 배열 설정 - 안전한 버전"""
        try:
            board_np = np.array(board_array, dtype=np.int8)
            
            if self.cpu_fallback_active or not self.gpu.gpu_available:
                self.board = board_np
            else:
                try:
                    self.board = self.gpu.to_gpu(board_np)
                except Exception as e:
                    logger.warning(f"GPU 보드 설정 실패, CPU 사용: {e}")
                    self.board = board_np
                    self.cpu_fallback_active = True
                    
        except Exception as e:
            logger.error(f"보드 배열 설정 실패: {e}")
    
    def _get_board_cpu(self):
        """CPU 보드 배열 얻기 (내부 사용)"""
        try:
            if self.cpu_fallback_active or not self.gpu.gpu_available:
                if isinstance(self.board, np.ndarray):
                    return self.board
                else:
                    return np.array(self.board, dtype=np.int8)
            else:
                return self.gpu.to_cpu(self.board)
        except Exception as e:
            logger.warning(f"CPU 보드 획득 실패: {e}")
            # 기본 보드 반환
            board = np.zeros((8, 8), dtype=np.int8)
            board[3, 3] = WHITE
            board[3, 4] = BLACK
            board[4, 3] = BLACK
            board[4, 4] = WHITE
            return board
    
    def is_game_over(self):
        """게임 종료 여부 확인 - 안전한 버전"""
        try:
            black_moves = self.get_valid_moves(BLACK)
            white_moves = self.get_valid_moves(WHITE)
            return len(black_moves) == 0 and len(white_moves) == 0
        except Exception as e:
            logger.warning(f"게임 종료 확인 실패: {e}")
            return False
    
    def get_winner(self):
        """승자 반환 - 안전한 버전"""
        try:
            if not self.is_game_over():
                return None
                
            black_count, white_count = self.count_stones()
            if black_count > white_count:
                return BLACK
            elif white_count > black_count:
                return WHITE
            else:
                return 0  # 무승부
        except Exception as e:
            logger.warning(f"승자 확인 실패: {e}")
            return None
        
    def is_valid_move(self, x, y, color):
        """
        유효한 수인지 확인 - 안전한 버전
        """
        try:
            if not (0 <= x < 8 and 0 <= y < 8):
                return False
            
            board_cpu = self._get_board_cpu()
            if board_cpu[x, y] != EMPTY:
                return False
            
            directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
            
            for dx, dy in directions:
                if self._check_direction(board_cpu, x, y, dx, dy, color):
                    return True
            return False
            
        except Exception as e:
            logger.warning(f"수 유효성 검사 실패: {e}")
            return False
    
    def _check_direction(self, board_cpu, x, y, dx, dy, color):
        """
        특정 방향으로 뒤집을 수 있는지 확인 - 안전한 버전
        """
        try:
            nx, ny = x + dx, y + dy
            
            # 첫 번째 인접 칸이 상대방 돌이어야 함
            if not (0 <= nx < 8 and 0 <= ny < 8) or board_cpu[nx, ny] != opponent(color):
                return False
            
            # 연속된 상대방 돌들 확인
            found_opponent = False
            while 0 <= nx < 8 and 0 <= ny < 8:
                cell_value = board_cpu[nx, ny]
                
                if cell_value == EMPTY:
                    return False
                elif cell_value == opponent(color):
                    found_opponent = True
                    nx += dx
                    ny += dy
                elif cell_value == color:
                    return found_opponent  # 상대방 돌이 있었고 내 돌로 끝남
                else:
                    return False
            
            return False
        except Exception as e:
            logger.warning(f"방향 확인 실패: {e}")
            return False
    
    def get_valid_moves(self, color):
        """
        유효한 수 목록 반환 - 안전한 버전
        """
        try:
            moves = []
            board_cpu = self._get_board_cpu()
            
            # 모든 빈 칸을 확인
            for x in range(8):
                for y in range(8):
                    if board_cpu[x, y] == EMPTY:
                        if self._is_valid_move_fast(board_cpu, x, y, color):
                            moves.append((x, y))
            
            logger.debug(f"색상 {color}에 대해 {len(moves)}개의 유효한 수 발견")
            return moves
            
        except Exception as e:
            logger.warning(f"유효한 수 찾기 실패: {e}")
            return []
    
    def _is_valid_move_fast(self, board_cpu, x, y, color):
        """
        빠른 유효 수 검증 - 안전한 버전
        """
        try:
            directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
            
            for dx, dy in directions:
                if self._check_direction(board_cpu, x, y, dx, dy, color):
                    return True
            return False
        except Exception as e:
            logger.warning(f"빠른 수 검증 실패: {e}")
            return False
    
    def apply_move(self, x, y, color):
        """
        수를 두고 새로운 보드 반환 - 안전한 버전
        """
        try:
            # 입력 검증
            if not (0 <= x < 8 and 0 <= y < 8):
                logger.warning(f"좌표 범위 초과: ({x}, {y})")
                return self.copy()
            
            # 유효한 수인지 검증
            if not self.is_valid_move(x, y, color):
                logger.warning(f"유효하지 않은 수: ({x}, {y}) for color {color}")
                return self.copy()
            
            new_board = self.copy()
            board_cpu = new_board._get_board_cpu().copy()  # 복사본에서 작업
            
            # 수 두기
            board_cpu[x, y] = color
            flipped = []
            
            directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
            
            for dx, dy in directions:
                line_flipped = new_board._flip_direction(board_cpu, x, y, dx, dy, color)
                flipped.extend(line_flipped)
            
            # 보드 업데이트
            new_board.set_board_array(board_cpu.tolist())
            new_board.move_history.append((x, y, color, flipped))
            
            logger.debug(f"수 적용 완료: ({x}, {y}), {len(flipped)}개 뒤집힘")
            return new_board
            
        except Exception as e:
            logger.error(f"수 적용 실패: {e}")
            return self.copy()
    
    def _flip_direction(self, board_cpu, x, y, dx, dy, color):
        """
        특정 방향의 돌들을 뒤집기 - 안전한 버전
        """
        try:
            flipped = []
            nx, ny = x + dx, y + dy
            
            # 상대방 돌들 수집
            while (0 <= nx < 8 and 0 <= ny < 8 and 
                   board_cpu[nx, ny] == opponent(color)):
                flipped.append((nx, ny))
                nx += dx
                ny += dy
            
            # 내 돌로 끝나는지 확인
            if (0 <= nx < 8 and 0 <= ny < 8 and 
                board_cpu[nx, ny] == color and flipped):
                # 실제로 뒤집기
                for fx, fy in flipped:
                    board_cpu[fx, fy] = color
                return flipped
            
            # 유효하지 않은 방향이면 빈 리스트 반환
            return []
            
        except Exception as e:
            logger.warning(f"돌 뒤집기 실패: {e}")
            return []
    
    def count_stones(self):
        """
        돌 개수 세기 - 안전한 버전
        """
        try:
            board_cpu = self._get_board_cpu()
            black_count = np.sum(board_cpu == BLACK)
            white_count = np.sum(board_cpu == WHITE)
            return int(black_count), int(white_count)
        except Exception as e:
            logger.warning(f"돌 개수 세기 실패: {e}")
            return 2, 2  # 기본값
    
    def get_empty_count(self):
        """
        빈 칸 개수 반환 - 안전한 버전
        """
        try:
            board_cpu = self._get_board_cpu()
            return int(np.sum(board_cpu == EMPTY))
        except Exception as e:
            logger.warning(f"빈 칸 개수 세기 실패: {e}")
            return 60  # 기본값
    
    def get_frontier_count(self, color):
        """
        프론티어 디스크 개수 (인접한 빈 칸이 있는 돌) - 안전한 버전
        """
        try:
            count = 0
            board_cpu = self._get_board_cpu()
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
        except Exception as e:
            logger.warning(f"프론티어 개수 세기 실패: {e}")
            return 0
    
    def get_status_info(self):
        """보드 상태 정보 반환 (디버깅용)"""
        return {
            'gpu_available': self.gpu.gpu_available,
            'cpu_fallback_active': self.cpu_fallback_active,
            'backend': self.gpu.backend,
            'board_type': type(self.board).__name__,
            'move_count': len(self.move_history)
        }

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
    
    def evaluate_position_gpu(self, board, color: int):
        """
        GPU 가속 위치 평가 함수 - 조용한 폴백 지원
        """
        try:
            if board.get_empty_count() == 0:
                return self._evaluate_endgame(board, color)
            
            empty_count = board.get_empty_count()
            
            # GPU 사용 가능하고 아직 폴백하지 않은 경우에만 GPU 시도
            if (self.gpu.gpu_available and 
                self.evaluation_tables_loaded and 
                not getattr(self, '_gpu_failed_once', False)):
                
                try:
                    score = self._evaluate_with_gpu(board, color, empty_count)
                    logger.debug(f"GPU 위치 평가 완료: 점수={score:.1f}, 빈칸={empty_count}")
                    return int(score)
                    
                except Exception as gpu_error:
                    # GPU 실패를 한 번만 로그에 기록하고, 이후엔 조용히 CPU 사용
                    if not getattr(self, '_gpu_failed_once', False):
                        logger.warning(f"GPU 평가 실패, CPU로 영구 전환: {type(gpu_error).__name__}")
                        self._gpu_failed_once = True
                        self.gpu.gpu_available = False
                    
                    # 조용히 CPU로 폴백
                    score = self._evaluate_with_cpu(board, color, empty_count)
            else:
                # 이미 폴백 상태이거나 GPU 사용 불가
                score = self._evaluate_with_cpu(board, color, empty_count)
            
            logger.debug(f"CPU 위치 평가 완료: 점수={score:.1f}, 빈칸={empty_count}")
            return int(score)
            
        except Exception as e:
            # 전체 평가 실패시에만 오류 로그
            logger.debug(f"평가 함수 오류 (복구됨): {e}")
            # 최후의 수단: 돌 개수 차이만 반환
            try:
                b, w = board.count_stones()
                diff = (b - w) if color == BLACK else (w - b)
                return diff * 100
            except:
                return 0
    
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

# ... [나머지 클래스들은 길이 제한으로 인해 별도 아티팩트로 분리]
# 이 부분을 gpu_ultra_strong_ai.py의 GPUEvaluator 클래스 다음에 추가하세요

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
        
        return self.children[best_action] if best_action else None
    
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
                if node is None:
                    break
            
            if node is None:
                continue
            
            # Expansion & Evaluation
            valid_moves = node.board.get_valid_moves(node.color)
            if not valid_moves:
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
        
        try:
            board_tensor = self.board_to_tensor(board, color)
            
            with torch.no_grad():
                policy_logits, value = self.neural_net(board_tensor.unsqueeze(0))
                policy = torch.exp(policy_logits).squeeze().cpu().numpy()
                value = value.item()
            
            return policy, value
        except Exception as e:
            logger.warning(f"신경망 예측 오류: {e}")
            return np.ones(64) / 64, 0.0
    
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
            if best_move:
                board = board.apply_move(*best_move, current_player)
                current_player = opponent(current_player)
                move_count += 1
            else:
                break
        
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
        if not valid_moves:
            return None
            
        if temperature == 0:
            best_prob = 0
            best_move = valid_moves[0]
            
            for move in valid_moves:
                action_idx = move[0] * 8 + move[1]
                if action_idx < len(action_probs) and action_probs[action_idx] > best_prob:
                    best_prob = action_probs[action_idx]
                    best_move = move
            
            return best_move
        else:
            move_probs = []
            for move in valid_moves:
                action_idx = move[0] * 8 + move[1]
                if action_idx < len(action_probs):
                    prob = action_probs[action_idx] ** (1 / temperature)
                else:
                    prob = 0.01  # 기본 확률
                move_probs.append(prob)
            
            total_prob = sum(move_probs)
            if total_prob > 0:
                move_probs = [p / total_prob for p in move_probs]
                try:
                    return np.random.choice(valid_moves, p=move_probs)
                except:
                    return random.choice(valid_moves)
            else:
                return random.choice(valid_moves)
    
    def train_iteration(self, num_games=50, batch_size=32, epochs=10):
        """훈련 반복"""
        logger.info(f"GPU 자가 대국 {num_games}게임 시작...")
        
        # 자가 대국으로 데이터 생성
        for i in range(num_games):
            temperature = max(0.1, 1.0 - (i / num_games) * 0.9)
            
            try:
                game_data = self.self_play_game(temperature=temperature)
                self.training_data.extend(game_data)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"  {i + 1}/{num_games} 게임 완료")
            except Exception as e:
                logger.warning(f"게임 {i+1} 중 오류: {e}")
                continue
        
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
                
                try:
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
                
                except Exception as e:
                    logger.warning(f"배치 훈련 중 오류: {e}")
                    continue
            
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
        try:
            torch.save({
                'model_state_dict': self.neural_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'training_stats': self.training_stats
            }, filepath)
            logger.info(f"GPU 모델 저장 완료: {filepath}")
        except Exception as e:
            logger.error(f"모델 저장 실패: {e}")
    
    def load_model(self, filepath):
        """모델 로드"""
        try:
            if os.path.exists(filepath):
                checkpoint = torch.load(filepath, map_location=self.device)
                self.neural_net.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.training_stats = checkpoint.get('training_stats', self.training_stats)
                logger.info(f"GPU 모델 로드 완료: {filepath}")
                return True
        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
        return False

class UltraStrongAI:
    """
    최강 오델로 AI - GPU 가속 버전 (수정된)
    탐색 알고리즘과 평가 함수를 GPU에서 병렬 처리
    """
    
    def __init__(self, color, difficulty='ultra', time_limit=10.0, use_neural_net=True):
        self.color = color
        self.difficulty = difficulty
        self.time_limit = time_limit
        
        # 통계 변수들을 먼저 초기화 (속성 오류 방지)
        self.nodes_searched = 0
        self.tt_hits = 0
        self.cutoffs = 0
        self.perfect_searches = 0
        self.tt_age = 0
        
        # GPU 관리자 및 평가자 초기화
        self.gpu = GPUManager()
        self.gpu.enable_silent_mode()  # 조용한 모드 활성화
        self.evaluator = GPUEvaluator(self.gpu)
        
        # 난이도별 설정
        self._configure_difficulty(difficulty)
        
        # 강화된 Transposition Table
        self.tt = {}
        self.max_tt_size = 1000000
        
        # 고급 휴리스틱들
        self.killer_moves = defaultdict(list)
        self.history_table = defaultdict(int)
        self.counter_moves = defaultdict(list)
        
        # 신경망 관련 설정
        self.use_neural_net = use_neural_net and TORCH_AVAILABLE
        self.continuous_learning = True
        self.learning_buffer = deque(maxlen=10000)
        
        # 자동 학습 스케줄러
        self.games_since_training = 0
        self.training_interval = 10  # 10게임마다 학습
        
        # 신경망 초기화 (안전하게)
        self.neural_net = None
        self.mcts = None
        self.trainer = None
        
        if self.use_neural_net:
            try:
                # 지연 로딩으로 순환 참조 방지
                self._initialize_neural_components()
            except Exception as e:
                logger.error(f"신경망 초기화 실패: {e}")
                self.use_neural_net = False
        
        logger.info(f"UltraStrongAI initialized - Color: {color}, Difficulty: {difficulty}, GPU: {self.gpu.gpu_available}, Neural: {self.use_neural_net}")
    
    def _initialize_neural_components(self):
        """신경망 컴포넌트 지연 초기화"""
        try:
            # 신경망 컴포넌트들을 안전하게 초기화
            from gpu_ultra_strong_ai import GPUOthelloNet, GPUAlphaZeroMCTS, GPUSelfPlayTrainer
            
            self.neural_net = GPUOthelloNet().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            self.mcts = GPUAlphaZeroMCTS(self.neural_net, self.gpu, num_simulations=800)
            self.trainer = GPUSelfPlayTrainer(self.neural_net, self.gpu)
            self.load_model()
            logger.info("Neural network components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize neural network: {e}")
            self.use_neural_net = False
    
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
    

# UltraStrongAI 클래스의 get_move 메서드를 이것으로 교체하세요

    def get_move(self, board):
        """최고의 수 반환 - 향상된 로깅 버전"""
        # 안전한 초기화
        self._ensure_stats_initialized()
        
        # 통계 초기화
        self.nodes_searched = 0
        self.tt_hits = 0
        self.cutoffs = 0
        self.perfect_searches = 0
        self.tt_age += 1
        
        start_time = time.time()
        
        try:
            # GPU 보드로 안전하게 변환
            gpu_board = self._safe_convert_to_gpu_board(board)
            if gpu_board is None:
                logger.error("GPU 보드 변환 실패")
                return self._emergency_move_selection(board)
            
            logger.info(f"🤖 === AI 분석 시작 (색상: {'흑' if self.color == BLACK else '백'}) ===")
            logger.info(f"📊 빈 칸 수: {gpu_board.get_empty_count()}")
            logger.info(f"🎯 난이도: {self.difficulty}, 시간제한: {self.time_limit}초")
            
            # 유효한 수 먼저 확인
            valid_moves = gpu_board.get_valid_moves(self.color)
            if not valid_moves:
                logger.warning("❌ AI에게 유효한 수가 없습니다")
                return None
            
            logger.info(f"📋 유효한 수: {len(valid_moves)}개 - {[chr(m[1] + ord('a')) + str(m[0] + 1) for m in valid_moves]}")
            
            # 한 수만 있으면 바로 반환
            if len(valid_moves) == 1:
                logger.info(f"✅ 유일한 수 선택: {chr(valid_moves[0][1] + ord('a'))}{valid_moves[0][0] + 1}")
                return valid_moves[0]
            
            # 메인 탐색
            best_move = None
            search_stats = {'nodes': 0, 'depth': 0, 'score': 0, 'time_ms': 0}
            
            try:
                if self.use_neural_net and self._is_neural_net_ready():
                    logger.info("🧠 신경망 기반 탐색 시작...")
                    best_move, search_stats = self._safe_neural_net_search(gpu_board)
                else:
                    logger.info("⚙️ 전통적 알파베타 탐색 시작...")
                    best_move, search_stats = self._safe_traditional_search(gpu_board)
                    
            except Exception as search_error:
                logger.error(f"❌ 주 탐색 실패: {search_error}")
                best_move = self._emergency_move_selection(board)
            
            # 결과 검증 - 강화된 버전
            if not self._validate_move(gpu_board, best_move):
                logger.warning("⚠️ 선택된 수가 유효하지 않음, 첫 번째 유효한 수 선택")
                best_move = valid_moves[0] if valid_moves else None
            
            # 최종 검증
            if best_move and not gpu_board.is_valid_move(best_move[0], best_move[1], self.color):
                logger.error(f"💥 최종 검증 실패: {best_move}")
                best_move = valid_moves[0] if valid_moves else None
            
            # 통계 출력
            self._log_search_results(best_move, search_stats, start_time)
            
            if best_move:
                logger.info(f"🎯 최종 선택: {chr(best_move[1] + ord('a'))}{best_move[0] + 1}")
            else:
                logger.error("❌ 최종 수 선택 실패")
            
            return best_move
            
        except Exception as e:
            logger.error(f"💥 get_move 전체 실패: {e}")
            return self._emergency_move_selection(board)

    def _log_search_results(self, best_move, search_stats, start_time):
        """탐색 결과 로깅 - 향상된 버전"""
        try:
            elapsed_time = time.time() - start_time
            elapsed_ms = elapsed_time * 1000
            
            logger.info(f"📈 === AI 분석 완료 ===")
            if best_move:
                logger.info(f"🎯 최적 수: {chr(best_move[1] + ord('a'))}{best_move[0] + 1}")
            else:
                logger.info("❌ 최적 수: 없음")
            
            logger.info(f"📊 평가 점수: {search_stats.get('score', 0)}")
            logger.info(f"🔍 탐색 깊이: {search_stats.get('depth', 0)}")
            logger.info(f"🌳 탐색 노드: {search_stats.get('nodes', 0):,}개")
            logger.info(f"⏱️ 소요 시간: {elapsed_ms:.1f}ms")
            
            # 추가 통계
            if hasattr(self, 'tt_hits') and self.tt_hits > 0:
                logger.info(f"💾 TT 히트: {self.tt_hits}")
            if hasattr(self, 'cutoffs') and self.cutoffs > 0:
                logger.info(f"✂️ 가지치기: {self.cutoffs}")
            
            # NPS 계산
            if elapsed_time > 0 and search_stats.get('nodes', 0) > 0:
                nps = search_stats['nodes'] / elapsed_time
                logger.info(f"🚀 초당 노드: {nps:,.0f} NPS")
            
            logger.info("=" * 50)
            
        except Exception as e:
            logger.debug(f"결과 로깅 실패: {e}")

    def ultra_iterative_deepening(self, board):
        """반복 심화 탐색 - 향상된 로깅 버전"""
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
            
            # 한 수만 있으면 바로 반환
            if len(valid_moves) == 1:
                result.best_move = valid_moves[0]
                result.depth = 1
                result.nodes = 1
                logger.info(f"유일한 수 선택: {result.best_move}")
                return result
            
            # 시간 제한 설정 (안전한 범위로)
            time_limit = min(self.time_limit, max(0.5, board.get_empty_count() * 0.2))
            max_depth = min(self.max_depth, board.get_empty_count())
            
            logger.info(f"🔍 탐색 설정: 최대깊이={max_depth}, 시간제한={time_limit:.1f}초")
            
            # 반복 심화 탐색
            for depth in range(1, max_depth + 1):
                depth_start_time = time.time()
                
                if time.time() - start_time > time_limit:
                    logger.info(f"⏰ 시간 제한으로 깊이 {depth-1}에서 탐색 종료")
                    break
                
                try:
                    logger.debug(f"🔍 깊이 {depth} 탐색 시작...")
                    
                    # 알파베타 탐색 실행 (시간 제한 포함)
                    end_time = start_time + time_limit
                    score, move = self.gpu_negamax(
                        board, depth, -float('inf'), float('inf'), True, end_time
                    )
                    
                    depth_time = time.time() - depth_start_time
                    
                    if move and board.is_valid_move(move[0], move[1], self.color):
                        result.score = score
                        result.best_move = move
                        result.depth = depth
                        result.nodes = self.nodes_searched
                        result.is_exact = (depth >= board.get_empty_count())
                        
                        logger.info(f"📊 깊이 {depth}: 점수={result.score}, 수={chr(result.best_move[1] + ord('a'))}{result.best_move[0] + 1}, 시간={depth_time:.2f}s, 노드={self.nodes_searched}")
                    else:
                        logger.warning(f"⚠️ 깊이 {depth}에서 유효하지 않은 수 반환: {move}")
                        break
                    
                except Exception as e:
                    logger.warning(f"❌ 깊이 {depth} 탐색 중 오류: {e}")
                    break
            
            # 최종 통계 설정
            result.time_ms = (time.time() - start_time) * 1000
            
            # 결과가 없으면 첫 번째 유효한 수 선택
            if not result.best_move and valid_moves:
                result.best_move = valid_moves[0]
                logger.warning("⚠️ 탐색 실패, 첫 번째 유효한 수 선택")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 반복 심화 탐색 전체 오류: {e}")
            # 최소한의 결과 반환
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
            valid_moves = board.get_valid_moves(self.color)
            if valid_moves:
                result.best_move = valid_moves[0]
            return result
    
    def _ensure_stats_initialized(self):
        """통계 변수 안전 초기화"""
        required_stats = ['nodes_searched', 'tt_hits', 'cutoffs', 'perfect_searches', 'tt_age']
        for stat in required_stats:
            if not hasattr(self, stat):
                setattr(self, stat, 0)
    
    def _safe_convert_to_gpu_board(self, board):
        """안전한 GPU 보드 변환 - 강화된 버전"""
        try:
            if isinstance(board, GPUBoard):
                return board
            
            gpu_board = GPUBoard(self.gpu)
            
            # 보드 데이터 복사
            if hasattr(board, 'board'):
                if isinstance(board.board, list):
                    board_array = np.array(board.board, dtype=np.int8)
                elif hasattr(board.board, 'tolist'):
                    # GPU 배열인 경우
                    board_array = np.array(board.board.tolist(), dtype=np.int8)
                else:
                    board_array = np.array(board.board, dtype=np.int8)
                
                # 보드 크기 검증
                if board_array.shape != (8, 8):
                    logger.error(f"잘못된 보드 크기: {board_array.shape}")
                    return None
                    
                gpu_board.board = gpu_board.gpu.to_gpu(board_array)
            
            # 히스토리 복사
            if hasattr(board, 'move_history'):
                gpu_board.move_history = board.move_history.copy()
            
            return gpu_board
            
        except Exception as e:
            logger.error(f"GPU 보드 변환 실패: {e}")
            return None
    
    def _is_neural_net_ready(self):
        """신경망 준비 상태 확인"""
        return (hasattr(self, 'mcts') and self.mcts is not None and
                hasattr(self, 'neural_net') and self.neural_net is not None)
    
    def _safe_neural_net_search(self, gpu_board):
        """안전한 신경망 탐색"""
        try:
            action_probs = self.mcts.search(gpu_board, self.color)
            
            valid_moves = gpu_board.get_valid_moves(self.color)
            if not valid_moves:
                return None, {'nodes': 0, 'depth': 0, 'score': 0}
            
            # 최고 확률 수 선택
            best_prob = 0
            best_move = valid_moves[0]
            
            for move in valid_moves:
                action_idx = move[0] * 8 + move[1]
                if action_idx < len(action_probs) and action_probs[action_idx] > best_prob:
                    best_prob = action_probs[action_idx]
                    best_move = move
            
            stats = {
                'nodes': getattr(self.mcts, 'num_simulations', 800),
                'depth': 0,  # MCTS는 가변 깊이
                'score': best_prob,
                'time_ms': 0
            }
            
            return best_move, stats
            
        except Exception as e:
            logger.error(f"신경망 탐색 실패: {e}")
            return None, {'nodes': 0, 'depth': 0, 'score': 0}
    
    def _safe_traditional_search(self, gpu_board):
        """안전한 전통적 탐색 - 개선된 버전"""
        try:
            # 반복 심화 탐색 실행
            result = self.ultra_iterative_deepening(gpu_board)
            
            if result and result.best_move:
                # 선택된 수가 유효한지 한번 더 검증
                if gpu_board.is_valid_move(result.best_move[0], result.best_move[1], self.color):
                    stats = {
                        'nodes': getattr(result, 'nodes', self.nodes_searched),
                        'depth': getattr(result, 'depth', 0),
                        'score': getattr(result, 'score', 0),
                        'time_ms': getattr(result, 'time_ms', 0)
                    }
                    return result.best_move, stats
                else:
                    logger.warning(f"탐색 결과가 유효하지 않음: {result.best_move}")
                    return None, {'nodes': 0, 'depth': 0, 'score': 0}
            else:
                return None, {'nodes': 0, 'depth': 0, 'score': 0}
                
        except Exception as e:
            logger.error(f"전통적 탐색 실패: {e}")
            return None, {'nodes': 0, 'depth': 0, 'score': 0}
    
    def _emergency_move_selection(self, board):
        """긴급 수 선택 (모든 다른 방법 실패시) - 강화된 버전"""
        try:
            # GPU 보드 시도
            if hasattr(self, 'gpu') and self.gpu:
                gpu_board = self._safe_convert_to_gpu_board(board)
                if gpu_board:
                    valid_moves = gpu_board.get_valid_moves(self.color)
                    if valid_moves:
                        # 가장 안전한 수 선택 (코너 > 모서리 > 중앙)
                        for move in valid_moves:
                            if move in CORNERS:
                                logger.info(f"긴급 선택: 코너 수 {move}")
                                return move
                        
                        # 모서리 수 선택
                        for move in valid_moves:
                            x, y = move
                            if x == 0 or x == 7 or y == 0 or y == 7:
                                logger.info(f"긴급 선택: 모서리 수 {move}")
                                return move
                        
                        # 첫 번째 유효한 수
                        logger.info(f"긴급 선택: 첫 번째 유효한 수 {valid_moves[0]}")
                        return valid_moves[0]
            
            # 일반 보드에서 유효한 수 찾기
            if hasattr(board, 'get_valid_moves'):
                valid_moves = board.get_valid_moves(self.color)
                if valid_moves:
                    return valid_moves[0]
            
            logger.error("유효한 수를 전혀 찾을 수 없음")
            return None
            
        except Exception as e:
            logger.error(f"긴급 수 선택도 실패: {e}")
            return None
    
    def _validate_move(self, gpu_board, move):
        """수 유효성 검증 - 강화된 버전"""
        try:
            if not move or len(move) != 2:
                return False
            
            x, y = move
            if not (0 <= x < 8 and 0 <= y < 8):
                return False
            
            # GPU 보드에서 직접 검증
            return gpu_board.is_valid_move(x, y, self.color)
            
        except Exception as e:
            logger.debug(f"수 검증 실패: {e}")
            return False
    
    def _log_search_results(self, best_move, search_stats, start_time):
        """탐색 결과 로깅"""
        try:
            elapsed_time = time.time() - start_time
            elapsed_ms = elapsed_time * 1000
            
            logger.info(f"=== AI 분석 완료 ===")
            if best_move:
                logger.info(f"최적 수: {chr(best_move[1] + ord('a'))}{best_move[0] + 1}")
            else:
                logger.info("최적 수: 없음")
            
            logger.info(f"평가 점수: {search_stats['score']}")
            logger.info(f"탐색 깊이: {search_stats['depth']}")
            logger.info(f"탐색 노드: {search_stats['nodes']:,}개")
            logger.info(f"소요 시간: {elapsed_ms:.1f}ms")
            
            # NPS 계산
            if elapsed_time > 0 and search_stats['nodes'] > 0:
                nps = search_stats['nodes'] / elapsed_time
                logger.info(f"초당 노드: {nps:,.0f} NPS")
            
            logger.info("=" * 40)
            
        except Exception as e:
            logger.debug(f"결과 로깅 실패: {e}")
    
    def ultra_iterative_deepening(self, board):
        """반복 심화 탐색 - 강화된 안전성 버전"""
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
            
            # 한 수만 있으면 바로 반환
            if len(valid_moves) == 1:
                result.best_move = valid_moves[0]
                result.depth = 1
                result.nodes = 1
                logger.info(f"유일한 수 선택: {result.best_move}")
                return result
            
            # 시간 제한 설정 (안전한 범위로)
            time_limit = min(self.time_limit, max(0.5, board.get_empty_count() * 0.2))
            max_depth = min(self.max_depth, board.get_empty_count())
            
            logger.info(f"탐색 제한: 깊이 {max_depth}, 시간 {time_limit:.1f}초")
            
            # 반복 심화 탐색
            for depth in range(1, max_depth + 1):
                if time.time() - start_time > time_limit:
                    logger.info(f"시간 제한으로 깊이 {depth-1}에서 탐색 종료")
                    break
                
                try:
                    # 알파베타 탐색 실행 (시간 제한 포함)
                    end_time = start_time + time_limit
                    score, move = self.gpu_negamax(
                        board, depth, -float('inf'), float('inf'), True, end_time
                    )
                    
                    if move and board.is_valid_move(move[0], move[1], self.color):
                        result.score = score
                        result.best_move = move
                        result.depth = depth
                        result.nodes = self.nodes_searched
                        result.is_exact = (depth >= board.get_empty_count())
                        
                        logger.debug(f"깊이 {depth}: 점수 {result.score}, 수 {chr(result.best_move[1] + ord('a'))}{result.best_move[0] + 1}")
                    else:
                        logger.warning(f"깊이 {depth}에서 유효하지 않은 수 반환: {move}")
                        break
                    
                except Exception as e:
                    logger.warning(f"깊이 {depth} 탐색 중 오류: {e}")
                    break
            
            # 최종 통계 설정
            result.time_ms = (time.time() - start_time) * 1000
            
            # 결과가 없으면 첫 번째 유효한 수 선택
            if not result.best_move and valid_moves:
                result.best_move = valid_moves[0]
                logger.warning("탐색 실패, 첫 번째 유효한 수 선택")
            
            return result
            
        except Exception as e:
            logger.error(f"반복 심화 탐색 전체 오류: {e}")
            # 최소한의 결과 반환
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
            valid_moves = board.get_valid_moves(self.color)
            if valid_moves:
                result.best_move = valid_moves[0]
            return result
    
    def gpu_negamax(self, board: GPUBoard, depth: int, alpha: int, beta: int, 
                    maximizing: bool, end_time: float, passes=0):
        """
        GPU 가속 네가맥스 알고리즘 - 안전성 강화 버전
        """
        self.nodes_searched += 1
        
        # 시간 체크
        if time.time() > end_time:
            score = self.evaluator.evaluate_position_gpu(board, self.color)
            return score, None
        
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
                    if passes >= 1:  # 연속 패스 방지
                        score = self.evaluator.evaluate_position_gpu(board, self.color)
                        return score, None
                    score, move = self.gpu_negamax(board, depth, -beta, -alpha, 
                                                 not maximizing, end_time, passes + 1)
                    return -score, None
            else:
                score = self.evaluator.evaluate_position_gpu(board, self.color)
                return score, None
        
        # 수 정렬 (간단한 버전)
        ordered_moves = self._simple_move_ordering(board, moves, current_color)
        best_move = None
        best_score = alpha if maximizing else beta
        
        for move in ordered_moves:
            try:
                # 수가 여전히 유효한지 확인
                if not board.is_valid_move(move[0], move[1], current_color):
                    continue
                    
                new_board = board.apply_move(*move, current_color)
                
                score, _ = self.gpu_negamax(new_board, depth - 1, 
                                          -beta, -best_score, not maximizing, end_time, 0)
                score = -score
                
                if maximizing:
                    if score > best_score:
                        best_score = score
                        best_move = move
                    
                    if best_score >= beta:
                        self.cutoffs += 1
                        break
                else:
                    if score < best_score:
                        best_score = score
                        best_move = move
                    
                    if best_score <= alpha:
                        self.cutoffs += 1
                        break
                        
            except Exception as e:
                logger.debug(f"수 {move} 처리 중 오류: {e}")
                continue
        
        return best_score, best_move
    
    def _simple_move_ordering(self, board: GPUBoard, moves: List[Tuple[int, int]], 
                             current_color: int):
        """간단한 수 정렬"""
        if not moves:
            return moves
        
        # 코너 > 모서리 > 중앙 순으로 정렬
        move_scores = []
        for move in moves:
            x, y = move
            score = 0
            
            # 코너
            if (x, y) in CORNERS:
                score += 1000
            # X-squares (위험한 수)
            elif (x, y) in X_SQUARES:
                score -= 500
            # C-squares
            elif (x, y) in C_SQUARES:
                score -= 200
            # 모서리
            elif x == 0 or x == 7 or y == 0 or y == 7:
                score += 200
            # 내부
            else:
                center_distance = abs(x - 3.5) + abs(y - 3.5)
                score += int((7 - center_distance) * 10)
            
            move_scores.append((score, move))
        
        move_scores.sort(reverse=True)
        return [move for _, move in move_scores]
    
    def load_model(self, model_path='models/gpu_best_model.pth'):
        """훈련된 모델 로드"""
        if not self.use_neural_net or not hasattr(self, 'trainer') or not self.trainer:
            return

        try:
            if self.trainer.load_model(model_path):
                logger.info("GPU 신경망 모델 로드 완료")
            else:
                logger.warning("GPU 모델 파일을 찾을 수 없음. 랜덤 가중치 사용")
        except Exception as e:
            logger.error(f"GPU 모델 로드 중 오류: {e}")
    
    def _convert_to_gpu_board(self, board):
        """일반 보드를 GPU 보드로 변환"""
        return self._safe_convert_to_gpu_board(board)

# 나머지 클래스들과 함수들은 동일하므로 생략...

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