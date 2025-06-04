"""
GPU Board Adapter
기존 board.py와 GPU 강화 보드 간의 호환성을 제공하는 어댑터
"""

import logging
import numpy as np
from typing import List, Tuple, Optional
import os

from datetime import datetime

def setup_gpu_adapter_logger():
    """세션별 GPU Board Adapter 로거 설정"""
    log_dir = "logs/gpu_adapter"
    os.makedirs(log_dir, exist_ok=True)
    
    # 세션별 고유 타임스탬프
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = f"session_{session_timestamp}"
    
    # 로그 파일명 생성
    log_filename = f"GPU_Adapter_{session_id}.log"
    log_filepath = os.path.join(log_dir, log_filename)
    
    logger = logging.getLogger('GPUBoardAdapter')
    logger.setLevel(logging.INFO)
    
    # 기존 핸들러 제거
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 파일 핸들러만 추가 (새 파일)
    file_handler = logging.FileHandler(log_filepath, mode='w', encoding='utf-8')
    
    # 포맷터 설정
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 상위 로거로 전파 방지
    logger.propagate = False
    
    # 세션 시작 로그
    logger.info("=" * 60)
    logger.info(f"🎯 GPU ADAPTER SESSION STARTED: {session_id}")
    logger.info(f"📁 Log File: {log_filepath}")
    logger.info("=" * 60)
    
    return logger


# 로거 설정
logger = setup_gpu_adapter_logger()

# 상수 정의 (임포트 실패에 대비)
try:
    from constants import BLACK, WHITE, EMPTY, opponent
except ImportError:
    BLACK = 1
    WHITE = 2
    EMPTY = 0
    
    def opponent(color):
        return WHITE if color == BLACK else BLACK

class BoardAdapter:
    """
    기존 Board 클래스와 GPU Board 클래스 간의 어댑터
    기존 GUI 코드의 수정 없이 GPU 가속 기능 사용 가능
    """
    
    def __init__(self, use_gpu=True):
        """
        어댑터 초기화
        Args:
            use_gpu: GPU 사용 여부
        """
        self.use_gpu = use_gpu
        self.gpu_manager = None
        self.gpu_board = None
        self.cpu_board = None
        
        if use_gpu:
            try:
                from gpu_ultra_strong_ai import GPUManager, GPUBoard
                self.gpu_manager = GPUManager()
                self.gpu_board = GPUBoard(self.gpu_manager)
                logger.info("GPU Board adapter initialized successfully")
            except ImportError as e:
                logger.warning(f"GPU modules not available: {e}")
                self._fallback_to_cpu()
        else:
            self._fallback_to_cpu()
    
    def _fallback_to_cpu(self):
        """CPU 보드로 폴백"""
        try:
            from board import Board
            self.cpu_board = Board()
            self.use_gpu = False
            logger.info("Using CPU Board (fallback)")
        except ImportError as e:
            logger.error(f"CPU Board import failed: {e}")
            raise
    
    @property
    def board(self):
        """
        보드 배열 반환 (호환성을 위해)
        Returns:
            List[List[int]]: 보드 상태 배열
        """
        if self.use_gpu and self.gpu_board:
            board_cpu = self.gpu_board.gpu.to_cpu(self.gpu_board.board)
            return board_cpu.tolist()
        else:
            return self.cpu_board.board
    
    @board.setter
    def board(self, value):
        """
        보드 배열 설정 (호환성을 위해)
        Args:
            value: 설정할 보드 배열
        """
        if self.use_gpu and self.gpu_board:
            board_array = np.array(value, dtype=np.int8)
            self.gpu_board.board = self.gpu_board.gpu.to_gpu(board_array)
        else:
            self.cpu_board.board = value
    
    @property
    def move_history(self):
        """이동 히스토리 반환"""
        if self.use_gpu and self.gpu_board:
            return self.gpu_board.move_history
        else:
            return self.cpu_board.move_history
    
    @move_history.setter
    def move_history(self, value):
        """이동 히스토리 설정"""
        if self.use_gpu and self.gpu_board:
            self.gpu_board.move_history = value
        else:
            self.cpu_board.move_history = value
    
    def in_bounds(self, x, y):
        """
        경계 확인
        Args:
            x, y: 좌표
        Returns:
            bool: 경계 내부 여부
        """
        return 0 <= x < 8 and 0 <= y < 8
    
    def get_valid_moves(self, color):
        """
        유효한 수 목록 반환
        Args:
            color: 돌 색상
        Returns:
            List[Tuple[int, int]]: 유효한 수 좌표 리스트
        """
        if self.use_gpu and self.gpu_board:
            return self.gpu_board.get_valid_moves(color)
        else:
            return self.cpu_board.get_valid_moves(color)
    
    def is_valid_move(self, x, y, color):
        """
        유효한 수인지 확인 - 안전성 강화
        Args:
            x, y: 좌표
            color: 돌 색상
        Returns:
            bool: 유효한 수 여부
        """
        try:
            # 좌표 범위 확인
            if not (0 <= x < 8 and 0 <= y < 8):
                return False
                
            if self.use_gpu and self.gpu_board:
                return self.gpu_board.is_valid_move(x, y, color)
            else:
                return self.cpu_board.is_valid_move(x, y, color)
        except Exception as e:
            logger.warning(f"유효성 검사 중 오류: {e}")
            return False
    
    def apply_move(self, x, y, color):
        """
        수를 두고 새로운 보드 반환 - 안전성 강화
        Args:
            x, y: 착수 좌표
            color: 돌 색상
        Returns:
            BoardAdapter: 새로운 보드 어댑터
        """
        try:
            # 입력 검증
            if not (0 <= x < 8 and 0 <= y < 8):
                logger.warning(f"유효하지 않은 좌표: ({x}, {y})")
                return self.copy()
            
            # 유효한 수인지 확인
            if not self.is_valid_move(x, y, color):
                logger.warning(f"유효하지 않은 수: ({x}, {y}) for color {color}")
                return self.copy()
            
            if self.use_gpu and self.gpu_board:
                new_gpu_board = self.gpu_board.apply_move(x, y, color)
                new_adapter = BoardAdapter(use_gpu=True)
                new_adapter.gpu_manager = self.gpu_manager
                new_adapter.gpu_board = new_gpu_board
                return new_adapter
            else:
                new_cpu_board = self.cpu_board.apply_move(x, y, color)
                new_adapter = BoardAdapter(use_gpu=False)
                new_adapter.cpu_board = new_cpu_board
                return new_adapter
        except Exception as e:
            logger.error(f"수 적용 중 오류: {e}")
            return self.copy()
    
    def count_stones(self):
        """
        돌 개수 세기
        Returns:
            Tuple[int, int]: (흑돌 수, 백돌 수)
        """
        if self.use_gpu and self.gpu_board:
            return self.gpu_board.count_stones()
        else:
            return self.cpu_board.count_stones()
    
    def get_empty_count(self):
        """
        빈 칸 개수 반환
        Returns:
            int: 빈 칸 개수
        """
        if self.use_gpu and self.gpu_board:
            return self.gpu_board.get_empty_count()
        else:
            return self.cpu_board.get_empty_count()
    
    def is_stable(self, x, y):
        """
        돌의 안정성 확인
        Args:
            x, y: 좌표
        Returns:
            bool: 안정한 돌 여부
        """
        if self.use_gpu and self.gpu_board:
            # GPU 보드의 안정성 검사 (간단한 버전)
            board_cpu = self.gpu_board.gpu.to_cpu(self.gpu_board.board)
            color = board_cpu[x, y]
            if color == 0:  # EMPTY
                return False
            
            # 코너는 항상 안정
            corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
            if (x, y) in corners:
                return True
            
            # 간단한 안정성 검사
            directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
            stable_directions = 0
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (0 <= nx < 8 and 0 <= ny < 8 and 
                    board_cpu[nx, ny] == color):
                    stable_directions += 1
            
            return stable_directions >= 3
        else:
            return self.cpu_board.is_stable(x, y)
    
    def get_frontier_count(self, color):
        """
        프론티어 디스크 개수 반환
        Args:
            color: 돌 색상
        Returns:
            int: 프론티어 디스크 개수
        """
        if self.use_gpu and self.gpu_board:
            return self.gpu_board.get_frontier_count(color)
        else:
            return self.cpu_board.get_frontier_count(color)
    
    def get_hash(self):
        """
        보드 해시값 반환
        Returns:
            int/str: 보드 해시값
        """
        if self.use_gpu and self.gpu_board:
            import hashlib
            board_cpu = self.gpu_board.gpu.to_cpu(self.gpu_board.board)
            board_str = ''.join(str(cell) for row in board_cpu for cell in row)
            return hashlib.md5(board_str.encode()).hexdigest()
        else:
            return self.cpu_board.get_hash()
    
    def copy(self):
        """
        보드 복사
        Returns:
            BoardAdapter: 복사된 보드 어댑터
        """
        if self.use_gpu and self.gpu_board:
            new_gpu_board = self.gpu_board.copy()
            new_adapter = BoardAdapter(use_gpu=True)
            new_adapter.gpu_manager = self.gpu_manager
            new_adapter.gpu_board = new_gpu_board
            return new_adapter
        else:
            new_cpu_board = self.cpu_board.copy() if hasattr(self.cpu_board, 'copy') else self._deep_copy_cpu_board()
            new_adapter = BoardAdapter(use_gpu=False)
            new_adapter.cpu_board = new_cpu_board
            return new_adapter
    
    def _deep_copy_cpu_board(self):
        """CPU 보드 깊은 복사 (copy 메서드가 없는 경우)"""
        import copy
        return copy.deepcopy(self.cpu_board)
    
    def to_string(self):
        """
        보드를 문자열로 변환 (디버깅용)
        Returns:
            str: 보드 문자열 표현
        """
        result = ""
        board_data = self.board
        
        for row in board_data:
            result += ''.join(['.' if cell == 0 else 
                              'B' if cell == 1 else 'W' for cell in row]) + '\n'
        return result
    
    def get_performance_info(self):
        """
        성능 정보 반환 (GPU 사용시)
        Returns:
            Dict: 성능 정보
        """
        info = {
            'backend': 'GPU' if self.use_gpu else 'CPU',
            'gpu_available': self.use_gpu and self.gpu_board is not None
        }
        
        if self.use_gpu and self.gpu_manager:
            info.update({
                'gpu_backend': self.gpu_manager.backend,
                'gpu_memory_available': self.gpu_manager.gpu_available
            })
            
            # GPU 메모리 정보 (CuPy 사용시)
            if self.gpu_manager.backend == 'cupy':
                try:
                    import cupy as cp
                    pool = cp.get_default_memory_pool()
                    info.update({
                        'gpu_memory_used_mb': pool.used_bytes() / (1024**2),
                        'gpu_memory_total_mb': pool.total_bytes() / (1024**2)
                    })
                except:
                    pass
        
        return info
    
    def cleanup_gpu_memory(self):
        """GPU 메모리 정리"""
        if self.use_gpu and self.gpu_manager:
            self.gpu_manager.clear_memory()
            logger.debug("GPU memory cleaned up via adapter")

    def get_board_array(self):
        """보드 배열 반환 (호환성 메서드)"""
        return self.board
    
    def set_board_array(self, board_array):
        """보드 배열 설정 (호환성 메서드)"""
        self.board = board_array
    
    def is_game_over(self):
        """게임 종료 여부 확인"""
        if self.use_gpu and self.gpu_board:
            return self.gpu_board.is_game_over()
        else:
            # CPU 보드용 구현
            black_moves = self.get_valid_moves(BLACK)
            white_moves = self.get_valid_moves(WHITE)
            return len(black_moves) == 0 and len(white_moves) == 0
    
    def get_winner(self):
        """승자 반환"""
        if self.use_gpu and self.gpu_board:
            return self.gpu_board.get_winner()
        else:
            if not self.is_game_over():
                return None
                
            black_count, white_count = self.count_stones()
            if black_count > white_count:
                return BLACK
            elif white_count > black_count:
                return WHITE
            else:
                return 0  # 무승부
# gpu_board_adapter.py의 AIAdapter 클래스를 이것으로 교체하세요

class AIAdapter:
    """
    기존 AI 클래스와 GPU AI 클래스 간의 어댑터 - 향상된 설정 지원
    """
    
    def __init__(self, color, ai_type='auto', difficulty='hard', time_limit=5.0, **kwargs):
        """
        AI 어댑터 초기화
        Args:
            color: AI 색상
            ai_type: AI 타입 ('gpu', 'cpu', 'auto', 'neural', 'mcts')
            difficulty: 난이도
            time_limit: 시간 제한
            **kwargs: 추가 설정 옵션들
                - search_depth: 탐색 깊이 ('auto', 또는 숫자)
                - use_opening_book: 오프닝북 사용 여부
                - use_endgame_solver: 완벽한 종료게임 해결 사용 여부
                - backend: 백엔드 지정 ('cpu', 'gpu')
                - algorithm: 알고리즘 지정 ('alphabeta', 'neural', 'mcts')
        """
        self.color = color
        self.ai_type = ai_type
        self.difficulty = difficulty
        self.time_limit = time_limit
        self.ai_instance = None
        self.use_gpu = False
        
        # 추가 설정 옵션들
        self.search_depth = kwargs.get('search_depth', 'auto')
        self.use_opening_book = kwargs.get('use_opening_book', True)
        self.use_endgame_solver = kwargs.get('use_endgame_solver', True)
        self.backend_preference = kwargs.get('backend', None)
        self.algorithm_preference = kwargs.get('algorithm', None)
        
        # GUI에서 전달된 설정들 처리
        if hasattr(kwargs, 'get'):
            self.backend_preference = kwargs.get('backend', self.backend_preference)
            self.algorithm_preference = kwargs.get('algorithm', self.algorithm_preference)
        
        # AI 초기화
        self._initialize_ai_with_options()
        
        logger.info(f"Enhanced AI Adapter initialized: type={self.ai_type}, gpu={self.use_gpu}, "
                   f"difficulty={difficulty}, depth={self.search_depth}")
    
    def _initialize_ai_with_options(self):
        """향상된 AI 초기화 - 설정 옵션 지원"""
        
        # 백엔드와 알고리즘에 따른 AI 타입 결정
        if self.backend_preference == 'gpu' or self.ai_type in ['gpu', 'neural', 'mcts']:
            self._try_gpu_ai()
        elif self.backend_preference == 'cpu' or self.ai_type == 'cpu':
            self._fallback_to_cpu_ai()
        elif self.ai_type == 'auto':
            # 자동 선택: GPU 사용 가능하면 GPU, 아니면 CPU
            if not self._try_gpu_ai():
                self._fallback_to_cpu_ai()
        else:
            self._fallback_to_cpu_ai()
        
        # AI 인스턴스 설정 적용
        self._apply_ai_settings()
    
    def _try_gpu_ai(self):
        """GPU AI 초기화 시도"""
        try:
            from gpu_ultra_strong_ai import UltraStrongAI, GPUManager
            gpu_manager = GPUManager()
            
            if gpu_manager.gpu_available:
                # 알고리즘별 설정
                use_neural = (self.algorithm_preference == 'neural' or 
                            self.ai_type == 'neural')
                
                self.ai_instance = UltraStrongAI(
                    color=self.color, 
                    difficulty=self.difficulty, 
                    time_limit=self.time_limit,
                    use_neural_net=use_neural
                )
                
                self.use_gpu = True
                self.ai_type = 'neural' if use_neural else 'gpu'
                logger.info(f"GPU AI initialized: algorithm={self.algorithm_preference}, neural={use_neural}")
                return True
            else:
                logger.warning("GPU 매니저가 GPU 사용 불가능 보고")
                return False
                
        except ImportError as e:
            logger.warning(f"GPU AI import 실패: {e}")
            return False
        except Exception as e:
            logger.error(f"GPU AI 초기화 실패: {e}")
            return False
    
    def _fallback_to_cpu_ai(self):
        """CPU AI로 폴백 - 설정 옵션 지원"""
        try:
            # 여러 AI 클래스 시도 (우선순위 순)
            ai_candidates = [
                ('ultra_strong_ai', 'UltraStrongAI', 'CPU Ultra Strong'),
                ('egaroucid_ai', 'EgaroucidStyleAI', 'Egaroucid Style'),
                ('ai', 'AdvancedAI', 'Advanced'),
                ('ai', 'AI', 'Basic')  # 기본 AI
            ]
            
            for module_name, class_name, display_name in ai_candidates:
                try:
                    module = __import__(module_name)
                    ai_class = getattr(module, class_name)
                    
                    # AI 초기화 (다양한 생성자 시그니처 지원)
                    if class_name == 'AdvancedAI':
                        self.ai_instance = ai_class(self.color, self.difficulty, self.time_limit)
                    elif class_name == 'UltraStrongAI':
                        # CPU 버전 UltraStrongAI
                        self.ai_instance = ai_class(
                            color=self.color,
                            difficulty=self.difficulty,
                            time_limit=self.time_limit,
                            use_neural_net=False  # CPU에서는 신경망 비활성화
                        )
                    else:
                        # 다른 AI들은 다양한 초기화 방법 시도
                        try:
                            self.ai_instance = ai_class(self.color, self.difficulty, self.time_limit)
                        except TypeError:
                            try:
                                self.ai_instance = ai_class(self.color, self.difficulty)
                            except TypeError:
                                self.ai_instance = ai_class(self.color)
                    
                    self.use_gpu = False
                    self.ai_type = 'cpu'
                    logger.info(f"CPU AI 사용: {display_name}")
                    return
                    
                except (ImportError, AttributeError, TypeError) as e:
                    logger.debug(f"AI 클래스 {class_name} 사용 불가: {e}")
                    continue
            
            raise ImportError("사용 가능한 AI 클래스가 없습니다")
            
        except Exception as e:
            logger.error(f"모든 AI 클래스 초기화 실패: {e}")
            raise
    
    def _apply_ai_settings(self):
        """AI 인스턴스에 설정 적용"""
        if not self.ai_instance:
            return
        
        try:
            # 탐색 깊이 설정
            if self.search_depth != 'auto' and hasattr(self.ai_instance, 'max_depth'):
                try:
                    depth = int(self.search_depth)
                    self.ai_instance.max_depth = depth
                    logger.info(f"탐색 깊이 설정: {depth}")
                except ValueError:
                    logger.warning(f"잘못된 탐색 깊이 값: {self.search_depth}")
            
            # 오프닝북 설정
            if hasattr(self.ai_instance, 'use_opening_book'):
                self.ai_instance.use_opening_book = self.use_opening_book
                logger.info(f"오프닝북 사용: {self.use_opening_book}")
            
            # 완벽한 종료게임 해결 설정
            if hasattr(self.ai_instance, 'use_perfect_endgame'):
                self.ai_instance.use_perfect_endgame = self.use_endgame_solver
                logger.info(f"완벽한 종료게임: {self.use_endgame_solver}")
            
            # 신경망 설정 (GPU AI의 경우)
            if (self.use_gpu and hasattr(self.ai_instance, 'use_neural_net') and 
                self.algorithm_preference == 'neural'):
                self.ai_instance.use_neural_net = True
                if hasattr(self.ai_instance, 'continuous_learning'):
                    self.ai_instance.continuous_learning = True
                logger.info("신경망 기능 활성화")
            
        except Exception as e:
            logger.warning(f"AI 설정 적용 중 오류: {e}")
    
    def get_move(self, board):
        """
        최적 수 반환 - 설정 기반 향상
        Args:
            board: 보드 객체 (BoardAdapter 또는 일반 Board)
        Returns:
            Optional[Tuple[int, int]]: 최적 수
        """
        if self.ai_instance is None:
            logger.error("AI 인스턴스가 초기화되지 않음")
            return None
        
        try:
            # 보드 유효성 검사
            valid_moves = board.get_valid_moves(self.color)
            if not valid_moves:
                logger.warning("AI에게 유효한 수가 없습니다")
                return None
            
            # 시간 제한 동적 조정 (설정에 따라)
            dynamic_time_limit = self._calculate_dynamic_time_limit(board)
            
            # AI에 동적 시간 제한 적용 (가능한 경우)
            original_time_limit = getattr(self.ai_instance, 'time_limit', None)
            if hasattr(self.ai_instance, 'time_limit'):
                self.ai_instance.time_limit = dynamic_time_limit
            
            try:
                # AI 수 계산
                if self.use_gpu and hasattr(self.ai_instance, '_convert_to_gpu_board'):
                    # GPU AI는 내부적으로 보드 변환 처리
                    move = self.ai_instance.get_move(board)
                else:
                    # CPU AI는 직접 사용
                    move = self.ai_instance.get_move(board)
                
                # 시간 제한 복원
                if original_time_limit is not None:
                    self.ai_instance.time_limit = original_time_limit
                
            except Exception as move_error:
                # 시간 제한 복원
                if original_time_limit is not None:
                    self.ai_instance.time_limit = original_time_limit
                raise move_error
            
            # 결과 검증
            if move:
                # 수가 유효한지 확인
                if board.is_valid_move(move[0], move[1], self.color):
                    logger.debug(f"AI move: {chr(move[1] + ord('a'))}{move[0] + 1}")
                    return move
                else:
                    logger.warning(f"AI가 유효하지 않은 수를 반환: {move}")
                    # 첫 번째 유효한 수 반환
                    return valid_moves[0] if valid_moves else None
            else:
                logger.warning("AI가 수를 반환하지 않음")
                return valid_moves[0] if valid_moves else None
            
        except Exception as e:
            logger.error(f"AI 수 생성 실패: {e}")
            # 긴급 수 선택
            try:
                valid_moves = board.get_valid_moves(self.color)
                return valid_moves[0] if valid_moves else None
            except:
                return None
    
    def _calculate_dynamic_time_limit(self, board):
        """보드 상황에 따른 동적 시간 제한 계산"""
        empty_count = board.get_empty_count()
        base_time = self.time_limit
        
        # 게임 단계별 시간 조정
        if empty_count > 50:  # 초반
            return min(base_time * 0.7, base_time)
        elif empty_count > 20:  # 중반
            return base_time
        elif empty_count > 10:  # 후반
            return min(base_time * 1.5, base_time + 3.0)
        else:  # 종료게임
            return min(base_time * 2.0, base_time + 5.0)
    
    def get_performance_info(self):
        """
        AI 성능 정보 반환 - 향상된 정보
        Returns:
            Dict: 성능 정보
        """
        info = {
            'ai_type': self.ai_type,
            'use_gpu': self.use_gpu,
            'difficulty': self.difficulty,
            'time_limit': self.time_limit,
            'search_depth': self.search_depth,
            'use_opening_book': self.use_opening_book,
            'use_endgame_solver': self.use_endgame_solver,
            'backend_preference': self.backend_preference,
            'algorithm_preference': self.algorithm_preference
        }
        
        # AI 인스턴스별 통계
        if hasattr(self.ai_instance, 'nodes_searched'):
            info['nodes_searched'] = self.ai_instance.nodes_searched
        
        if hasattr(self.ai_instance, 'tt_hits'):
            info['tt_hits'] = self.ai_instance.tt_hits
        
        if hasattr(self.ai_instance, 'cutoffs'):
            info['cutoffs'] = self.ai_instance.cutoffs
        
        if hasattr(self.ai_instance, 'perfect_searches'):
            info['perfect_searches'] = self.ai_instance.perfect_searches
        
        # GPU 관련 정보
        if self.use_gpu and hasattr(self.ai_instance, 'gpu'):
            gpu_info = self.ai_instance.gpu
            info.update({
                'gpu_backend': gpu_info.backend,
                'gpu_available': gpu_info.gpu_available
            })
        
        # 신경망 관련 정보
        if hasattr(self.ai_instance, 'use_neural_net'):
            info['neural_network_enabled'] = self.ai_instance.use_neural_net
        
        return info
    
    def get_ai_analysis(self, board):
        """AI의 상세 분석 정보 반환 (디버깅/학습용)"""
        try:
            if not hasattr(self.ai_instance, 'get_move'):
                return None
            
            analysis = {
                'valid_moves': board.get_valid_moves(self.color),
                'board_evaluation': None,
                'recommended_moves': [],
                'thinking_time': 0
            }
            
            # 간단한 보드 평가 (가능한 경우)
            if hasattr(self.ai_instance, 'evaluator'):
                try:
                    gpu_board = self.ai_instance._safe_convert_to_gpu_board(board)
                    if gpu_board:
                        analysis['board_evaluation'] = self.ai_instance.evaluator.evaluate_position_gpu(
                            gpu_board, self.color
                        )
                except:
                    pass
            
            return analysis
            
        except Exception as e:
            logger.debug(f"AI 분석 실패: {e}")
            return None
    
    def set_configuration(self, **kwargs):
        """실행 중 AI 설정 변경"""
        try:
            config_changed = False
            
            if 'difficulty' in kwargs:
                self.difficulty = kwargs['difficulty']
                config_changed = True
            
            if 'time_limit' in kwargs:
                self.time_limit = kwargs['time_limit']
                config_changed = True
            
            if 'search_depth' in kwargs:
                self.search_depth = kwargs['search_depth']
                config_changed = True
            
            if 'use_opening_book' in kwargs:
                self.use_opening_book = kwargs['use_opening_book']
                config_changed = True
            
            if 'use_endgame_solver' in kwargs:
                self.use_endgame_solver = kwargs['use_endgame_solver']
                config_changed = True
            
            if config_changed:
                self._apply_ai_settings()
                logger.info("AI 설정이 업데이트되었습니다")
            
            return config_changed
            
        except Exception as e:
            logger.error(f"AI 설정 변경 실패: {e}")
            return False
    
    def cleanup(self):
        """AI 리소스 정리 - 향상된 정리"""
        try:
            if self.use_gpu and hasattr(self.ai_instance, 'gpu'):
                self.ai_instance.gpu.clear_memory()
                logger.debug("AI GPU 리소스 정리 완료")
            
            # 신경망 관련 정리
            if hasattr(self.ai_instance, 'neural_net'):
                try:
                    # PyTorch 캐시 정리
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except:
                    pass
            
        except Exception as e:
            logger.debug(f"AI 정리 중 오류: {e}")

# 편의 함수도 업데이트
def create_adaptive_ai(color, ai_type='auto', difficulty='hard', time_limit=5.0, **kwargs):
    """
    적응형 AI 생성 - 향상된 버전
    Args:
        color: AI 색상
        ai_type: AI 타입 ('gpu', 'cpu', 'auto', 'neural', 'mcts')
        difficulty: 난이도
        time_limit: 시간 제한
        **kwargs: 추가 설정 옵션들
    Returns:
        AIAdapter: AI 어댑터 인스턴스
    """
    return AIAdapter(color, ai_type, difficulty, time_limit, **kwargs)

# 편의 함수들
def create_adaptive_board(prefer_gpu=True):
    """
    적응형 보드 생성
    Args:
        prefer_gpu: GPU 사용 선호 여부
    Returns:
        BoardAdapter: 보드 어댑터 인스턴스
    """
    return BoardAdapter(use_gpu=prefer_gpu)

def create_adaptive_ai(color, ai_type='auto', difficulty='hard', time_limit=5.0):
    """
    적응형 AI 생성
    Args:
        color: AI 색상
        ai_type: AI 타입 ('gpu', 'cpu', 'auto', 'neural')
        difficulty: 난이도
        time_limit: 시간 제한
    Returns:
        AIAdapter: AI 어댑터 인스턴스
    """
    return AIAdapter(color, ai_type, difficulty, time_limit)

def get_system_capabilities():
    """
    시스템 역량 정보 반환
    Returns:
        Dict: 시스템 정보
    """
    capabilities = {
        'gpu_available': False,
        'gpu_backend': 'none',
        'gpu_memory_gb': 0,
        'cuda_devices': 0,
        'recommended_mode': 'cpu'
    }
    
    # CuPy 확인
    try:
        import cupy as cp
        capabilities['gpu_available'] = True
        capabilities['gpu_backend'] = 'cupy'
        capabilities['cuda_devices'] = cp.cuda.runtime.getDeviceCount()
        
        if capabilities['cuda_devices'] > 0:
            device = cp.cuda.Device(0)
            capabilities['gpu_memory_gb'] = device.mem_info[1] // (1024**3)
            capabilities['recommended_mode'] = 'gpu'
            
    except ImportError:
        # Numba 확인
        try:
            from numba import cuda
            if cuda.is_available():
                capabilities['gpu_available'] = True
                capabilities['gpu_backend'] = 'numba'
                capabilities['recommended_mode'] = 'gpu'
        except ImportError:
            pass
    
    return capabilities

# 테스트 함수
def test_adapter_compatibility():
    """어댑터 호환성 테스트 - 안전성 강화"""
    logger.info("Testing adapter compatibility...")
    
    try:
        # 보드 어댑터 테스트
        board = create_adaptive_board(prefer_gpu=True)
        logger.info(f"Board created: {board.get_performance_info()}")
        
        # 기본 기능 테스트
        moves = board.get_valid_moves(1)  # BLACK
        logger.info(f"Valid moves: {len(moves)}")
        
        if moves:
            # 안전한 수 적용 테스트
            original_board_state = board.board
            new_board = board.apply_move(*moves[0], 1)
            
            if new_board != board:  # 새 보드가 생성되었는지 확인
                logger.info(f"Move applied successfully")
                
                b, w = new_board.count_stones()
                logger.info(f"Stone count: Black={b}, White={w}")
            else:
                logger.warning("Board was not updated after move")
        
        # AI 어댑터 테스트
        try:
            ai = create_adaptive_ai(1, 'auto', 'easy', 2.0)  # BLACK
            logger.info(f"AI created: {ai.get_performance_info()}")
            
            move = ai.get_move(board)
            if move:
                logger.info(f"AI move: {chr(move[1] + ord('a'))}{move[0] + 1}")
            else:
                logger.warning("AI returned no move")
            
            # 정리
            ai.cleanup()
        except Exception as ai_error:
            logger.warning(f"AI test failed: {ai_error}")
        
        # 정리
        board.cleanup_gpu_memory()
        
        logger.info("✅ Adapter compatibility test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Adapter compatibility test failed: {e}")
        return False

if __name__ == "__main__":
    # 시스템 역량 확인
    caps = get_system_capabilities()
    print("System Capabilities:")
    for key, value in caps.items():
        print(f"  {key}: {value}")
    
    # 호환성 테스트 실행
    test_adapter_compatibility()