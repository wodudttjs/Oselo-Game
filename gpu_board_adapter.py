"""
GPU Board Adapter
기존 board.py와 GPU 강화 보드 간의 호환성을 제공하는 어댑터
"""

import logging
import numpy as np
from typing import List, Tuple, Optional

# 로거 설정
logger = logging.getLogger('GPUBoardAdapter')

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
        유효한 수인지 확인
        Args:
            x, y: 좌표
            color: 돌 색상
        Returns:
            bool: 유효한 수 여부
        """
        if self.use_gpu and self.gpu_board:
            return self.gpu_board.is_valid_move(x, y, color)
        else:
            return self.cpu_board.is_valid_move(x, y, color)
    
    def apply_move(self, x, y, color):
        """
        수를 두고 새로운 보드 반환
        Args:
            x, y: 착수 좌표
            color: 돌 색상
        Returns:
            BoardAdapter: 새로운 보드 어댑터
        """
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

class AIAdapter:
    """
    기존 AI 클래스와 GPU AI 클래스 간의 어댑터
    """
    
    def __init__(self, color, ai_type='auto', difficulty='hard', time_limit=5.0):
        """
        AI 어댑터 초기화
        Args:
            color: AI 색상
            ai_type: AI 타입 ('gpu', 'cpu', 'auto')
            difficulty: 난이도
            time_limit: 시간 제한
        """
        self.color = color
        self.ai_type = ai_type
        self.difficulty = difficulty
        self.time_limit = time_limit
        self.ai_instance = None
        self.use_gpu = False
        
        self._initialize_ai()
        
        logger.info(f"AI Adapter initialized: type={ai_type}, gpu={self.use_gpu}, difficulty={difficulty}")
    
    def _initialize_ai(self):
        """AI 인스턴스 초기화"""
        if self.ai_type == 'auto':
            # 자동 선택: GPU 사용 가능하면 GPU, 아니면 CPU
            try:
                from gpu_ultra_strong_ai import UltraStrongAI, GPUManager
                gpu_manager = GPUManager()
                if gpu_manager.gpu_available:
                    self.ai_instance = UltraStrongAI(self.color, self.difficulty, self.time_limit)
                    self.use_gpu = True
                    self.ai_type = 'gpu'
                    logger.info("Auto-selected GPU AI")
                else:
                    raise ImportError("GPU not available")
            except ImportError:
                self._fallback_to_cpu_ai()
        
        elif self.ai_type == 'gpu':
            # 강제 GPU 사용
            try:
                from gpu_ultra_strong_ai import UltraStrongAI
                self.ai_instance = UltraStrongAI(self.color, self.difficulty, self.time_limit)
                self.use_gpu = True
                logger.info("GPU AI forced")
            except ImportError as e:
                logger.error(f"GPU AI not available: {e}")
                self._fallback_to_cpu_ai()
        
        else:
            # CPU 사용
            self._fallback_to_cpu_ai()
    
    def _fallback_to_cpu_ai(self):
        """CPU AI로 폴백"""
        try:
            # 여러 AI 클래스 시도
            ai_classes = [
                ('ai', 'AdvancedAI'),
                ('egaroucid_ai', 'EgaroucidStyleAI'),
                ('ultra_strong_ai', 'UltraStrongAI')
            ]
            
            for module_name, class_name in ai_classes:
                try:
                    module = __import__(module_name)
                    ai_class = getattr(module, class_name)
                    self.ai_instance = ai_class(self.color, self.difficulty, self.time_limit)
                    self.use_gpu = False
                    self.ai_type = 'cpu'
                    logger.info(f"Using CPU AI: {class_name}")
                    return
                except (ImportError, AttributeError) as e:
                    logger.debug(f"AI class {class_name} not available: {e}")
                    continue
            
            raise ImportError("No AI class available")
            
        except ImportError as e:
            logger.error(f"All AI classes failed: {e}")
            raise
    
    def get_move(self, board):
        """
        최적 수 반환
        Args:
            board: 보드 객체 (BoardAdapter 또는 일반 Board)
        Returns:
            Optional[Tuple[int, int]]: 최적 수
        """
        if self.ai_instance is None:
            logger.error("AI instance not initialized")
            return None
        
        try:
            # GPU AI인 경우 보드 변환이 필요할 수 있음
            if self.use_gpu and hasattr(self.ai_instance, '_convert_to_gpu_board'):
                # GPU AI는 내부적으로 보드 변환 처리
                move = self.ai_instance.get_move(board)
            else:
                # CPU AI는 직접 사용
                move = self.ai_instance.get_move(board)
            
            if move:
                logger.debug(f"AI move: {chr(move[1] + ord('a'))}{move[0] + 1}")
            else:
                logger.warning("AI returned no move")
            
            return move
            
        except Exception as e:
            logger.error(f"AI move generation failed: {e}")
            return None
    
    def get_performance_info(self):
        """
        AI 성능 정보 반환
        Returns:
            Dict: 성능 정보
        """
        info = {
            'ai_type': self.ai_type,
            'use_gpu': self.use_gpu,
            'difficulty': self.difficulty,
            'time_limit': self.time_limit
        }
        
        if hasattr(self.ai_instance, 'nodes_searched'):
            info['nodes_searched'] = self.ai_instance.nodes_searched
        
        if hasattr(self.ai_instance, 'tt_hits'):
            info['tt_hits'] = self.ai_instance.tt_hits
        
        if hasattr(self.ai_instance, 'cutoffs'):
            info['cutoffs'] = self.ai_instance.cutoffs
        
        if self.use_gpu and hasattr(self.ai_instance, 'gpu'):
            gpu_info = self.ai_instance.gpu
            info.update({
                'gpu_backend': gpu_info.backend,
                'gpu_available': gpu_info.gpu_available
            })
        
        return info
    
    def cleanup(self):
        """AI 리소스 정리"""
        if self.use_gpu and hasattr(self.ai_instance, 'gpu'):
            self.ai_instance.gpu.clear_memory()
            logger.debug("AI GPU resources cleaned up")

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
        ai_type: AI 타입 ('gpu', 'cpu', 'auto')
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
    """어댑터 호환성 테스트"""
    logger.info("Testing adapter compatibility...")
    
    try:
        # 보드 어댑터 테스트
        board = create_adaptive_board(prefer_gpu=True)
        logger.info(f"Board created: {board.get_performance_info()}")
        
        # 기본 기능 테스트
        moves = board.get_valid_moves(1)  # BLACK
        logger.info(f"Valid moves: {len(moves)}")
        
        if moves:
            new_board = board.apply_move(*moves[0], 1)
            logger.info(f"Move applied successfully")
            
            b, w = new_board.count_stones()
            logger.info(f"Stone count: Black={b}, White={w}")
        
        # AI 어댑터 테스트
        ai = create_adaptive_ai(1, 'auto', 'easy', 2.0)  # BLACK
        logger.info(f"AI created: {ai.get_performance_info()}")
        
        move = ai.get_move(board)
        if move:
            logger.info(f"AI move: {chr(move[1] + ord('a'))}{move[0] + 1}")
        
        # 정리
        board.cleanup_gpu_memory()
        ai.cleanup()
        
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