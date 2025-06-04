import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from collections import deque
import logging
import os
import datetime
from datetime import datetime

def setup_training_logger():
    """세션별 Training Pipeline 로거 설정"""
    log_dir = "logs/training"
    os.makedirs(log_dir, exist_ok=True)
    
    # 세션별 고유 타임스탬프
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = f"session_{session_timestamp}"
    
    # 로그 파일명 생성
    log_filename = f"Training_{session_id}.log"
    log_filepath = os.path.join(log_dir, log_filename)
    
    logger = logging.getLogger('TrainingPipeline')
    logger.setLevel(logging.INFO)
    
    # 기존 핸들러 제거
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 파일 핸들러만 추가 (새 파일)
    file_handler = logging.FileHandler(log_filepath, mode='w', encoding='utf-8')
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.propagate = False
    
    # 세션 시작 로그
    logger.info("=" * 60)
    logger.info(f"🎓 TRAINING SESSION STARTED: {session_id}")
    logger.info(f"📁 Log File: {log_filepath}")
    logger.info("=" * 60)
    
    return logger

logger = setup_training_logger()

# 안전한 GPU 모듈 import
try:
    from gpu_ultra_strong_ai import (
        UltraStrongAI, GPUSelfPlayTrainer, GPUManager, 
        GPUOthelloNet, BLACK, WHITE
    )
    GPU_MODULES_AVAILABLE = True
except ImportError as e:
    GPU_MODULES_AVAILABLE = False
    logger.warning(f"GPU 모듈을 가져올 수 없습니다: {e}")
    
    # 백업 상수
    BLACK = 1
    WHITE = 2

class TrainingPipeline:
    """AlphaZero 스타일 훈련 파이프라인 - 안정화 버전"""
    
    def __init__(self, model_dir='models', device=None, auto_save_interval=5):
        # 디바이스 설정
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # 자동 저장 간격 설정
        self.auto_save_interval = auto_save_interval
        self.game_counter = 0
        self.training_data = deque(maxlen=50000)
        
        # 안전한 트레이너 초기화
        self._safe_initialize_trainer()
        
        logger.info(f"훈련 파이프라인 초기화 완료 (디바이스: {self.device})")

    def _safe_initialize_trainer(self):
        """안전한 트레이너 초기화"""
        self.gpu_available = False
        
        try:
            if (GPU_MODULES_AVAILABLE and 
                torch.cuda.is_available() and 
                GPUManager and GPUOthelloNet and GPUSelfPlayTrainer):
                
                # GPU 트레이너 사용
                self.gpu_manager = GPUManager()
                if self.gpu_manager.gpu_available:
                    self.neural_net = GPUOthelloNet().to(self.device)
                    self.trainer = GPUSelfPlayTrainer(self.neural_net, self.gpu_manager)
                    self.gpu_available = True
                    logger.info("GPU 트레이너 초기화 완료")
                else:
                    raise RuntimeError("GPU 매니저가 GPU를 사용할 수 없다고 보고")
        except Exception as e:
            logger.warning(f"GPU 트레이너 초기화 실패: {e}")
            
        # GPU 실패시 CPU 백업
        if not self.gpu_available:
            try:
                self.neural_net = self._create_simple_neural_net()
                self.optimizer = optim.Adam(self.neural_net.parameters(), lr=0.001)
                self.loss_fn = nn.MSELoss()
                self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)
                logger.info("CPU 트레이너로 초기화 완료")
            except Exception as e:
                logger.error(f"CPU 트레이너 초기화도 실패: {e}")
                # 최소한의 더미 구현
                self.neural_net = None
                self.optimizer = None
                self.loss_fn = None
                self.scheduler = None

    def _create_simple_neural_net(self):
        """간단한 신경망 생성 (GPU 모듈이 없을 때 사용)"""
        class SimpleOthelloNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
                self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
                self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
                
                # 정책 헤드 (64개 액션)
                self.policy_head = nn.Sequential(
                    nn.Conv2d(256, 32, 1),
                    nn.ReLU(),
                    nn.Flatten(),
                    nn.Linear(32 * 8 * 8, 64),
                    nn.LogSoftmax(dim=1)  # CrossEntropyLoss 사용을 위해 LogSoftmax 사용
                )
                
                # 가치 헤드
                self.value_head = nn.Sequential(
                    nn.Conv2d(256, 16, 1),
                    nn.ReLU(),
                    nn.Flatten(),
                    nn.Linear(16 * 8 * 8, 256),
                    nn.ReLU(),
                    nn.Linear(256, 1),
                    nn.Tanh()
                )

            def forward(self, x):
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                x = torch.relu(self.conv3(x))
                
                policy = self.policy_head(x)
                value = self.value_head(x)
                
                return policy, value

        return SimpleOthelloNet().to(self.device)

    def continuous_learning_mode(self):
        """연속 학습 모드 콜백 반환 - 안전한 버전"""
        logger.info("🎓 연속 학습 모드 시작")
        
        def safe_post_game_callback(game_data):
            """안전한 게임 완료 콜백"""
            try:
                if not game_data or not self.neural_net:
                    return
                
                # 게임 데이터를 신경망 훈련 데이터로 변환
                training_data = self.safe_convert_game_data(game_data)
                
                if training_data:
                    self.training_data.extend(training_data)
                    self.game_counter += 1
                    
                    # 일정 게임마다 학습 실행
                    if self.game_counter % self.auto_save_interval == 0:
                        logger.info(f"🎓 {self.game_counter}게임 완료, 학습 시작...")
                        
                        # 안전한 학습 실행
                        if self._safe_train_neural_network():
                            # 모델 자동 저장
                            model_path = f"{self.model_dir}/auto_save_{self.game_counter}.pth"
                            if self._safe_save_model(model_path):
                                logger.info(f"✅ 자동 학습 및 저장 완료: {model_path}")
                            else:
                                logger.warning("모델 저장 실패")
                        else:
                            logger.warning("학습 실패")
                    
            except Exception as e:
                logger.error(f"❌ 학습 콜백 오류: {e}")
        
        return safe_post_game_callback
    
    def safe_convert_game_data(self, game_data):
        """안전한 게임 데이터 변환 - 강화된 버전"""
        try:
            training_data = []
            
            for data in game_data:
                try:
                    # 데이터 유효성 검사
                    if not isinstance(data, dict):
                        continue
                        
                    # 필수 키 확인
                    required_keys = ['board', 'move', 'color']
                    if not all(key in data for key in required_keys):
                        continue
                    
                    # 보드 상태를 텐서로 변환
                    board_tensor = self.safe_board_to_tensor(data.get('board'), data.get('color'))
                    
                    if board_tensor is not None:
                        # 정책 (실제 수를 인덱스로)
                        move = data.get('move')
                        if move and len(move) >= 2:
                            x, y = move[0], move[1]
                            # 좌표 유효성 검사
                            if 0 <= x < 8 and 0 <= y < 8:
                                move_idx = x * 8 + y
                                value = float(data.get('value', 0.0))
                                # 값 범위 제한
                                value = max(-1.0, min(1.0, value))
                                training_data.append((board_tensor, move_idx, value))
                        
                except Exception as e:
                    logger.debug(f"개별 데이터 변환 오류: {e}")
                    continue
            
            logger.debug(f"변환된 훈련 데이터: {len(training_data)}개")
            return training_data
            
        except Exception as e:
            logger.error(f"게임 데이터 변환 실패: {e}")
            return []

    def safe_board_to_tensor(self, board_array, color):
        """안전한 보드 텐서 변환 - 강화된 버전"""
        try:
            if not board_array or not color:
                return None
            
            # 보드 배열 유효성 검사
            if not isinstance(board_array, (list, np.ndarray)):
                return None
            
            # 8x8 크기 확인
            if len(board_array) != 8:
                return None
                
            tensor = torch.zeros(3, 8, 8, dtype=torch.float32)
            
            for i in range(8):
                if not isinstance(board_array[i], (list, np.ndarray)) or len(board_array[i]) != 8:
                    continue
                    
                for j in range(8):
                    try:
                        cell_value = board_array[i][j]
                        
                        # 셀 값 유효성 검사
                        if cell_value == color:
                            tensor[0][i][j] = 1.0
                        elif cell_value != 0 and cell_value != color:  # 상대방 돌
                            tensor[1][i][j] = 1.0
                    except (IndexError, TypeError, ValueError):
                        continue
            
            # 현재 플레이어 정보
            if color == BLACK:
                tensor[2] = torch.ones(8, 8)
            
            return tensor
            
        except Exception as e:
            logger.debug(f"보드 텐서 변환 오류: {e}")
            return None
    
    def _safe_train_neural_network(self):
        """안전한 신경망 훈련"""
        try:
            if len(self.training_data) < 32:
                logger.debug("훈련 데이터 부족")
                return False
                
            if self.gpu_available and hasattr(self.trainer, 'train_neural_net'):
                # GPU 트레이너 사용
                try:
                    self.trainer.train_neural_net(batch_size=32, epochs=2)
                    return True
                except Exception as e:
                    logger.error(f"GPU 훈련 실패: {e}")
                    return False
            elif self.neural_net and self.optimizer:
                # CPU 트레이너 사용
                return self._safe_train_cpu_model()
            else:
                logger.warning("훈련 가능한 모델이 없습니다")
                return False
                
        except Exception as e:
            logger.error(f"신경망 훈련 실패: {e}")
            return False
    
    def _safe_train_cpu_model(self):
        """안전한 CPU 모델 훈련 - 개선된 버전"""
        try:
            if not self.neural_net or not self.optimizer or len(self.training_data) < 16:
                return False
            
            batch_size = 16
            epochs = 2
            
            # 안전한 데이터 샘플링
            sample_data = list(self.training_data)[-min(1000, len(self.training_data)):]
            
            self.neural_net.train()
            total_loss = 0
            batches = 0
            
            for epoch in range(epochs):
                epoch_loss = 0
                epoch_batches = 0
                
                for i in range(0, len(sample_data), batch_size):
                    batch = sample_data[i:i+batch_size]
                    if len(batch) < batch_size // 2:  # 최소 절반은 유효해야 함
                        continue
                    
                    try:
                        # 안전한 배치 처리
                        boards = []
                        policies = []
                        values = []
                        
                        for item in batch:
                            try:
                                if len(item) >= 3:
                                    boards.append(item[0])
                                    policies.append(item[1])
                                    values.append(item[2])
                            except (IndexError, TypeError):
                                continue
                        
                        if len(boards) < batch_size // 4:  # 최소 1/4은 유효해야 함
                            continue
                        
                        # 텐서 변환
                        boards_tensor = torch.stack(boards[:len(boards)]).to(self.device)
                        
                        # 정책을 원핫에서 클래스 인덱스로 변환
                        policy_indices = []
                        for p in policies[:len(boards)]:
                            if isinstance(p, (int, np.integer)):
                                # 이미 인덱스인 경우
                                policy_indices.append(min(63, max(0, int(p))))
                            elif isinstance(p, (list, np.ndarray)):
                                # 원핫 벡터인 경우
                                try:
                                    idx = np.argmax(p) if len(p) > 0 else 0
                                    policy_indices.append(min(63, max(0, int(idx))))
                                except:
                                    policy_indices.append(0)
                            else:
                                policy_indices.append(0)
                        
                        policies_tensor = torch.tensor(policy_indices, dtype=torch.long).to(self.device)
                        values_tensor = torch.tensor(values[:len(boards)], dtype=torch.float32).unsqueeze(1).to(self.device)
                        
                        # 훈련 단계
                        self.optimizer.zero_grad()
                        pred_policies, pred_values = self.neural_net(boards_tensor)
                        
                        # 손실 계산
                        policy_loss = nn.CrossEntropyLoss()(pred_policies, policies_tensor)
                        value_loss = self.loss_fn(pred_values, values_tensor)
                        total_loss_batch = policy_loss + value_loss
                        
                        # 역전파
                        total_loss_batch.backward()
                        torch.nn.utils.clip_grad_norm_(self.neural_net.parameters(), 1.0)
                        self.optimizer.step()
                        
                        epoch_loss += total_loss_batch.item()
                        epoch_batches += 1
                        
                    except Exception as batch_error:
                        logger.debug(f"배치 처리 오류: {batch_error}")
                        continue
                
                if epoch_batches > 0:
                    total_loss += epoch_loss / epoch_batches
                    batches += 1
                    logger.debug(f"Epoch {epoch+1}/{epochs}, 배치: {epoch_batches}, 평균 손실: {epoch_loss/epoch_batches:.4f}")
            
            # 스케줄러 업데이트
            if self.scheduler:
                self.scheduler.step()
            
            if batches > 0:
                avg_loss = total_loss / batches
                logger.info(f"CPU 모델 훈련 완료 - 평균 손실: {avg_loss:.4f}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"CPU 모델 훈련 실패: {e}")
            return False
    
    def _safe_save_model(self, model_path):
        """안전한 모델 저장"""
        try:
            if self.gpu_available and hasattr(self.trainer, 'save_model'):
                self.trainer.save_model(model_path)
                return True
            elif self.neural_net and self.optimizer:
                checkpoint = {
                    'model_state_dict': self.neural_net.state_dict(),
                    'game_counter': self.game_counter
                }
                
                if self.optimizer:
                    checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
                    
                if hasattr(self, 'scheduler') and self.scheduler:
                    checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
                
                torch.save(checkpoint, model_path)
                return True
            else:
                logger.warning("저장할 모델이 없습니다")
                return False
                
        except Exception as e:
            logger.error(f"모델 저장 실패: {e}")
            return False

    def load_model(self, model_path):
        """모델 로드 - 개선된 버전"""
        try:
            if not os.path.exists(model_path):
                logger.warning(f"모델 파일이 존재하지 않습니다: {model_path}")
                return False
                
            if self.gpu_available and hasattr(self.trainer, 'load_model'):
                return self.trainer.load_model(model_path)
            elif self.neural_net:
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # 모델 상태 로드
                if 'model_state_dict' in checkpoint:
                    self.neural_net.load_state_dict(checkpoint['model_state_dict'])
                
                # 옵티마이저 상태 로드
                if self.optimizer and 'optimizer_state_dict' in checkpoint:
                    try:
                        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    except Exception as e:
                        logger.warning(f"옵티마이저 상태 로드 실패: {e}")
                
                # 스케줄러 상태 로드
                if (hasattr(self, 'scheduler') and self.scheduler and 
                    'scheduler_state_dict' in checkpoint):
                    try:
                        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    except Exception as e:
                        logger.warning(f"스케줄러 상태 로드 실패: {e}")
                
                # 게임 카운터 로드
                self.game_counter = checkpoint.get('game_counter', 0)
                
                logger.info(f"모델 로드 완료: {model_path}")
                return True
            else:
                logger.error("로드할 모델이 초기화되지 않았습니다")
                return False
                
        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
            return False
    
    def get_training_stats(self):
        """훈련 통계 반환"""
        stats = {
            'game_counter': self.game_counter,
            'training_data_size': len(self.training_data),
            'gpu_available': self.gpu_available,
            'device': str(self.device)
        }
        
        if self.gpu_available and hasattr(self.trainer, 'training_stats'):
            stats.update(self.trainer.training_stats)
        
        return stats
    
    def manual_training_step(self, num_epochs=5, batch_size=32):
        """수동 훈련 단계 실행"""
        logger.info(f"수동 훈련 시작: {num_epochs} epochs, batch_size={batch_size}")
        
        if len(self.training_data) < batch_size:
            logger.warning(f"훈련 데이터 부족: {len(self.training_data)} < {batch_size}")
            return False
        
        try:
            if self.gpu_available and hasattr(self.trainer, 'train_neural_net'):
                self.trainer.train_neural_net(batch_size=batch_size, epochs=num_epochs)
                return True
            elif self.neural_net and self.optimizer:
                # CPU 훈련 실행
                return self._safe_train_cpu_model()
            else:
                logger.error("훈련 가능한 모델이 없습니다")
                return False
                
        except Exception as e:
            logger.error(f"수동 훈련 실패: {e}")
            return False
    
    def cleanup(self):
        """리소스 정리"""
        try:
            if self.gpu_available and hasattr(self, 'gpu_manager'):
                self.gpu_manager.clear_memory()
                logger.info("GPU 메모리 정리 완료")
        except Exception as e:
            logger.debug(f"정리 중 오류: {e}")

def main():
    """메인 실행 함수"""
    pipeline = TrainingPipeline()
    
    # 기존 모델이 있으면 로드
    best_model_path = 'models/best_model.pth'
    if os.path.exists(best_model_path):
        logger.info("기존 모델 발견, 로드합니다.")
        pipeline.load_model(best_model_path)
    else:
        logger.info("새로운 훈련을 시작합니다.")
    
    # 연속 학습 모드 콜백 생성
    learning_callback = pipeline.continuous_learning_mode()
    
    logger.info("훈련 파이프라인이 준비되었습니다.")
    logger.info("GUI에서 게임을 시작하면 자동으로 학습이 진행됩니다.")
    
    return pipeline, learning_callback

if __name__ == "__main__":
    main()