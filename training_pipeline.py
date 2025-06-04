import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

from collections import deque


import logging
import os

def setup_training_logger():
    """Training Pipeline 전용 로거 설정"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger('TrainingPipeline')
    logger.setLevel(logging.INFO)
    
    # 기존 핸들러 제거
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 파일 핸들러만 추가
    file_handler = logging.FileHandler(
        os.path.join(log_dir, 'training_pipeline.log'),
        mode='a',
        encoding='utf-8'
    )
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.propagate = False
    return logger

logger = setup_training_logger()
# 로거 설정


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
                    logger.info("CPU 트레이너로 초기화 완료")
                except Exception as e:
                    logger.error(f"CPU 트레이너 초기화도 실패: {e}")
                    # 최소한의 더미 구현
                    self.neural_net = None
                    self.optimizer = None
                    self.loss_fn = None

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
    
    def _train_neural_network(self):
        """신경망 훈련 실행"""
        if len(self.training_data) < 32:
            return
            
        if self.gpu_available and hasattr(self.trainer, 'train_neural_net'):
            # GPU 트레이너 사용
            self.trainer.train_neural_net(batch_size=32, epochs=2)
        else:
            # CPU 트레이너 사용
            self._train_cpu_model()

    def _train_cpu_model(self):
        """CPU 모델 훈련"""
        batch_size = 16
        epochs = 2
        
        # 훈련 데이터 샘플링
        sample_size = min(len(self.training_data), 1000)
        sample_data = list(self.training_data)[-sample_size:]
        
        for epoch in range(epochs):
            total_loss = 0
            batches = 0
            
            for i in range(0, len(sample_data), batch_size):
                batch = sample_data[i:i+batch_size]
                if len(batch) < batch_size:
                    continue
                
                try:
                    # 배치 데이터 준비
                    boards = torch.stack([item[0] for item in batch]).to(self.device)
                    
                    # 정책 타겟을 원핫 벡터에서 클래스 인덱스로 변환
                    policy_targets = []
                    for item in batch:
                        action_probs = item[1]
                        if isinstance(action_probs, np.ndarray):
                            target_idx = np.argmax(action_probs)
                        else:
                            target_idx = action_probs
                        policy_targets.append(target_idx)
                    
                    target_policies = torch.tensor(policy_targets, dtype=torch.long).to(self.device)
                    target_values = torch.tensor([item[2] for item in batch], dtype=torch.float32).unsqueeze(1).to(self.device)
                    
                    # 순전파
                    self.optimizer.zero_grad()
                    pred_policies, pred_values = self.neural_net(boards)
                    
                    # 손실 계산
                    policy_loss = nn.CrossEntropyLoss()(pred_policies, target_policies)
                    value_loss = self.loss_fn(pred_values, target_values)
                    total_loss_batch = policy_loss + value_loss
                    
                    # 역전파
                    total_loss_batch.backward()
                    torch.nn.utils.clip_grad_norm_(self.neural_net.parameters(), 1.0)
                    self.optimizer.step()
                    
                    total_loss += total_loss_batch.item()
                    batches += 1
                    
                except Exception as e:
                    logger.warning(f"배치 훈련 중 오류: {e}")
                    continue
            
            if batches > 0:
                avg_loss = total_loss / batches
                logger.info(f"Epoch {epoch+1}/{epochs}, 평균 손실: {avg_loss:.4f}")

    def _safe_train_cpu_model(self):
        """안전한 CPU 모델 훈련"""
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
                for i in range(0, len(sample_data), batch_size):
                    batch = sample_data[i:i+batch_size]
                    if len(batch) < batch_size:
                        continue
                    
                    try:
                        # 안전한 배치 처리
                        boards = []
                        policies = []
                        values = []
                        
                        for item in batch:
                            try:
                                boards.append(item[0])
                                policies.append(item[1])
                                values.append(item[2])
                            except (IndexError, KeyError, TypeError):
                                continue
                        
                        if len(boards) < batch_size // 2:  # 최소 절반은 유효해야 함
                            continue
                        
                        # 텐서 변환
                        boards_tensor = torch.stack(boards).to(self.device)
                        policies_tensor = torch.tensor(policies, dtype=torch.long).to(self.device)
                        values_tensor = torch.tensor(values, dtype=torch.float32).unsqueeze(1).to(self.device)
                        
                        # 훈련 단계
                        self.optimizer.zero_grad()
                        pred_policies, pred_values = self.neural_net(boards_tensor)
                        
                        policy_loss = nn.CrossEntropyLoss()(pred_policies, policies_tensor)
                        value_loss = self.loss_fn(pred_values, values_tensor)
                        total_loss_batch = policy_loss + value_loss
                        
                        total_loss_batch.backward()
                        torch.nn.utils.clip_grad_norm_(self.neural_net.parameters(), 1.0)
                        self.optimizer.step()
                        
                        total_loss += total_loss_batch.item()
                        batches += 1
                        
                    except Exception as batch_error:
                        logger.debug(f"배치 처리 오류: {batch_error}")
                        continue
            
            if batches > 0:
                avg_loss = total_loss / batches
                logger.info(f"훈련 완료 - 평균 손실: {avg_loss:.4f}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"CPU 모델 훈련 실패: {e}")
            return False
        
    def safe_convert_game_data(self, game_data):
        """안전한 게임 데이터 변환"""
        try:
            training_data = []
            
            for data in game_data:
                try:
                    # 보드 상태를 텐서로 변환
                    board_tensor = self.safe_board_to_tensor(data.get('board'), data.get('color'))
                    
                    if board_tensor is not None:
                        # 정책 (실제 수를 인덱스로)
                        move = data.get('move')
                        if move and len(move) >= 2:
                            move_idx = move[0] * 8 + move[1]
                            value = float(data.get('value', 0.0))
                            training_data.append((board_tensor, move_idx, value))
                        
                except Exception as e:
                    logger.debug(f"개별 데이터 변환 오류: {e}")
                    continue
            
            return training_data
            
        except Exception as e:
            logger.error(f"게임 데이터 변환 실패: {e}")
            return []

    def safe_board_to_tensor(self, board_array, color):
            """안전한 보드 텐서 변환"""
            try:
                if not board_array or not color:
                    return None
                    
                tensor = torch.zeros(3, 8, 8, dtype=torch.float32)
                
                for i in range(8):
                    for j in range(8):
                        try:
                            if isinstance(board_array[i], list):
                                cell_value = board_array[i][j]
                            else:
                                cell_value = board_array[i][j]
                            
                            if cell_value == color:
                                tensor[0][i][j] = 1.0
                            elif cell_value != 0:  # 상대방 돌
                                tensor[1][i][j] = 1.0
                        except (IndexError, TypeError):
                            continue
                
                # 현재 플레이어 정보
                if color == BLACK:
                    tensor[2] = torch.ones(8, 8)
                
                return tensor
                
            except Exception as e:
                logger.debug(f"보드 텐서 변환 오류: {e}")
                return None
            

    def _save_model(self, model_path):
        """모델 저장"""
        try:
            if self.gpu_available and hasattr(self.trainer, 'save_model'):
                self.trainer.save_model(model_path)
            else:
                torch.save({
                    'model_state_dict': self.neural_net.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'game_counter': self.game_counter
                }, model_path)
        except Exception as e:
            logger.error(f"모델 저장 실패: {e}")

    def _safe_train_neural_network(self):
        """안전한 신경망 훈련"""
        try:
            if len(self.training_data) < 32:
                return False
                
            if self.gpu_available and hasattr(self.trainer, 'train_neural_net'):
                # GPU 트레이너 사용
                self.trainer.train_neural_net(batch_size=32, epochs=2)
                return True
            elif self.neural_net and self.optimizer:
                # CPU 트레이너 사용
                return self._safe_train_cpu_model()
            else:
                logger.warning("훈련 가능한 모델이 없습니다")
                return False
                
        except Exception as e:
            logger.error(f"신경망 훈련 실패: {e}")
            return False
    
    def _safe_save_model(self, model_path):
        """안전한 모델 저장"""
        try:
            if self.gpu_available and hasattr(self.trainer, 'save_model'):
                self.trainer.save_model(model_path)
                return True
            elif self.neural_net and self.optimizer:
                torch.save({
                    'model_state_dict': self.neural_net.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'game_counter': self.game_counter
                }, model_path)
                return True
            else:
                logger.warning("저장할 모델이 없습니다")
                return False
                
        except Exception as e:
            logger.error(f"모델 저장 실패: {e}")
            return False

    def load_model(self, model_path):
        """모델 로드"""
        try:
            if os.path.exists(model_path):
                if self.gpu_available and hasattr(self.trainer, 'load_model'):
                    return self.trainer.load_model(model_path)
                else:
                    checkpoint = torch.load(model_path, map_location=self.device)
                    self.neural_net.load_state_dict(checkpoint['model_state_dict'])
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    self.game_counter = checkpoint.get('game_counter', 0)
                    logger.info(f"모델 로드 완료: {model_path}")
                    return True
        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
        return False

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
