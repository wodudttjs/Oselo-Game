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
    
    # 동일한 형태의 포맷터
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 콘솔 출력 차단
    logger.propagate = False
    
    return logger

# 로거 설정
logger = setup_training_logger()

try:
    from gpu_ultra_strong_ai import UltraStrongAI, GPUSelfPlayTrainer, GPUManager, GPUOthelloNet, BLACK, WHITE
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    logger.warning("GPU 모듈을 가져올 수 없습니다. CPU 모드로 실행됩니다.")

class TrainingPipeline:
    """AlphaZero 스타일 훈련 파이프라인"""
    
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
        self.training_data = deque(maxlen=50000)  # 훈련 데이터 버퍼
        
        # GPU/CPU 트레이너 초기화
        self._initialize_trainer()
        
        logger.info(f"훈련 파이프라인 초기화 완료 (디바이스: {self.device})")

    def _initialize_trainer(self):
        """트레이너 초기화"""
        try:
            if GPU_AVAILABLE and torch.cuda.is_available():
                # GPU 트레이너 사용
                self.gpu_manager = GPUManager()
                self.neural_net = GPUOthelloNet().to(self.device)
                self.trainer = GPUSelfPlayTrainer(self.neural_net, self.gpu_manager)
                self.gpu_available = True
                logger.info("GPU 트레이너 초기화 완료")
            else:
                raise ImportError("GPU 모듈 사용 불가")
                
        except (ImportError, Exception) as e:
            # CPU 백업 트레이너
            self.gpu_available = False
            self.neural_net = self._create_simple_neural_net()
            self.optimizer = optim.Adam(self.neural_net.parameters(), lr=0.001)
            self.loss_fn = nn.MSELoss()
            logger.info("CPU 트레이너로 초기화 완료")

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
        """연속 학습 모드 콜백 반환"""
        logger.info("🎓 연속 학습 모드 시작 - 게임할 때마다 학습합니다")
        
        def post_game_callback(game_data):
            """게임 완료 후 호출되는 콜백"""
            try:
                if not game_data:
                    return
                
                # 게임 데이터를 신경망 훈련 데이터로 변환
                training_data = self.convert_game_data_to_training_data(game_data)
                
                # 훈련 데이터에 추가
                self.training_data.extend(training_data)
                self.game_counter += 1
                
                # 일정 게임마다 학습 실행
                if self.game_counter % self.auto_save_interval == 0:
                    logger.info(f"🎓 {self.game_counter}게임 완료, 학습 시작...")
                    
                    # 학습 실행
                    self._train_neural_network()
                    
                    # 모델 자동 저장
                    model_path = f"{self.model_dir}/auto_save_{self.game_counter}.pth"
                    self._save_model(model_path)
                    logger.info(f"✅ 자동 학습 및 저장 완료: {model_path}")
                    
            except Exception as e:
                logger.error(f"❌ 학습 콜백 오류: {e}")
        
        return post_game_callback

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

    def convert_game_data_to_training_data(self, game_data):
        """게임 데이터를 신경망 훈련 데이터로 변환"""
        training_data = []
        
        for data in game_data:
            try:
                # 보드 상태를 텐서로 변환
                board_tensor = self.board_to_tensor(data['board'], data['color'])
                
                # 정책 (실제 수를 인덱스로)
                move = data['move']
                move_idx = move[0] * 8 + move[1]
                
                # 가치는 게임 결과 사용
                value = float(data.get('value', 0.0))
                
                training_data.append((board_tensor, move_idx, value))
                
            except Exception as e:
                logger.warning(f"데이터 변환 오류: {e}")
                continue
        
        return training_data

    def board_to_tensor(self, board_array, color):
        """보드 배열을 텐서로 변환"""
        tensor = torch.zeros(3, 8, 8, dtype=torch.float32)
        
        for i in range(8):
            for j in range(8):
                cell_value = board_array[i][j] if isinstance(board_array[i], list) else board_array[i][j]
                
                if cell_value == color:
                    tensor[0][i][j] = 1.0
                elif cell_value != 0:  # 상대방 돌
                    tensor[1][i][j] = 1.0
        
        # 현재 플레이어 정보
        if color == 1:  # BLACK
            tensor[2] = torch.ones(8, 8)
        
        return tensor

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
