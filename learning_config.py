# 학습 설정
LEARNING_CONFIG = {
    'continuous_learning': True,
    'training_interval': 5,  # 5게임마다 학습
    'batch_size': 32,
    'learning_epochs': 2,
    'auto_save_interval': 10,
    'buffer_size': 5000,
    'neural_net_priority': True  # 항상 신경망 우선
}

# 모델 경로
MODEL_PATHS = {
    'best_model': 'models/best_model.pth',
    'auto_save_dir': 'models/auto_saves/',
    'backup_dir': 'models/backups/'
}
