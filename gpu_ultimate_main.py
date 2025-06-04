#!/usr/bin/env python3

"""
🚀 GPU ENHANCED ULTIMATE OTHELLO AI 🚀
GPU 가속 버전의 최강 오델로 AI 프로젝트
"""

import tkinter as tk
from tkinter import messagebox
import sys
import os
import logging
import time
import threading
from pathlib import Path

# GPU 지원 확인 및 로깅 설정
def setup_main_logging():
    """메인 애플리케이션 로깅 설정"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"othello_gpu_ai_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger('OthelloGPUMain')
    logger.info("=" * 60)
    logger.info("🚀 GPU Enhanced Ultimate Othello AI Starting")
    logger.info("=" * 60)
    return logger

# 메인 로거 초기화
main_logger = setup_main_logging()

def check_gpu_support():
    """GPU 지원 여부 확인 및 정보 출력"""
    gpu_info = {
        'cupy_available': False,
        'numba_available': False,
        'cuda_devices': 0,
        'gpu_memory': 0,
        'recommended_backend': 'cpu'
    }
    
    main_logger.info("🔍 Checking GPU support...")
    
    # CuPy 확인
    try:
        import cupy as cp
        gpu_info['cupy_available'] = True
        gpu_info['cuda_devices'] = cp.cuda.runtime.getDeviceCount()
        if gpu_info['cuda_devices'] > 0:
            device = cp.cuda.Device(0)
            gpu_info['gpu_memory'] = device.mem_info[1] // (1024**3)  # GB
            gpu_info['recommended_backend'] = 'cupy'
        main_logger.info(f"✅ CuPy available: {gpu_info['cuda_devices']} CUDA devices")
        main_logger.info(f"📊 GPU Memory: {gpu_info['gpu_memory']} GB")
    except ImportError:
        main_logger.warning("⚠️ CuPy not available")
    except Exception as e:
        main_logger.warning(f"⚠️ CuPy error: {e}")
    
    # Numba CUDA 확인
    try:
        from numba import cuda
        if cuda.is_available():
            gpu_info['numba_available'] = True
            if not gpu_info['cupy_available']:
                gpu_info['recommended_backend'] = 'numba'
            main_logger.info("✅ Numba CUDA available")
        else:
            main_logger.warning("⚠️ Numba CUDA not available")
    except ImportError:
        main_logger.warning("⚠️ Numba not available")
    except Exception as e:
        main_logger.warning(f"⚠️ Numba error: {e}")
    
    # 최종 추천
    if gpu_info['cupy_available'] or gpu_info['numba_available']:
        main_logger.info(f"🚀 Recommended backend: {gpu_info['recommended_backend']}")
    else:
        main_logger.info("💻 Using CPU backend (GPU not available)")
        gpu_info['recommended_backend'] = 'cpu'
    
    return gpu_info

def check_dependencies():
    """필수 파일 및 의존성 확인"""
    main_logger.info("🔍 Checking dependencies...")
    
    required_files = [
        'constants.py',
        'board.py',
        'ai.py',
        'ultimate_gui.py'
    ]
    
    optional_files = [
        'egaroucid_ai.py',
        'enhanced_board.py',
        'enhanced_gui.py',
        'gpu_board_adapter.py',
        'gpu_ultra_strong_ai.py',
        'training_pipeline.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
            main_logger.error(f"❌ Missing required file: {file}")
    
    # 선택적 파일 확인
    for file in optional_files:
        if os.path.exists(file):
            main_logger.info(f"✅ Optional file found: {file}")
        else:
            main_logger.warning(f"⚠️ Optional file missing: {file}")
    
    if missing_files:
        error_msg = "❌ Missing required files:\n\n"
        for file in missing_files:
            error_msg += f"• {file}\n"
        error_msg += "\nPlease ensure all files are in the same directory."
        messagebox.showerror("Missing Files", error_msg)
        main_logger.error("Dependency check failed")
        return False
    
    main_logger.info("✅ All dependencies satisfied")
    return True

def check_python_version():
    """Python 버전 및 필수 라이브러리 확인"""
    main_logger.info("🐍 Checking Python environment...")
    
    # Python 버전 확인
    python_version = sys.version_info
    main_logger.info(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 7):
        messagebox.showerror("Python Version Error",
                           "Python 3.7 or higher is required!\n"
                           f"Current version: {python_version.major}.{python_version.minor}")
        return False
    
    # 필수 라이브러리 확인
    required_libs = {
        'tkinter': 'GUI framework',
        'numpy': 'Numerical computing',
        'threading': 'Multi-threading support',
        'hashlib': 'Hashing algorithms',
        'time': 'Time utilities',
        'logging': 'Logging system'
    }
    
    missing_libs = []
    for lib, description in required_libs.items():
        try:
            __import__(lib)
            main_logger.debug(f"✅ {lib}: {description}")
        except ImportError:
            missing_libs.append(lib)
            main_logger.error(f"❌ Missing: {lib} ({description})")
    
    if missing_libs:
        error_msg = f"❌ Missing required libraries:\n\n"
        for lib in missing_libs:
            error_msg += f"• {lib}\n"
        error_msg += "\nPlease install missing libraries using pip."
        messagebox.showerror("Missing Libraries", error_msg)
        return False
    
    return True

def show_gpu_welcome_message(gpu_info):
    """GPU 지원 정보를 포함한 환영 메시지 표시"""
    if gpu_info['recommended_backend'] != 'cpu':
        # GPU 사용 가능한 경우
        welcome_msg = f"""
🚀 GPU ENHANCED ULTIMATE OTHELLO AI 🚀

축하합니다! GPU 가속을 사용할 수 있습니다!

🔥 GPU 정보:
• Backend: {gpu_info['recommended_backend'].upper()}
• CUDA Devices: {gpu_info['cuda_devices']}
• GPU Memory: {gpu_info['gpu_memory']} GB
• CuPy Available: {'Yes' if gpu_info['cupy_available'] else 'No'}
• Numba CUDA: {'Yes' if gpu_info['numba_available'] else 'No'}

⚡ GPU 가속 기능:
✓ 병렬 보드 평가 (10x+ 빠름)
✓ 배치 이동 생성 (5x+ 빠름)
✓ 고속 transposition table
✓ 벡터화된 패턴 인식
✓ 병렬 종료게임 탐색

⚠️ 주의: 이 AI는 극도로 강력합니다!
GPU 가속으로 더욱 무시무시해졌습니다.

준비되셨나요? 🔥"""
    else:
        # CPU만 사용 가능한 경우
        welcome_msg = """
💻 ULTIMATE OTHELLO AI (CPU MODE) 💻

GPU를 사용할 수 없지만 여전히 강력한 AI입니다!

🎯 CPU 최적화 기능:
✓ 18-ply 깊이 탐색
✓ 완벽한 종료게임 솔버
✓ 고급 평가 함수
✓ 토너먼트급 오프닝북
✓ Alpha-beta 가지치기
✓ 반복 심화 탐색

💡 GPU 가속을 원한다면:
pip install cupy-cuda11x  # CUDA 11.x용
또는
pip install cupy-cuda12x  # CUDA 12.x용

그래도 이 AI는 충분히 강력합니다! 💪"""
    
    response = messagebox.askyesno("🎮 ULTIMATE CHALLENGE", welcome_msg)
    
    if response:
        main_logger.info("🎮 User accepted the challenge!")
    else:
        main_logger.info("👋 User declined the challenge")
    
    return response

def main():
    """메인 애플리케이션 진입점"""
    try:
        main_logger.info("🚀 Application startup initiated")
        
        # 1. Python 환경 확인
        if not check_python_version():
            main_logger.error("Python environment check failed")
            sys.exit(1)
        
        # 2. 의존성 확인
        if not check_dependencies():
            main_logger.error("Dependency check failed")
            sys.exit(1)
        
        # 3. GPU 지원 확인
        gpu_info = check_gpu_support()
        
        # 4. 환영 메시지 및 사용자 확인
        if not show_gpu_welcome_message(gpu_info):
            main_logger.info("👋 Application terminated by user choice")
            sys.exit(0)
        
        # 5. GUI 모듈 import (의존성 확인 후)
        try:
            from ultimate_gui import UltimateOthelloGUI
            main_logger.info("✅ GUI module imported successfully")
        except ImportError as e:
            error_msg = f"❌ GUI Import Error: {e}\n\n"
            error_msg += "Please ensure ultimate_gui.py is in the same directory."
            messagebox.showerror("Import Error", error_msg)
            main_logger.error(f"GUI import failed: {e}")
            sys.exit(1)
        
        # 6. 메인 윈도우 생성
        main_logger.info("🎮 Creating main application window")
        root = tk.Tk()
        
        # 윈도우 설정
        root.configure(bg="#0f0f23")
        root.title("🚀 GPU Enhanced Ultimate Othello AI")
        
        # 7. GUI 애플리케이션 생성
        main_logger.info("🎮 Initializing game interface...")
        app = UltimateOthelloGUI(root)
        
        # 8. 연속 학습 모드 설정 (GUI 생성 후)
        try:
            from training_pipeline import TrainingPipeline
            pipeline = TrainingPipeline()
            
            # GUI에 학습 기능 추가
            if hasattr(app, 'enable_learning_mode'):
                learning_callback = pipeline.continuous_learning_mode()
                app.enable_learning_mode(learning_callback)
                main_logger.info("🔥 Learning mode activated")
            else:
                main_logger.warning("GUI does not support learning mode")
        except ImportError as e:
            main_logger.warning(f"Training pipeline not available: {e}")
        
        # 9. GPU 정보를 GUI에 전달 (GUI가 지원한다면)
        if hasattr(app, 'set_gpu_info'):
            app.set_gpu_info(gpu_info)
            main_logger.info("🔥 GPU information passed to GUI")
        
        # 10. 시작 완료 로그
        main_logger.info("🚀 Game interface ready!")
        main_logger.info("💡 Tips for playing:")
        main_logger.info(" • GPU 가속으로 더 깊은 탐색 가능")
        main_logger.info(" • 더 정확한 위치 평가")
        main_logger.info(" • 빠른 종료게임 계산")
        main_logger.info(" • Good luck... you'll need it! 😈")
        main_logger.info("=" * 60)
        
        # 11. GUI 이벤트 루프 시작
        main_logger.info("▶️ Starting GUI event loop")
        root.mainloop()
        
        main_logger.info("🏁 Application terminated normally")
        
    except KeyboardInterrupt:
        main_logger.info("⌨️ Application interrupted by user (Ctrl+C)")
        sys.exit(0)
    except ImportError as e:
        error_msg = f"❌ Import Error: {e}\n\n"
        error_msg += "Please ensure all Python files are in the same directory\n"
        error_msg += "and that there are no syntax errors in the code."
        messagebox.showerror("Import Error", error_msg)
        main_logger.error(f"Import error: {e}")
        sys.exit(1)
    except Exception as e:
        error_msg = f"❌ Unexpected Error: {e}\n\n"
        error_msg += "Please check the log file for more details.\n"
        error_msg += f"Log location: logs/othello_gpu_ai_*.log"
        messagebox.showerror("Unexpected Error", error_msg)
        main_logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # 정리 작업
        try:
            # GPU 메모리 정리
            try:
                import cupy as cp
                cp.get_default_memory_pool().free_all_blocks()
                main_logger.info("🧹 GPU memory cleaned up")
            except:
                pass
            
            main_logger.info("👋 Thanks for playing GPU Enhanced Ultimate Othello AI!")
        except Exception as e:
            main_logger.error(f"Cleanup error: {e}")

if __name__ == "__main__":
    # 명령줄 인수 처리
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help":
            print("""
🚀 GPU Enhanced Ultimate Othello AI

Usage:
  python gpu_ultimate_main.py        # Normal game mode
  python gpu_ultimate_main.py --help # Show this help

Features:
🔥 GPU acceleration with CuPy/Numba
⚡ 18-ply depth search
🧠 Perfect endgame solver
📊 Real-time performance monitoring
🎯 Tournament-level AI

Requirements:
- Python 3.7+
- tkinter (GUI)
- numpy (computation)
- cupy or numba (GPU acceleration, optional)

For best performance, install GPU support:
  pip install cupy-cuda11x  # For CUDA 11.x
  pip install cupy-cuda12x  # For CUDA 12.x
""")
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Use --help for usage information")
    else:
        # 일반 실행
        main()
