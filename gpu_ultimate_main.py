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

def setup_main_logging():
    """메인 애플리케이션 세션별 로깅 설정 - 수정된 버전"""
    from datetime import datetime
    from pathlib import Path
    import os
    import logging
    
    # logs/main 디렉토리 생성 (parents=True로 상위 디렉토리도 함께 생성)
    log_dir = Path("logs") / "main"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 세션별 고유 타임스탬프 및 ID 생성
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = f"session_{session_timestamp}"
    
    # 다양한 로그 파일 생성
    main_log_file = log_dir / f"Main_{session_id}.log"
    error_log_file = log_dir / f"Errors_{session_id}.log"
    
    # 기존 로깅 설정 초기화 (중복 방지)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # 기본 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        handlers=[
            logging.FileHandler(main_log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler()  # 콘솔 출력
        ],
        force=True  # 기존 설정 강제 덮어쓰기
    )
    
    # 에러 전용 로거 설정
    error_logger = logging.getLogger('ErrorLogger')
    # 기존 핸들러 제거
    for handler in error_logger.handlers[:]:
        error_logger.removeHandler(handler)
    
    error_handler = logging.FileHandler(error_log_file, mode='w', encoding='utf-8')
    error_handler.setLevel(logging.ERROR)
    error_formatter = logging.Formatter(
        '%(asctime)s - ERROR - [%(filename)s:%(lineno)d] - %(funcName)s - %(message)s'
    )
    error_handler.setFormatter(error_formatter)
    error_logger.addHandler(error_handler)
    error_logger.propagate = False
    
    # 메인 로거 생성
    logger = logging.getLogger('OthelloGPUMain')
    
    # 로그 파일 생성 확인
    print(f"📁 Log directory created: {log_dir}")
    print(f"📄 Main log file: {main_log_file}")
    print(f"🚨 Error log file: {error_log_file}")
    
    # 세션 시작 정보 로깅
    logger.info("=" * 80)
    logger.info("🚀 ULTIMATE OTHELLO AI - NEW SESSION")
    logger.info("=" * 80)
    logger.info(f"📅 Session ID: {session_id}")
    logger.info(f"🕒 Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"📁 Main Log: {main_log_file}")
    logger.info(f"🚨 Error Log: {error_log_file}")
    logger.info(f"💻 Platform: {os.name}")
    
    # 시스템 정보 로깅
    try:
        import platform
        import sys
        logger.info(f"🐍 Python: {sys.version}")
        logger.info(f"🖥️ System: {platform.system()} {platform.release()}")
        logger.info(f"🏗️ Architecture: {platform.architecture()[0]}")
    except Exception as e:
        logger.warning(f"System info logging failed: {e}")
    
    logger.info("=" * 80)
    
    return logger

# 메인 로거 초기화
main_logger = setup_main_logging()


def check_gpu_support():
    """GPU 지원 여부 확인 및 정보 출력 - 향상된 버전"""
    gpu_info = {
        'cupy_available': False,
        'numba_available': False,
        'cuda_devices': 0,
        'gpu_memory': 0,
        'recommended_backend': 'cpu',
        'error_details': [],
        'cuda_version': 'Unknown'
    }
    
    main_logger.info("🔍 GPU 지원 상태 확인 중...")
    
    # CuPy 상세 확인
    main_logger.info("1️⃣ CuPy 확인 중...")
    try:
        import cupy as cp
        
        # 기본 정보 수집
        gpu_info['cuda_devices'] = cp.cuda.runtime.getDeviceCount()
        main_logger.info(f"   CUDA 디바이스 발견: {gpu_info['cuda_devices']}개")
        
        if gpu_info['cuda_devices'] > 0:
            # 디바이스 상세 정보
            device = cp.cuda.Device(0)
            props = cp.cuda.runtime.getDeviceProperties(0)
            gpu_info['gpu_memory'] = device.mem_info[1] // (1024**3)  # GB
            gpu_info['device_name'] = props['name'].decode()
            
            main_logger.info(f"   GPU 0: {gpu_info['device_name']}")
            main_logger.info(f"   GPU 메모리: {gpu_info['gpu_memory']} GB")
            
            # CUDA 버전 확인
            try:
                cuda_version = cp.cuda.runtime.runtimeGetVersion()
                gpu_info['cuda_version'] = f"{cuda_version // 1000}.{(cuda_version % 1000) // 10}"
                main_logger.info(f"   CUDA 런타임 버전: {gpu_info['cuda_version']}")
            except Exception as cuda_ver_error:
                gpu_info['error_details'].append(f"CUDA 버전 확인 실패: {cuda_ver_error}")
            
            # 실제 GPU 연산 테스트
            main_logger.info("   GPU 연산 테스트 중...")
            try:
                test_array = cp.array([1.0, 2.0, 3.0])
                result = cp.sum(test_array)
                cpu_result = result.get()
                
                if abs(cpu_result - 6.0) < 1e-6:
                    gpu_info['cupy_available'] = True
                    gpu_info['recommended_backend'] = 'cupy'
                    main_logger.info("   ✅ CuPy GPU 연산 테스트 성공!")
                else:
                    raise RuntimeError(f"연산 결과 불일치: {cpu_result} != 6.0")
                    
            except Exception as compute_error:
                error_msg = f"CuPy 연산 테스트 실패: {compute_error}"
                gpu_info['error_details'].append(error_msg)
                main_logger.warning(f"   ❌ {error_msg}")
        else:
            gpu_info['error_details'].append("CUDA 디바이스를 찾을 수 없음")
            main_logger.warning("   ❌ CUDA 디바이스를 찾을 수 없습니다")
            
    except ImportError as import_error:
        error_msg = f"CuPy 모듈 import 실패: {import_error}"
        gpu_info['error_details'].append(error_msg)
        main_logger.warning(f"   ❌ {error_msg}")
    except Exception as e:
        error_msg = f"CuPy 초기화 실패: {e}"
        gpu_info['error_details'].append(error_msg)
        main_logger.warning(f"   ❌ {error_msg}")
    
    # Numba CUDA 확인
    main_logger.info("2️⃣ Numba CUDA 확인 중...")
    try:
        from numba import cuda
        
        if cuda.is_available():
            detected = cuda.detect()
            main_logger.info(f"   Numba가 감지한 CUDA 디바이스: {detected.count}개")
            
            try:
                # Numba 컨텍스트 테스트
                ctx = cuda.current_context()
                device_name = ctx.device.name.decode()
                main_logger.info(f"   현재 디바이스: {device_name}")
                
                gpu_info['numba_available'] = True
                if not gpu_info['cupy_available']:
                    gpu_info['recommended_backend'] = 'numba'
                    
                main_logger.info("   ✅ Numba CUDA 사용 가능!")
                
            except Exception as numba_ctx_error:
                error_msg = f"Numba 컨텍스트 오류: {numba_ctx_error}"
                gpu_info['error_details'].append(error_msg)
                main_logger.warning(f"   ❌ {error_msg}")
        else:
            error_msg = "Numba에서 CUDA를 사용할 수 없음"
            gpu_info['error_details'].append(error_msg)
            main_logger.warning(f"   ❌ {error_msg}")
            
    except ImportError as numba_import_error:
        error_msg = f"Numba 모듈 import 실패: {numba_import_error}"
        gpu_info['error_details'].append(error_msg)
        main_logger.warning(f"   ❌ {error_msg}")
    except Exception as numba_error:
        error_msg = f"Numba 초기화 실패: {numba_error}"
        gpu_info['error_details'].append(error_msg)
        main_logger.warning(f"   ❌ {error_msg}")
    
    # 최종 추천 및 요약
    main_logger.info("📊 GPU 지원 상태 요약:")
    if gpu_info['cupy_available']:
        main_logger.info(f"🚀 추천 백엔드: CuPy (CUDA {gpu_info['cuda_version']})")
        main_logger.info(f"💾 GPU 메모리: {gpu_info['gpu_memory']} GB")
        main_logger.info("⚡ 성능: 최고 (GPU 가속 + 고급 메모리 관리)")
    elif gpu_info['numba_available']:
        main_logger.info("🔥 추천 백엔드: Numba CUDA")
        main_logger.info("⚡ 성능: 우수 (GPU 가속)")
    else:
        main_logger.info("💻 백엔드: CPU 전용")
        main_logger.info("⚡ 성능: 양호 (CPU 최적화)")
        
        if gpu_info['error_details']:
            main_logger.info("❌ GPU 사용 불가 원인:")
            for i, error in enumerate(gpu_info['error_details'], 1):
                main_logger.info(f"   {i}. {error}")
            
            main_logger.info("\n💡 GPU 지원을 원한다면:")
            main_logger.info("   1. NVIDIA GPU 드라이버 최신 버전 설치")
            main_logger.info("   2. CUDA Toolkit 설치 (11.x 또는 12.x)")
            main_logger.info("   3. CuPy 설치: pip install cupy-cuda11x 또는 cupy-cuda12x")
            main_logger.info("   4. 시스템 재시작 후 다시 시도")
    
    return gpu_info

def show_gpu_welcome_message(gpu_info):
    """GPU 지원 정보를 포함한 환영 메시지 표시 - 향상된 버전"""
    if gpu_info['recommended_backend'] in ['cupy', 'numba']:
        # GPU 사용 가능한 경우
        welcome_msg = f"""
🚀 GPU ENHANCED ULTIMATE OTHELLO AI 🚀

축하합니다! GPU 가속을 사용할 수 있습니다!

🔥 GPU 정보:
• Backend: {gpu_info['recommended_backend'].upper()}
• CUDA Devices: {gpu_info['cuda_devices']}
• GPU Memory: {gpu_info['gpu_memory']} GB
• CUDA Version: {gpu_info['cuda_version']}
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
        welcome_msg = f"""
💻 ULTIMATE OTHELLO AI (CPU MODE) 💻

GPU를 사용할 수 없지만 여전히 강력한 AI입니다!

❌ GPU 사용 불가 원인:
"""
        for i, error in enumerate(gpu_info['error_details'][:3], 1):  # 최대 3개만 표시
            welcome_msg += f"{i}. {error}\n"
        
        welcome_msg += f"""
🎯 CPU 최적화 기능:
✓ 18-ply 깊이 탐색
✓ 완벽한 종료게임 솔버
✓ 고급 평가 함수
✓ 토너먼트급 오프닝북
✓ Alpha-beta 가지치기
✓ 반복 심화 탐색

💡 GPU 가속을 원한다면:
1. NVIDIA GPU 드라이버 업데이트
2. CUDA Toolkit 설치 (11.x/12.x)
3. pip install cupy-cuda11x 또는 cupy-cuda12x
4. 시스템 재시작

그래도 이 AI는 충분히 강력합니다! 💪"""
    
    response = messagebox.askyesno("🎮 ULTIMATE CHALLENGE", welcome_msg)
    
    if response:
        main_logger.info("🎮 사용자가 도전을 수락했습니다!")
    else:
        main_logger.info("👋 사용자가 도전을 거절했습니다")
    
    return response

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
