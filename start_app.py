
import os
import subprocess
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable

processes = []

try:
    # Start FastAPI health endpoint
    api = subprocess.Popen([
        PYTHON, "-m", "uvicorn",
        "health_api:app",
        "--host", "127.0.0.1",
        "--port", "8000"
    ], cwd=BASE_DIR)
    processes.append(api)

    # Start Streamlit UI
    streamlit = subprocess.Popen([
        PYTHON, "-m", "streamlit", "run",
        "app.py",
        "--server.address", "127.0.0.1",
        "--server.port", "8502"
    ], cwd=BASE_DIR)
    processes.append(streamlit)

    print("TeamRAG services are starting.")
    print("Streamlit: http://localhost:8502")
    print("Health API: http://127.0.0.1:8000/health")
    print("Press Ctrl+C to stop both services.")

    while True:
        for process in processes:
            if process.poll() is not None:
                raise RuntimeError(
                    f"A service stopped with exit code {process.returncode}"
                )
        time.sleep(1)

except KeyboardInterrupt:
    print("\nStopping TeamRAG services...")

except RuntimeError as error:
    print(error)

finally:
    for process in processes:
        if process.poll() is None:
            process.terminate()

    for process in processes:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
