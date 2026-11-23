
import subprocess
import sys
import time
import urllib.request

API_COMMAND = [
    sys.executable,
    "-m",
    "uvicorn",
    "app.api.routes:app",
    "--host",
    "127.0.0.1",
    "--port",
    "8000",
]

UI_COMMAND = [
    sys.executable,
    "-m",
    "streamlit",
    "run",
    "app/ui/dashboard.py",
    "--server.address",
    "127.0.0.1",
    "--server.port",
    "8501",
    "--server.headless",
    "true",
]


def wait_for_api(process, timeout=30):
    start = time.time()

    while time.time() - start < timeout:
        if process.poll() is not None:
            raise RuntimeError("FastAPI backend stopped unexpectedly.")

        try:
            with urllib.request.urlopen(
                "http://127.0.0.1:8000/health",
                timeout=2,
            ) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(1)

    raise RuntimeError("FastAPI backend failed to become healthy.")


def main():
    backend = None
    frontend = None

    try:
        print("Starting RegulaRAG FastAPI backend...")

        backend = subprocess.Popen(API_COMMAND)

        wait_for_api(backend)

        print("FastAPI is healthy.")
        print("Starting Streamlit frontend...")

        frontend = subprocess.Popen(UI_COMMAND)

        print("\nRegulaRAG AI is running!")
        print("Frontend: http://localhost:8501")
        print("Backend:  http://localhost:8000")
        print("API docs: http://localhost:8000/docs")
        print("\nPress CTRL+C to stop both services.")

        while True:
            if backend.poll() is not None:
                raise RuntimeError("FastAPI backend stopped.")

            if frontend.poll() is not None:
                raise RuntimeError("Streamlit frontend stopped.")

            time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping RegulaRAG AI...")

    finally:
        for process in (frontend, backend):
            if process is not None and process.poll() is None:
                process.terminate()

        for process in (frontend, backend):
            if process is not None:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()

        print("RegulaRAG AI stopped.")


if __name__ == "__main__":
    main()
