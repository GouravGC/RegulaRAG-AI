
#!/usr/bin/env bash
set -Eeuo pipefail

APP_PORT="${PORT:-8501}"
API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-8000}"

echo "Starting FastAPI..."
uvicorn app.api.routes:app \
  --host "$API_HOST" \
  --port "$API_PORT" \
  --workers 1 &

API_PID=$!

cleanup() {
  echo "Stopping application..."
  kill "$API_PID" "${UI_PID:-}" 2>/dev/null || true
  wait "$API_PID" 2>/dev/null || true

  if [[ -n "${UI_PID:-}" ]]; then
    wait "$UI_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

API_READY=0

for attempt in $(seq 1 30); do
  if ! kill -0 "$API_PID" 2>/dev/null; then
    echo "FastAPI exited during startup."
    exit 1
  fi

  if python -c "import urllib.request; urllib.request.urlopen('http://${API_HOST}:${API_PORT}/health', timeout=1).read()" >/dev/null 2>&1; then
    API_READY=1
    break
  fi

  sleep 1
done

if [[ "$API_READY" -ne 1 ]]; then
  echo "FastAPI did not become healthy."
  exit 1
fi

echo "FastAPI is healthy."
echo "Starting Streamlit on port $APP_PORT..."

streamlit run app/ui/dashboard.py \
  --server.address 0.0.0.0 \
  --server.port "$APP_PORT" \
  --server.headless true \
  --server.enableCORS false &

UI_PID=$!

set +e
wait -n "$API_PID" "$UI_PID"
STATUS=$?
set -e

exit "$STATUS"
