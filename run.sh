#!/usr/bin/env bash
# Start Chimera with one command: ComfyUI (with this node pack), the Comfy SDK proxy and the web app.
# Anything already running on its port is reused. Ctrl+C stops what this script started.
# Usage: ./run.sh   (set COMFYUI_DIR if ComfyUI is not in ~/ComfyUI)
set -e
cd "$(dirname "$0")"
COMFYUI_DIR=${COMFYUI_DIR:-$HOME/ComfyUI}
mkdir -p runs
pids=()
trap 'if ((${#pids[@]})); then kill "${pids[@]}" 2>/dev/null || true; fi' EXIT
trap 'exit 130' INT TERM

up() { curl -fsS -m 2 -o /dev/null "http://127.0.0.1:$1" 2>/dev/null; }

start() {  # health endpoint, log name, command...
  local endpoint=$1 log=runs/$2.log pid deadline
  shift 2
  if up "$endpoint"; then echo "$endpoint: already running"; return; fi
  "$@" > "$log" 2>&1 &
  pid=$!
  pids+=("$pid")
  deadline=$((SECONDS + 180))
  echo "$endpoint: starting (log: $log)"
  until up "$endpoint"; do
    kill -0 "$pid" 2>/dev/null || { echo "failed to start, see $log"; exit 1; }
    ((SECONDS < deadline)) || { echo "startup timed out, see $log"; exit 1; }
    sleep 2
  done
}

for executable in "$COMFYUI_DIR/.venv/bin/python" app/.venv/bin/comfy-api-proxy app/.venv/bin/uvicorn; do
  [[ -x "$executable" ]] || { echo "Missing $executable. Follow SETUP.md first."; exit 1; }
done

start 8188/system_stats comfyui bash -c 'cd "$1" && exec .venv/bin/python main.py --listen 127.0.0.1 --port 8188 --cache-ram 12' _ "$COMFYUI_DIR"
start 8189/api/v2/health proxy app/.venv/bin/comfy-api-proxy run --comfyui http://127.0.0.1:8188 --comfyui-base-dir "$COMFYUI_DIR" --max-upload-mb 2000
export COMFY_BASE_URL=http://127.0.0.1:8189
start 8000/openapi.json app app/.venv/bin/uvicorn server:app --app-dir app --port 8000
echo "Chimera is running at http://127.0.0.1:8000 (Ctrl+C to stop)"
if ((${#pids[@]})); then wait -n "${pids[@]}"; fi
