#!/usr/bin/env bash
# Start Chimera with one command: ComfyUI (with this node pack), the Comfy SDK proxy and the web app.
# Anything already running on its port is reused. Ctrl+C stops what this script started.
# Usage: ./run.sh   (set COMFYUI_DIR if ComfyUI is not in ~/ComfyUI)
cd "$(dirname "$0")"
COMFYUI_DIR=${COMFYUI_DIR:-$HOME/ComfyUI}
mkdir -p runs
pids=()
trap 'kill "${pids[@]}" 2>/dev/null' EXIT

up() { curl -s -m 2 -o /dev/null "http://127.0.0.1:$1/"; }

start() {  # port, log name, command...
  local port=$1 log=runs/$2.log
  shift 2
  if up $port; then echo "$port: already running"; return; fi
  "$@" > $log 2>&1 &
  pids+=($!)
  echo "$port: starting (log: $log)"
  until up $port; do
    kill -0 $! 2>/dev/null || { echo "failed to start, see $log"; exit 1; }
    sleep 2
  done
}

start 8188 comfyui bash -c "cd '$COMFYUI_DIR' && exec .venv/bin/python main.py --listen 127.0.0.1 --port 8188 --cache-ram 12"
start 8189 proxy app/.venv/bin/comfy-api-proxy --comfyui http://127.0.0.1:8188 --comfyui-base-dir "$COMFYUI_DIR"
export COMFY_BASE_URL=http://127.0.0.1:8189
start 8000 app app/.venv/bin/uvicorn server:app --app-dir app --port 8000
echo "Chimera is running at http://127.0.0.1:8000 (Ctrl+C to stop)"
wait
