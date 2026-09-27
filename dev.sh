#!/usr/bin/env bash
#
# Marathi NLP Studio — start backend + frontend with one command.
#
#   ./dev.sh
#
# Starts the FastAPI backend (port 8000) and the Vite dev server (port 5173).
# Vite proxies /api/* to the backend. Press Ctrl+C once to stop both.
#
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$ROOT/backend"
FRONTEND_DIR="$ROOT/frontend"
BACKEND_PORT=8000
FRONTEND_PORT=5173

# --- pick a Python that actually has torch installed -------------------------
# On this machine `py -3.12` is the interpreter with torch/transformers; a bare
# `python`/`uvicorn` on PATH may resolve to a different install without torch.
if py -3.12 -c "import torch" >/dev/null 2>&1; then
  PY="py -3.12"
elif python -c "import torch" >/dev/null 2>&1; then
  PY="python"
else
  echo "ERROR: No Python with 'torch' installed was found." >&2
  echo "       Tried: 'py -3.12' and 'python'." >&2
  echo "       Install backend deps, e.g.:  py -3.12 -m pip install -r backend/requirements.txt" >&2
  exit 1
fi
echo "Using Python: $PY"

# --- ensure frontend deps are present ----------------------------------------
if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
  echo "Installing frontend dependencies (first run)…"
  npm --prefix "$FRONTEND_DIR" install
fi

BACKEND_PID=""
FRONTEND_PID=""

# Kill a whole process tree by PID (Windows-friendly, with a POSIX fallback).
kill_tree() {
  local pid="$1"
  [ -z "$pid" ] && return 0
  if command -v taskkill >/dev/null 2>&1; then
    taskkill //F //T //PID "$pid" >/dev/null 2>&1 || true
  else
    kill "$pid" >/dev/null 2>&1 || true
  fi
}

# Fallback: kill whatever is LISTENING on a TCP port (handles orphaned children).
kill_port() {
  local port="$1"
  command -v netstat >/dev/null 2>&1 || return 0
  local pids
  pids=$(netstat -ano 2>/dev/null | grep ":$port " | grep -i LISTENING | awk '{print $NF}' | sort -u)
  for p in $pids; do kill_tree "$p"; done
}

cleanup() {
  echo ""
  echo "Shutting down…"
  kill_tree "$FRONTEND_PID"
  kill_tree "$BACKEND_PID"
  kill_port "$BACKEND_PORT"
  kill_port "$FRONTEND_PORT"
}
trap cleanup EXIT INT TERM

# --- start backend -----------------------------------------------------------
echo "Starting backend on http://localhost:$BACKEND_PORT …"
(
  cd "$BACKEND_DIR"
  PYTHONIOENCODING=utf-8 exec $PY -m uvicorn app.main:app --port "$BACKEND_PORT"
) &
BACKEND_PID=$!

# --- wait until the backend answers /health ----------------------------------
echo -n "Waiting for the model to load"
for _ in $(seq 1 60); do
  if curl -sf "http://localhost:$BACKEND_PORT/health" >/dev/null 2>&1; then
    echo " — ready."
    break
  fi
  # Bail out early if the backend process already died.
  if ! kill -0 "$BACKEND_PID" >/dev/null 2>&1; then
    echo ""
    echo "ERROR: backend exited during startup. See the traceback above." >&2
    exit 1
  fi
  echo -n "."
  sleep 2
done

# --- start frontend ----------------------------------------------------------
echo "Starting frontend on http://localhost:$FRONTEND_PORT …"
(
  cd "$FRONTEND_DIR"
  exec npm run dev -- --port "$FRONTEND_PORT"
) &
FRONTEND_PID=$!

echo ""
echo "Both servers are running:"
echo "  Frontend : http://localhost:$FRONTEND_PORT"
echo "  Backend  : http://localhost:$BACKEND_PORT  (docs at /docs)"
echo "Press Ctrl+C to stop both."

# Exit (and trigger cleanup) as soon as either process stops.
wait -n "$BACKEND_PID" "$FRONTEND_PID"
