#!/usr/bin/env bash
set -euo pipefail

# --- Go to project root (where this script lives) ---
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# --- Ensure venv exists (bootstrap if missing) ---
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install -U pip
  # Try editable install with UI extras; fall back to plain if extras not defined
  if ! pip install -e ".[ui]"; then
    pip install -e .
  fi
else
  source .venv/bin/activate
fi

# --- Defaults for LLM (can be overridden by your env or Streamlit secrets) ---
export LLM_PROVIDER="${LLM_PROVIDER:-ollama}"
export LLM_MODEL="${LLM_MODEL:-llama3.2:3b}"
export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"

# --- If Ollama selected but not reachable, fall back to Basic (rule-based) ---
if [ "$LLM_PROVIDER" = "ollama" ]; then
  if ! curl -fsS "$OLLAMA_BASE_URL/api/tags" >/dev/null 2>&1; then
    echo "⚠️  Ollama not reachable at $OLLAMA_BASE_URL; using Basic summary."
    export LLM_PROVIDER="rule"
  fi
fi

# --- Pick a free port (8501..8510) to avoid 'already in use' ---
PORT="${PORT:-8501}"
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port $PORT busy; searching..."
  for p in 8501 8502 8503 8504 8505 8506 8507 8508 8509 8510; do
    if ! lsof -nP -iTCP:"$p" -sTCP:LISTEN >/dev/null 2>&1; then
      PORT="$p"
      break
    fi
  done
fi
URL="http://localhost:$PORT"
echo "Launching EDA Copilot on $URL"

# --- Run Streamlit (background), then open browser when ready ---
LOGDIR="logs"; mkdir -p "$LOGDIR"
streamlit run ui/app.py --server.port "$PORT" > "$LOGDIR/streamlit_${PORT}.log" 2>&1 &
SPID=$!

# Wait for server to respond (max ~10s)
for _ in {1..40}; do
  if curl -fsS "$URL" >/dev/null 2>&1; then
    break
  fi
  sleep 0.25
done

# --- Open browser (macOS 'open', Linux 'xdg-open' if present) ---
if command -v open >/dev/null 2>&1; then
  open "$URL"
elif command -v xdg-open >/dev/null 2>&1; then
  xdg-open "$URL" >/dev/null 2>&1 || true
fi

# --- Attach to the Streamlit process so Ctrl-C stops it ---
wait "$SPID"