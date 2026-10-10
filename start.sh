#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# STAIR – Startup script
# Starts the FastAPI server and opens the browser
# Usage: ./start.sh [--model MODEL] [--port PORT]
# ─────────────────────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$SCRIPT_DIR/stair_app"
PORT=${PORT:-8765}
MODEL=${MODEL:-"qwen2.5:14b"}

echo ""
echo "  ╔═══════════════════════════════════════════════╗"
echo "  ║   STAIR – STructure-Aware Information Retriever ║"
echo "  ╚═══════════════════════════════════════════════╝"
echo ""

# Check Python
if ! command -v python3 &>/dev/null; then
  echo "❌ Python3 is required. Please install it first."
  exit 1
fi

# Check Ollama
if ! curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "⚠️  Ollama is not running. Please start it with: ollama serve"
  echo "   Then run this script again."
  exit 1
fi

echo "✅ Ollama detected"

# Install dependencies if needed
echo "📦 Checking Python dependencies..."
pip3 install fastapi uvicorn pymupdf rank_bm25 requests -q

echo "🚀 Starting STAIR server on http://localhost:${PORT}..."
echo ""

# Export environment variables
export OLLAMA_BASE_URL="http://localhost:11434"
export STAIR_RETRIEVAL_MODEL="$MODEL"
export STAIR_ANSWER_MODEL="$MODEL"

# Open browser after a short delay (macOS)
(sleep 2 && open "http://localhost:${PORT}" 2>/dev/null || \
             xdg-open "http://localhost:${PORT}" 2>/dev/null || true) &

# Start server
cd "$APP_DIR"
python3 server.py
