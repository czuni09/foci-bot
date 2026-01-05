#!/usr/bin/env bash
set -euo pipefail

# Use PORT provided by Render, default to 8501
PORT="${PORT:-8501}"

exec streamlit run streamlit_app.py --server.port "$PORT" --server.headless true
