#!/bin/bash
# Restart the RegIntel app. Usage:
#   ./run.sh          — restart the UI (local only)
#   ./run.sh share    — restart and expose on the local network for teammates
cd "$(dirname "$0")"
source .venv/bin/activate

# Stop any running instance
pkill -f "streamlit run" 2>/dev/null && sleep 1

if [ "$1" = "share" ]; then
    exec streamlit run ui/app.py --server.address 0.0.0.0
else
    exec streamlit run ui/app.py
fi
