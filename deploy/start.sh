#!/bin/bash
# Start Script for Car Dashboard Streamlit App
#
# Usage:
#   ./start.sh          # Run in foreground (for testing)
#   ./start.sh &        # Run in background
#
# For production, use systemd service instead of this script.
# See: car-dashboard.service

set -euo pipefail

# Configuration
APP_DIR="$HOME/car-dashboard"
PORT=8501
ADDRESS="0.0.0.0"

echo "Starting Car Dashboard..."
echo "  Directory: $APP_DIR"
echo "  Port: $PORT"
echo "  Address: $ADDRESS"
echo ""

# Ensure Poetry is in PATH
export PATH="$HOME/.local/bin:$PATH"

# Change to app directory
cd "$APP_DIR"

# Verify Poetry environment exists
if ! poetry env info --path &> /dev/null; then
    echo "Error: Poetry environment not found. Run 'poetry install' first."
    exit 1
fi

# Start Streamlit
echo "Launching Streamlit server..."
poetry run streamlit run app.py \
    --server.port="$PORT" \
    --server.address="$ADDRESS" \
    --server.headless=true
