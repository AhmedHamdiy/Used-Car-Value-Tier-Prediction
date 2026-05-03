#!/bin/bash
# EC2 Setup Script for Car Price Tier Prediction Dashboard
# 
# This script prepares a fresh Ubuntu EC2 instance for the Streamlit dashboard.
# Run this once after creating a new EC2 instance.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
#
# Requirements:
#   - Ubuntu 22.04 LTS or similar
#   - Internet access
#   - Run as ubuntu user (not root)

set -euo pipefail

echo "=========================================="
echo "Car Dashboard - EC2 Setup"
echo "=========================================="

# Update system packages
echo "[1/5] Updating system packages..."
sudo apt-get update -y
sudo apt-get upgrade -y

# Install Python 3.11 (required by project)
echo "[2/5] Installing Python 3.11..."
sudo apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    git \
    curl \
    lsof

# Install Poetry
echo "[3/5] Installing Poetry..."
if ! command -v poetry &> /dev/null; then
    curl -sSL https://install.python-poetry.org | python3 -
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    export PATH="$HOME/.local/bin:$PATH"
    echo "Poetry installed: $(poetry --version)"
else
    echo "Poetry already installed: $(poetry --version)"
fi

# Create app directory
echo "[4/5] Creating application directory..."
mkdir -p ~/car-dashboard
echo "Directory created: ~/car-dashboard"

echo "[5/5] Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Copy project files: scp -i ~/.ssh/key.pem car-dashboard.tar.gz ubuntu@<IP>:~/"
echo "  2. Extract: tar -xzf ~/car-dashboard.tar.gz -C ~/car-dashboard"
echo "  3. Install deps: cd ~/car-dashboard && poetry install --only main"
echo "  4. Start app: poetry run streamlit run app.py --server.port=8501 --server.address=0.0.0.0"
echo ""
echo "Or use systemd service for production deployment."
echo "=========================================="
