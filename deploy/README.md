# Deployment Files

## Files

| File | Purpose | When to Use |
|------|---------|-------------|
| `setup.sh` | Initial EC2 setup (installs Python 3.11, Poetry, dependencies) | Run once on a fresh EC2 instance |
| `car-dashboard.service` | systemd service configuration | Install for production auto-start |

## Quick Deployment

### 1. Initial Setup (Run on EC2)
```bash
chmod +x setup.sh ./setup.sh
```

### 2. Copy Project Files (Run from local)
```bash
tar -czf car-dashboard.tar.gz \
  --exclude='.git' --exclude='.pytest_cache' --exclude='__pycache__' \
  --exclude='.stakpak' --exclude='data/raw' --exclude='notebooks' \
  --exclude='htmlcov' --exclude='.coverage' .

scp -i ~/.ssh/car-dashboard-key.pem car-dashboard.tar.gz ubuntu@<IP>:~/
```

### 3. Extract and Install (Run on EC2)
```bash
ssh -i ~/.ssh/car-dashboard-key.pem ubuntu@<IP>
tar -xzf ~/car-dashboard.tar.gz -C ~/car-dashboard
cd ~/car-dashboard
poetry install --only main
```

### 4. Start the App

```bash
sudo cp deploy/car-dashboard.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable car-dashboard
sudo systemctl start car-dashboard
```

## Verifying Deployment

```bash
sudo systemctl status car-dashboard # check status

sudo journalctl -u car-dashboard -f # View logs

curl http://<IP>:8501 # Test dashboard
```

