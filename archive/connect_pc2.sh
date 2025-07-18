#!/bin/bash

echo "Connecting to PC2 to start Ray worker..."

# First, ensure PC2 is accessible
echo "Testing PC2 connectivity..."
if ! ping -c 1 192.168.1.11 > /dev/null 2>&1; then
    echo "ERROR: Cannot reach PC2 at 192.168.1.11"
    exit 1
fi

echo "PC2 is reachable. Connecting via SSH..."

# Stop any existing Ray processes on PC2
echo "Stopping any existing Ray processes on PC2..."
sshpass -p 'w' ssh -o StrictHostKeyChecking=no w@192.168.1.11 "pkill -f ray || true"

# Wait a moment
sleep 2

# Start Ray worker on PC2
echo "Starting Ray worker on PC2..."
sshpass -p 'w' ssh -o StrictHostKeyChecking=no w@192.168.1.11 << 'EOF'
cd /home/w1/cursor-to-copilot-backup/TaskmasterForexBots
source ~/.bashrc
conda activate Training_env
ray start --address='192.168.1.10:6379' --node-ip-address=192.168.1.11 --object-manager-port=10001 --ray-client-server-port=10201 --min-worker-port=10300 --max-worker-port=10399
EOF

echo "Ray worker connection attempt completed."
echo "Checking cluster status..."
ray status
