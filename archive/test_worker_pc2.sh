#!/bin/bash
# Test Worker PC 2 GPU Access with sshpass
# ========================================
# 
# Uses sshpass for automated SSH access to Worker PC 2
# Username: w2, Password: w, IP: 192.168.1.11

echo "🔍 TESTING WORKER PC 2 GPU ACCESS WITH SSHPASS"
echo "==============================================="

# Check if sshpass is installed
if ! command -v sshpass &> /dev/null; then
    echo "❌ sshpass not found. Installing..."
    sudo apt-get update && sudo apt-get install -y sshpass
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install sshpass. Please install manually:"
        echo "   sudo apt-get install sshpass"
        exit 1
    fi
fi

echo "✅ sshpass available"

# Copy diagnostic script to Worker PC 2 using sshpass
echo "📋 Copying diagnostic script to Worker PC 2..."
sshpass -p 'w' scp -o StrictHostKeyChecking=no check_worker_pc2_gpu.py w2@192.168.1.11:/tmp/
if [ $? -ne 0 ]; then
    echo "❌ Failed to copy diagnostic script to Worker PC 2"
    exit 1
fi

echo "✅ Diagnostic script copied successfully"

# Run diagnostic on Worker PC 2 using sshpass
echo "🧪 Running GPU diagnostic on Worker PC 2..."
sshpass -p 'w' ssh -o StrictHostKeyChecking=no w2@192.168.1.11 "cd /tmp && /home/w2/miniconda3/envs/BotsTraining_env/bin/python3.12 check_worker_pc2_gpu.py"

echo "🎯 Worker PC 2 GPU diagnostic complete"
