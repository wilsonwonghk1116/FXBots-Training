#!/bin/bash
"""
WORKER PC 2 CONNECTION SCRIPT
============================
This script connects Worker PC 2 to the Ray cluster on Head PC 1.

Instructions:
1. Copy this script to Worker PC 2
2. Make it executable: chmod +x connect_worker_pc2.sh
3. Run it: ./connect_worker_pc2.sh

Network Configuration:
- Head PC 1: 192.168.1.10
- Worker PC 2: 192.168.1.11
- Connected via LAN cable
"""

echo "🔗 CONNECTING WORKER PC 2 TO RAY CLUSTER"
echo "========================================"
echo "Head PC 1 IP: 192.168.1.10"
echo "Worker PC 2 IP: 192.168.1.11"
echo "========================================"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Miniconda/Anaconda first."
    exit 1
fi

# Activate the correct conda environment
echo "🐍 Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate BotsTraining_env

# Check if the environment was activated successfully
if [ $? -ne 0 ]; then
    echo "❌ Failed to activate BotsTraining_env. Please ensure it exists."
    echo "💡 You may need to create it with: conda create -n BotsTraining_env python=3.12"
    exit 1
fi

echo "✅ Environment activated: $(conda info --envs | grep '*')"

# Check Python version
echo "🐍 Python version: $(python --version)"

# Check if Ray is installed
echo "🔍 Checking Ray installation..."
python -c "import ray; print(f'Ray version: {ray.__version__}')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Ray not found. Installing Ray..."
    pip install ray[default]==2.47.1
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install Ray. Please install manually."
        exit 1
    fi
fi

# Check if PyTorch is installed
echo "🔍 Checking PyTorch installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ PyTorch not found. Installing PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install PyTorch. Please install manually."
        exit 1
    fi
fi

# Test connection to Head PC 1
echo "🌐 Testing connection to Head PC 1..."
ping -c 2 192.168.1.10 > /dev/null
if [ $? -ne 0 ]; then
    echo "❌ Cannot ping Head PC 1 (192.168.1.10). Please check network connection."
    exit 1
fi
echo "✅ Network connection to Head PC 1 is working"

# Stop any existing Ray processes
echo "🛑 Stopping any existing Ray processes..."
ray stop --force 2>/dev/null

# Connect to the Ray cluster
echo "🚀 Connecting Worker PC 2 to Ray cluster..."
ray start --address='192.168.1.10:6379' --node-ip-address=192.168.1.11 --redis-password=mypassword

# Check if connection was successful
if [ $? -eq 0 ]; then
    echo "✅ SUCCESS! Worker PC 2 connected to Ray cluster"
    echo "📊 You can check cluster status from Head PC 1 with: ray status"
    echo "🌐 Ray dashboard: http://192.168.1.10:8265"
else
    echo "❌ Failed to connect to Ray cluster"
    echo "💡 Troubleshooting tips:"
    echo "   1. Ensure Head PC 1 Ray cluster is running"
    echo "   2. Check firewall settings on both PCs"
    echo "   3. Verify LAN cable connection"
    echo "   4. Check IP addresses are correct"
fi
