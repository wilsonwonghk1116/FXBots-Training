#!/bin/bash
# Quick Resume Script for TaskmasterForexBots Project
# Run this script to quickly resume your distributed training session

echo "🚀 TaskmasterForexBots - Quick Resume Script"
echo "============================================="

echo "📍 Current directory: $(pwd)"
echo "📅 Resume date: $(date)"

echo "🔍 Checking Python environment..."
if conda info --envs | grep -q "Training_env"; then
    echo "✅ Training_env conda environment found"
    echo "🔄 Activating Training_env..."
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate Training_env
    echo "✅ Environment activated: $(conda info --show-active)"
else
    echo "❌ Training_env not found. Please create it first:"
    echo "conda create -n Training_env python=3.12.2"
    echo "conda activate Training_env"
    echo "pip install ray[default] torch PyQt6 numpy pandas psutil gputil"
    exit 1
fi

echo ""
echo "🔍 Checking Ray cluster status..."
if command -v ray &> /dev/null; then
    echo "✅ Ray is installed"
    
    # Check if Ray is already running
    if ray status &> /dev/null; then
        echo "✅ Ray cluster is already running"
        ray status
    else
        echo "⚠️ Ray cluster not running. Please start it manually:"
        echo ""
        echo "PC1 (Head Node - 192.168.1.10):"
        echo "ray start --head --node-ip-address=192.168.1.10"
        echo ""
        echo "PC2 (Worker Node - 192.168.1.11):"
        echo "ray start --address='192.168.1.10:6379' --node-ip-address=192.168.1.11"
        echo ""
        echo "Then run this script again."
        exit 1
    fi
else
    echo "❌ Ray not installed. Installing..."
    pip install ray[default]
fi

echo ""
echo "🔍 Checking required Python packages..."
python -c "
import sys
missing = []
try:
    import ray
    print('✅ Ray available')
except ImportError:
    missing.append('ray[default]')

try:
    import torch
    print('✅ PyTorch available')
except ImportError:
    missing.append('torch')

try:
    from PyQt6.QtWidgets import QApplication
    print('✅ PyQt6 available')
except ImportError:
    missing.append('PyQt6')

try:
    import numpy, pandas, psutil
    print('✅ NumPy, Pandas, psutil available')
except ImportError:
    missing.append('numpy pandas psutil')

if missing:
    print('❌ Missing packages:', ' '.join(missing))
    print('Install with: pip install', ' '.join(missing))
    sys.exit(1)
else:
    print('✅ All required packages available')
"

if [ $? -ne 0 ]; then
    echo "❌ Package check failed. Please install missing packages."
    exit 1
fi

echo ""
echo "🔍 Checking main training file..."
if [ -f "fixed_integrated_training_75_percent.py" ]; then
    echo "✅ Main training file found"
else
    echo "❌ Main training file not found. Please ensure you're in the correct directory."
    exit 1
fi

echo ""
echo "🎯 Everything looks good! Ready to resume training."
echo ""
echo "🚀 Starting the FIXED Kelly Monte Carlo Trading Fleet..."
echo "📊 Target: 75% CPU/GPU/VRAM utilization on both PC1 & PC2"
echo ""

# Start the main application
python fixed_integrated_training_75_percent.py

echo ""
echo "🔄 Training session ended."
echo "💾 Project state saved in PROJECT_STATE_SAVE_20250713.md"
