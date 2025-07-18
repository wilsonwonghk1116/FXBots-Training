#!/bin/bash
# Quick Resume Script for RTX 3090 + RTX 3070 Dual-GPU Optimizer
# Run this after PC restart to quickly resume work

echo "🔄 RESUMING RTX 3090 + RTX 3070 DUAL-GPU OPTIMIZER PROJECT"
echo "=========================================================="

# 1. Activate conda environment
echo "📦 Activating BotsTraining_env..."
conda activate BotsTraining_env
if [ $? -eq 0 ]; then
    echo "✅ Environment activated"
else
    echo "❌ Failed to activate environment"
    exit 1
fi

# 2. Start Ray cluster
echo "🚀 Starting Ray cluster..."
ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265 > /dev/null 2>&1 &
sleep 3

# 3. Check Ray status
echo "📊 Checking Ray cluster status..."
ray status

# 4. Verify file syntax
echo "🔍 Verifying code syntax..."
python -m py_compile rtx3090_smart_compute_optimizer_dual_gpu.py
if [ $? -eq 0 ]; then
    echo "✅ Code syntax OK"
else
    echo "❌ Syntax error in code"
    exit 1
fi

echo ""
echo "🎯 READY TO RESUME!"
echo "Test with: python rtx3090_smart_compute_optimizer_dual_gpu.py --duration=3"
echo "Monitor: CPU should be 70-85%, both GPUs active, no system freezing"
