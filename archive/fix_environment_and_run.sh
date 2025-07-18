#!/bin/bash

echo "🔧 DUAL-GPU ENVIRONMENT & VRAM CLEANUP SETUP"
echo "============================================="

# Check current Python version
echo "🐍 Current Python version:"
python --version

# Check if we're in BotsTraining_env
if [[ "$CONDA_DEFAULT_ENV" == "BotsTraining_env" ]]; then
    echo "✅ Already in BotsTraining_env"
else
    echo "⚠️ Not in BotsTraining_env. Current environment: $CONDA_DEFAULT_ENV"
    echo "💡 Recommendation: Run 'conda activate BotsTraining_env' first"
fi

# Check Ray status
echo ""
echo "🔍 Checking Ray cluster status..."
ray status 2>/dev/null || echo "❌ Ray cluster not running or not accessible"

# Check GPU availability
echo ""
echo "🎮 GPU Status:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits

# Check CUDA availability in Python
echo ""
echo "🔍 CUDA availability check:"
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        total_mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f'   Total VRAM: {total_mem:.1f}GB')
"

echo ""
echo "🧹 Performing comprehensive VRAM cleanup..."

# System-level GPU reset (if possible)
sudo nvidia-smi --gpu-reset 2>/dev/null || echo "⚠️ GPU reset requires sudo (optional)"

# Python-level cleanup
python -c "
import torch
import gc

if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print(f'🧹 Cleaning {device_count} GPU(s)...')
    
    for i in range(device_count):
        torch.cuda.set_device(i)
        allocated_before = torch.cuda.memory_allocated(i) / 1024**3
        cached_before = torch.cuda.memory_reserved(i) / 1024**3
        
        # Comprehensive cleanup
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        
        allocated_after = torch.cuda.memory_allocated(i) / 1024**3
        cached_after = torch.cuda.memory_reserved(i) / 1024**3
        
        print(f'GPU {i}: {allocated_before:.2f}GB -> {allocated_after:.2f}GB allocated')
        print(f'        {cached_before:.2f}GB -> {cached_after:.2f}GB cached')
    
    print('✅ VRAM cleanup completed')
else:
    print('❌ CUDA not available')
"

echo ""
echo "🚀 Environment setup completed!"
echo ""
echo "💡 Next steps:"
echo "1. If not in BotsTraining_env: conda activate BotsTraining_env"
echo "2. Ensure Ray cluster is running on both PCs"
echo "3. Run: python rtx3090_smart_compute_optimizer_dual_gpu.py --duration=2"
echo ""
echo "🎯 For maximum performance:"
echo "- Head PC (RTX 3090): Should show 80-90% GPU utilization"
echo "- Worker PC 2 (RTX 3070): Should show 75-85% GPU utilization"
echo "- Both PCs: Should show 70-80% CPU utilization"
