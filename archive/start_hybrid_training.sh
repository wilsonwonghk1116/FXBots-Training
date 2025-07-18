#!/bin/bash

echo "🚀 === ULTIMATE HYBRID PROCESSING SYSTEM ==="
echo "🎯 TARGET: 60 CPU Threads + 75%+ GPU + 85% VRAM"
echo ""

# Check system requirements
echo "📋 System Check:"
python -c "
import torch
import multiprocessing
import GPUtil

print(f'✅ CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ GPU: {torch.cuda.get_device_name()}') 
    gpu = GPUtil.getGPUs()[0]
    print(f'✅ VRAM: {gpu.memoryTotal}MB available')

print(f'✅ CPU Cores: {multiprocessing.cpu_count()}')
print(f'✅ Optimal Threads: {min(60, multiprocessing.cpu_count() * 4)}')
"

echo ""
echo "🚀 Starting Hybrid Training System..."
echo "📊 Monitoring: hybrid_training.log"

# Start the hybrid system
nohup python run_hybrid_75_gpu_60_cpu.py > hybrid_training.log 2>&1 &
HYBRID_PID=$!

echo "✅ Hybrid system started (PID: $HYBRID_PID)"
echo ""
echo "🔍 Monitoring performance for 60 seconds..."

# Monitor for 60 seconds
for i in {1..12}; do
    echo "Check $i/12:"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | \
    awk -F',' '{printf "  🔥 GPU: %s%% | 💾 VRAM: %.1fGB/%.1fGB (%.1f%%) | 🌡️ %s°C\n", $1, $2/1024, $3/1024, ($2/$3)*100, $4}'
    
    if [ $i -eq 6 ]; then
        echo "  📋 Log status:" 
        tail -2 hybrid_training.log | sed 's/^/    /'
    fi
    
    sleep 5
done

echo ""
echo "🎯 Hybrid system is running!"
echo "📊 Monitor logs: tail -f hybrid_training.log"
echo "🛑 Stop system: pkill -f run_hybrid_75_gpu_60_cpu.py" 