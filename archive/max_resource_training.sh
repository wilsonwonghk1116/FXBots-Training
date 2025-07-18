#!/bin/bash
# Direct Maximum Resource Utilization Training Command
# Pushes both PCs to 75% CPU, GPU, VRAM usage with overclocked VRAM

echo "🚀 MAXIMUM RESOURCE UTILIZATION TRAINING"
echo "========================================="
echo "Target: 75% CPU + 75% GPU + 75% VRAM"
echo "Optimized for overclocked VRAM"
echo ""

# Check current resource usage
echo "📊 Current Resource Usage:"
echo "CPU: $(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1"%"}')"

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | \
    while IFS=, read -r gpu_util mem_used mem_total temp; do
        mem_percent=$(echo "scale=1; $mem_used * 100 / $mem_total" | bc -l 2>/dev/null || echo "0")
        echo "  GPU: ${gpu_util}%, VRAM: ${mem_percent}%, Temp: ${temp}°C"
    done
fi
echo ""

# Start GUI monitoring in background
echo "🖥️  Starting GUI monitoring..."
python3 generate_demo_fleet_data.py > /dev/null 2>&1
python3 kelly_bot_dashboard.py > /dev/null 2>&1 &
python3 ray_cluster_monitor_75_percent.py > /dev/null 2>&1 &
echo "   ✅ GUI components started"
echo ""

# Main training command for maximum resource utilization
echo "🎯 Launching MAXIMUM RESOURCE SATURATION training..."
echo "   • Target: 75% CPU utilization"
echo "   • Target: 75% GPU utilization" 
echo "   • Target: 75% VRAM utilization"
echo "   • Optimized for overclocked VRAM"
echo ""

# Run the main training with all optimizations
python3 ray_kelly_ultimate_75_percent.py

echo ""
echo "✅ Maximum resource utilization training session completed!"
