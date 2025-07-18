#!/bin/bash

# Maximum Resource Utilization Training Launcher
# Designed to push CPU/GPU/VRAM to 75% utilization and fix training failures

echo "🚀 MAXIMUM RESOURCE UTILIZATION KELLY BOT SYSTEM"
echo "=================================================="
echo "Target: 75% CPU/GPU/VRAM utilization"
echo "Fixes training failures and ensures proper resource saturation"
echo ""

# Check system resources
echo "📊 System Resource Check:"
echo "CPU Cores: $(nproc)"
echo "Total RAM: $(free -h | awk '/^Mem:/ {print $2}')"
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Info:"
    nvidia-smi --query-gpu=name,memory.total,utilization.gpu --format=csv,noheader,nounits | head -2
else
    echo "GPU: Not detected or nvidia-smi not available"
fi
echo ""

# Set environment variables for maximum performance
export OMP_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=$(nproc)
export CUDA_LAUNCH_BLOCKING=0
export PYTHONUNBUFFERED=1

echo "🔧 Environment configured for maximum utilization:"
echo "   OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "   MKL_NUM_THREADS: $MKL_NUM_THREADS"
echo "   CUDA_LAUNCH_BLOCKING: $CUDA_LAUNCH_BLOCKING"
echo ""

# Function to check if Python packages are available
check_package() {
    python3 -c "import $1" 2>/dev/null
    return $?
}

# Check required packages
echo "📦 Checking required packages..."
packages_needed=()

if ! check_package "torch"; then
    packages_needed+=("torch")
fi

if ! check_package "PyQt6"; then
    packages_needed+=("PyQt6")
fi

if ! check_package "psutil"; then
    packages_needed+=("psutil")
fi

if ! check_package "numpy"; then
    packages_needed+=("numpy")
fi

if ! check_package "pandas"; then
    packages_needed+=("pandas")
fi

# Install missing packages if needed
if [ ${#packages_needed[@]} -gt 0 ]; then
    echo "⚠️  Missing packages detected: ${packages_needed[*]}"
    echo "Installing missing packages..."
    
    for package in "${packages_needed[@]}"; do
        echo "Installing $package..."
        if [ "$package" = "torch" ]; then
            pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        else
            pip3 install "$package"
        fi
    done
    echo "✅ Package installation completed"
else
    echo "✅ All required packages are available"
fi
echo ""

# Create launch menu
echo "🎛️  LAUNCH OPTIONS:"
echo "1) Enhanced GUI with Maximum Utilization Training"
echo "2) Terminal-only Maximum Utilization System"
echo "3) Resource Monitor Only"
echo "4) Quick System Test"
echo "5) Exit"
echo ""

read -p "Select option (1-5): " choice

case $choice in
    1)
        echo ""
        echo "🚀 Launching Enhanced GUI with Maximum Utilization Training..."
        echo "This will push your CPU/GPU/VRAM to 75% utilization"
        echo "Press Ctrl+C to stop at any time"
        echo ""
        
        # Launch the enhanced GUI system
        python3 enhanced_integrated_training_max_util.py
        ;;
        
    2)
        echo ""
        echo "🚀 Launching Terminal-only Maximum Utilization System..."
        echo "This will run intensive training without GUI"
        echo "Press Ctrl+C to stop at any time"
        echo ""
        
        # Launch terminal-only system
        python3 max_utilization_system.py
        ;;
        
    3)
        echo ""
        echo "📊 Launching Resource Monitor..."
        echo "Real-time CPU/GPU/VRAM monitoring"
        echo "Press Ctrl+C to stop"
        echo ""
        
        # Simple resource monitor
        while true; do
            clear
            echo "📊 REAL-TIME RESOURCE MONITOR"
            echo "=============================="
            echo "Time: $(date)"
            echo ""
            echo "CPU Usage:"
            top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1
            echo ""
            echo "Memory Usage:"
            free -h
            echo ""
            if command -v nvidia-smi &> /dev/null; then
                echo "GPU Status:"
                nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits
            fi
            echo ""
            echo "Press Ctrl+C to exit..."
            sleep 2
        done
        ;;
        
    4)
        echo ""
        echo "🧪 Running Quick System Test..."
        echo "Testing CPU and GPU capabilities..."
        echo ""
        
        # Quick system test
        python3 -c "
import time
import multiprocessing as mp
import psutil
print(f'✅ CPU Cores: {mp.cpu_count()}')
print(f'✅ Available RAM: {psutil.virtual_memory().total // (1024**3)} GB')

try:
    import torch
    if torch.cuda.is_available():
        print(f'✅ CUDA Available: {torch.cuda.device_count()} GPU(s)')
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_mem = torch.cuda.get_device_properties(i).total_memory // (1024**3)
            print(f'   GPU {i}: {gpu_name} ({gpu_mem} GB)')
    else:
        print('⚠️  CUDA not available')
except ImportError:
    print('⚠️  PyTorch not installed')

try:
    import PyQt6
    print('✅ PyQt6 available for GUI')
except ImportError:
    print('⚠️  PyQt6 not available - GUI disabled')

print('')
print('🎯 System is ready for maximum utilization training!')
print('   Target: 75% CPU/GPU/VRAM utilization')
"
        echo ""
        echo "Test completed. Press Enter to return to menu..."
        read
        exec "$0"  # Restart script
        ;;
        
    5)
        echo "👋 Goodbye!"
        exit 0
        ;;
        
    *)
        echo "❌ Invalid option. Please try again."
        sleep 2
        exec "$0"  # Restart script
        ;;
esac

echo ""
echo "🏁 Session completed."
echo "Check fleet_results.json for training results."
echo "Check max_utilization_results_*.json for performance analysis."
