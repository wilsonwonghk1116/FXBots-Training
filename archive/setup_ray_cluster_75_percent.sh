#!/bin/bash
# Ray Cluster Setup Script for Maximum 75% Resource Utilization
# Use this script to set up and monitor the Ray cluster for optimal performance

set -e

echo "🚀 Ray Cluster Setup for 75% Resource Utilization"
echo "=================================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to get system info
get_system_info() {
    echo "📊 System Information:"
    echo "   • CPU: $(lscpu | grep 'Model name' | cut -d: -f2 | xargs)"
    echo "   • CPU Cores: $(nproc)"
    echo "   • Total RAM: $(free -h | grep '^Mem:' | awk '{print $2}')"
    
    if command_exists nvidia-smi; then
        echo "   • GPU Info:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | \
        while IFS=, read -r name memory; do
            echo "     - $name: ${memory}MB VRAM"
        done
    else
        echo "   • GPU: No NVIDIA GPU detected"
    fi
    echo ""
}

# Function to check Ray installation
check_ray_installation() {
    echo "🔍 Checking Ray installation..."
    if python3 -c "import ray; print(f'Ray version: {ray.__version__}')" 2>/dev/null; then
        echo "✅ Ray is installed"
    else
        echo "❌ Ray not found. Installing..."
        pip install ray[default]
    fi
    echo ""
}

# Function to check other dependencies
check_dependencies() {
    echo "🔍 Checking dependencies..."
    
    python3 -c "
import sys
required_packages = ['torch', 'numpy', 'pandas', 'psutil', 'GPUtil']
missing = []

for package in required_packages:
    try:
        __import__(package)
        print(f'✅ {package}')
    except ImportError:
        missing.append(package)
        print(f'❌ {package}')

if missing:
    print(f'\\nInstalling missing packages: {missing}')
    sys.exit(1)
else:
    print('\\n✅ All dependencies satisfied')
"
    
    if [ $? -eq 1 ]; then
        echo "Installing missing packages..."
        pip install torch numpy pandas psutil GPUtil
    fi
    echo ""
}

# Function to optimize system settings
optimize_system() {
    echo "⚡ Optimizing system settings for maximum performance..."
    
    # Set CPU governor to performance mode (if available)
    if [ -d "/sys/devices/system/cpu/cpu0/cpufreq" ]; then
        echo "   • Setting CPU governor to performance mode..."
        for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
            if [ -w "$cpu" ]; then
                echo performance | sudo tee "$cpu" > /dev/null
            fi
        done
    fi
    
    # Increase file descriptor limits
    echo "   • Increasing file descriptor limits..."
    ulimit -n 65536
    
    # Set GPU persistence mode (if NVIDIA GPU available)
    if command_exists nvidia-smi; then
        echo "   • Setting GPU persistence mode..."
        sudo nvidia-smi -pm 1 || true
    fi
    
    echo "✅ System optimization complete"
    echo ""
}

# Function to start Ray head node
start_ray_head() {
    echo "🌟 Starting Ray head node..."
    
    # Kill any existing Ray processes
    ray stop 2>/dev/null || true
    
    # Start Ray head node with optimized settings
    ray start --head \
        --port=8265 \
        --dashboard-host=0.0.0.0 \
        --dashboard-port=8265 \
        --object-manager-port=8076 \
        --node-manager-port=8077 \
        --gcs-server-port=8078 \
        --min-worker-port=10000 \
        --max-worker-port=19999 \
        --num-cpus=$(nproc) \
        --memory=$(($(free -b | grep '^Mem:' | awk '{print $2}') * 90 / 100)) \
        --object-store-memory=$(($(free -b | grep '^Mem:' | awk '{print $2}') * 20 / 100))
    
    echo "✅ Ray head node started"
    echo "📊 Dashboard available at: http://$(hostname -I | awk '{print $1}'):8265"
    echo ""
}

# Function to display connection command for worker nodes
show_worker_connection() {
    echo "🔗 To connect worker nodes, run this command on each worker:"
    echo "=================================================="
    HEAD_NODE_IP=$(hostname -I | awk '{print $1}')
    echo "ray start --address='${HEAD_NODE_IP}:10001' \\"
    echo "    --num-cpus=\$(nproc) \\"
    echo "    --memory=\$(($(free -b | grep '^Mem:' | awk '{print $2}') * 90 / 100)) \\"
    echo "    --object-store-memory=\$(($(free -b | grep '^Mem:' | awk '{print $2}') * 20 / 100))"
    echo "=================================================="
    echo ""
}

# Function to monitor cluster resources
monitor_cluster() {
    echo "📊 Ray Cluster Resource Monitoring"
    echo "=================================="
    
    python3 -c "
import ray
import time
import psutil
try:
    import GPUtil
    gpu_available = True
except ImportError:
    gpu_available = False

if not ray.is_initialized():
    try:
        ray.init(address='auto')
    except:
        print('❌ Cannot connect to Ray cluster')
        exit(1)

print('Ray cluster resources:')
resources = ray.cluster_resources()
for resource, amount in resources.items():
    print(f'   • {resource}: {amount}')

print('\\nNode information:')
for node in ray.nodes():
    node_id = node['NodeID']
    alive = node['Alive']
    resources = node['Resources']
    print(f'   • Node {node_id[:8]}... (Alive: {alive})')
    for resource, amount in resources.items():
        print(f'     - {resource}: {amount}')

print('\\nSystem utilization:')
print(f'   • CPU: {psutil.cpu_percent(interval=1):.1f}%')
print(f'   • Memory: {psutil.virtual_memory().percent:.1f}%')

if gpu_available:
    try:
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            print(f'   • GPU {gpu.id} ({gpu.name}): {gpu.load*100:.1f}% util, {gpu.memoryUtil*100:.1f}% vRAM')
    except:
        print('   • GPU info unavailable')
else:
    print('   • GPU monitoring unavailable (GPUtil not installed)')
"
    echo ""
}

# Function to run the ultimate Kelly bot
run_ultimate_bot() {
    echo "🚀 Running Ultimate Kelly Monte Carlo Bot..."
    echo "==========================================="
    
    if [ ! -f "ray_kelly_ultimate_75_percent.py" ]; then
        echo "❌ ray_kelly_ultimate_75_percent.py not found in current directory"
        echo "Please ensure the script is in the current directory"
        exit 1
    fi
    
    echo "⚡ Starting maximum resource utilization simulation..."
    echo "Target: 75% CPU/GPU/vRAM utilization"
    echo ""
    
    python3 ray_kelly_ultimate_75_percent.py
}

# Function to show help
show_help() {
    echo "Ray Cluster Setup and Management Script"
    echo "======================================"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  setup     - Complete setup (check deps, optimize, start head node)"
    echo "  head      - Start Ray head node only"
    echo "  monitor   - Monitor cluster resources"
    echo "  run       - Run the ultimate Kelly bot simulation"
    echo "  info      - Show system information"
    echo "  help      - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 setup     # Complete setup on head node"
    echo "  $0 monitor   # Monitor cluster resources"
    echo "  $0 run       # Run the trading bot simulation"
}

# Main script logic
case "${1:-setup}" in
    "setup")
        get_system_info
        check_ray_installation
        check_dependencies
        optimize_system
        start_ray_head
        show_worker_connection
        echo "🎯 Setup complete! Ready to run simulations."
        echo "Use '$0 run' to start the ultimate Kelly bot simulation."
        ;;
    "head")
        start_ray_head
        show_worker_connection
        ;;
    "monitor")
        monitor_cluster
        ;;
    "run")
        run_ultimate_bot
        ;;
    "info")
        get_system_info
        ;;
    "help"|"-h"|"--help")
        show_help
        ;;
    *)
        echo "❌ Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac
