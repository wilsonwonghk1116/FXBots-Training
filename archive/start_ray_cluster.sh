#!/bin/bash
"""
Ray Cluster Startup Script for Kelly Monte Carlo Bot System

This script sets up and starts the Ray cluster on both PCs for maximum
CPU/GPU utilization targeting 75% resource usage.

Head PC: Xeon E5 x2 (80 threads) + RTX 3090 (24GB)
Worker PC: i9 (16 threads) + RTX 3070 (8GB)

Usage:
1. Run on Head PC: ./start_ray_cluster.sh head
2. Run on Worker PC: ./start_ray_cluster.sh worker <head_ip>

Author: TaskMaster AI System
Date: 2025-01-12
"""

set -e

# Configuration
HEAD_PORT=10001
DASHBOARD_PORT=8265
OBJECT_STORE_MEMORY=8000000000  # 8GB object store
PLASMA_DIRECTORY="/tmp/ray_plasma"

# Function to detect system resources
detect_resources() {
    local cpu_count=$(nproc)
    local memory_gb=$(free -g | awk '/^Mem:/{print $2}')
    local gpu_count=0
    
    if command -v nvidia-smi >/dev/null 2>&1; then
        gpu_count=$(nvidia-smi --list-gpus | wc -l)
    fi
    
    echo "Detected system resources:"
    echo "- CPUs: $cpu_count"
    echo "- Memory: ${memory_gb}GB"
    echo "- GPUs: $gpu_count"
    echo ""
}

# Function to start Ray head node
start_head_node() {
    echo "Starting Ray HEAD node..."
    detect_resources
    
    # Kill any existing Ray processes
    ray stop -f 2>/dev/null || true
    
    # Clean up plasma directory
    rm -rf $PLASMA_DIRECTORY
    mkdir -p $PLASMA_DIRECTORY
    
    # Start Ray head with optimized settings for maximum performance
    ray start \
        --head \
        --port=$HEAD_PORT \
        --dashboard-host=0.0.0.0 \
        --dashboard-port=$DASHBOARD_PORT \
        --object-store-memory=$OBJECT_STORE_MEMORY \
        --plasma-directory=$PLASMA_DIRECTORY \
        --num-cpus=80 \
        --num-gpus=1 \
        --verbose
    
    echo ""
    echo "Ray HEAD node started successfully!"
    echo "Dashboard: http://$(hostname -I | awk '{print $1}'):$DASHBOARD_PORT"
    echo "Connection command for workers:"
    echo "ray start --address='$(hostname -I | awk '{print $1}'):$HEAD_PORT'"
    echo ""
}

# Function to start Ray worker node
start_worker_node() {
    local head_ip=$1
    
    if [ -z "$head_ip" ]; then
        echo "Error: Head node IP address required"
        echo "Usage: $0 worker <head_ip>"
        exit 1
    fi
    
    echo "Starting Ray WORKER node connecting to $head_ip..."
    detect_resources
    
    # Kill any existing Ray processes
    ray stop -f 2>/dev/null || true
    
    # Clean up plasma directory
    rm -rf $PLASMA_DIRECTORY
    mkdir -p $PLASMA_DIRECTORY
    
    # Start Ray worker with optimized settings
    ray start \
        --address="$head_ip:$HEAD_PORT" \
        --object-store-memory=$OBJECT_STORE_MEMORY \
        --plasma-directory=$PLASMA_DIRECTORY \
        --num-cpus=16 \
        --num-gpus=1 \
        --verbose
    
    echo ""
    echo "Ray WORKER node connected successfully to $head_ip!"
    echo ""
}

# Function to stop Ray cluster
stop_cluster() {
    echo "Stopping Ray cluster..."
    ray stop -f
    rm -rf $PLASMA_DIRECTORY
    echo "Ray cluster stopped."
}

# Function to check cluster status
check_status() {
    echo "Ray Cluster Status:"
    echo "==================="
    
    if ray status >/dev/null 2>&1; then
        ray status
        echo ""
        echo "Ray cluster is running."
    else
        echo "Ray cluster is not running."
    fi
}

# Function to setup Python environment
setup_environment() {
    echo "Setting up Python environment for Ray Kelly Bot System..."
    
    # Install required packages
    pip install -q ray[default] torch torchvision torchaudio pandas numpy matplotlib seaborn tqdm
    
    # Verify installations
    python -c "import ray, torch; print(f'Ray: {ray.__version__}, PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
    
    echo "Environment setup complete!"
}

# Main execution
case "${1:-help}" in
    "head")
        start_head_node
        ;;
    "worker")
        start_worker_node $2
        ;;
    "stop")
        stop_cluster
        ;;
    "status")
        check_status
        ;;
    "setup")
        setup_environment
        ;;
    "help"|*)
        echo "Ray Cluster Management Script"
        echo "============================"
        echo ""
        echo "Usage: $0 <command> [options]"
        echo ""
        echo "Commands:"
        echo "  head              Start Ray head node (run on Head PC)"
        echo "  worker <head_ip>  Start Ray worker node (run on Worker PC)"
        echo "  stop              Stop Ray cluster"
        echo "  status            Check cluster status"
        echo "  setup             Setup Python environment"
        echo "  help              Show this help message"
        echo ""
        echo "Example workflow:"
        echo "1. Head PC:   $0 setup && $0 head"
        echo "2. Worker PC: $0 setup && $0 worker 192.168.1.100"
        echo "3. Run simulation: python ray_distributed_kelly_bot.py"
        echo "4. Stop:      $0 stop"
        echo ""
        ;;
esac
