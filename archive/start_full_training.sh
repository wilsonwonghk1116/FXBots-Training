#!/bin/bash
# Complete Ray Cluster Training Launcher with GUI Monitoring
# This script sets up and starts the full Kelly Monte Carlo training system

set -e

echo "🚀 Starting Kelly Monte Carlo Ray Cluster Training with GUI"
echo "============================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
HEAD_NODE_IP="192.168.1.100"  # Change this to your head node IP
RAY_PORT=8265
DASHBOARD_PORT=8266
TRAINING_DURATION_HOURS=24  # 24 hour training session
N_BOTS=2000  # Full 2000 bot fleet

echo -e "${BLUE}Configuration:${NC}"
echo "  • Head Node IP: $HEAD_NODE_IP"
echo "  • Ray Port: $RAY_PORT"
echo "  • Dashboard Port: $DASHBOARD_PORT"
echo "  • Training Duration: $TRAINING_DURATION_HOURS hours"
echo "  • Bot Fleet Size: $N_BOTS bots"
echo ""

# Function to check if we're on head node or worker
check_node_type() {
    local_ip=$(hostname -I | awk '{print $1}')
    if [[ "$local_ip" == "$HEAD_NODE_IP" ]]; then
        echo "HEAD"
    else
        echo "WORKER"
    fi
}

# Function to start Ray head node
start_ray_head() {
    echo -e "${GREEN}🔧 Starting Ray Head Node...${NC}"
    
    # Kill any existing Ray processes
    ray stop --force 2>/dev/null || true
    sleep 2
    
    # Start Ray head with optimized settings
    ray start \
        --head \
        --port=$RAY_PORT \
        --dashboard-host=0.0.0.0 \
        --dashboard-port=$DASHBOARD_PORT \
        --num-cpus=$(nproc) \
        --num-gpus=$(nvidia-smi -L | wc -l) \
        --memory=$(free -b | grep '^Mem:' | awk '{printf "%.0f", $2 * 0.75}') \
        --object-store-memory=$(free -b | grep '^Mem:' | awk '{printf "%.0f", $2 * 0.15}') \
        --include-dashboard=true \
        --disable-usage-stats
    
    echo -e "${GREEN}✅ Ray head node started${NC}"
    echo -e "   • Ray Dashboard: http://$HEAD_NODE_IP:$DASHBOARD_PORT"
    echo -e "   • Connect workers with: ray start --address='$HEAD_NODE_IP:10001'"
}

# Function to start Ray worker
start_ray_worker() {
    echo -e "${GREEN}🔧 Starting Ray Worker Node...${NC}"
    
    # Kill any existing Ray processes
    ray stop --force 2>/dev/null || true
    sleep 2
    
    # Start Ray worker
    ray start \
        --address="$HEAD_NODE_IP:10001" \
        --num-cpus=$(nproc) \
        --num-gpus=$(nvidia-smi -L | wc -l) \
        --memory=$(free -b | grep '^Mem:' | awk '{printf "%.0f", $2 * 0.75}') \
        --object-store-memory=$(free -b | grep '^Mem:' | awk '{printf "%.0f", $2 * 0.15}')
    
    echo -e "${GREEN}✅ Ray worker node connected${NC}"
}

# Function to start resource monitor
start_resource_monitor() {
    echo -e "${YELLOW}📊 Starting Resource Monitor GUI...${NC}"
    
    # Start the resource monitor in background
    python3 ray_cluster_monitor_75_percent.py &
    MONITOR_PID=$!
    echo "   • Monitor PID: $MONITOR_PID"
    
    # Give monitor time to start
    sleep 3
}

# Function to start bot dashboard
start_bot_dashboard() {
    echo -e "${YELLOW}🤖 Starting Bot Performance Dashboard...${NC}"
    
    # Generate fresh demo data
    python3 generate_demo_fleet_data.py
    
    # Start the bot dashboard in background
    python3 kelly_bot_dashboard.py &
    DASHBOARD_PID=$!
    echo "   • Dashboard PID: $DASHBOARD_PID"
    
    # Give dashboard time to start
    sleep 3
}

# Function to start main training
start_training() {
    echo -e "${GREEN}🎯 Starting Main Kelly Monte Carlo Training...${NC}"
    echo "   • Fleet size: $N_BOTS bots"
    echo "   • Training duration: $TRAINING_DURATION_HOURS hours"
    echo "   • Resource utilization target: 75%"
    echo ""
    
    # Create training session directory
    SESSION_DIR="training_session_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$SESSION_DIR"
    echo "   • Session directory: $SESSION_DIR"
    
    # Start the main training script
    python3 ray_kelly_ultimate_75_percent.py \
        --n-bots $N_BOTS \
        --duration-hours $TRAINING_DURATION_HOURS \
        --session-dir "$SESSION_DIR" \
        --auto-save-interval 3600 \
        --gpu-utilization-target 0.75 \
        --cpu-utilization-target 0.75 \
        2>&1 | tee "$SESSION_DIR/training.log"
}

# Function to show cluster status
show_cluster_status() {
    echo -e "${BLUE}📈 Cluster Status:${NC}"
    
    # Ray cluster status
    echo "Ray Cluster Nodes:"
    ray status 2>/dev/null || echo "  Ray cluster not accessible"
    
    # System resources
    echo ""
    echo "System Resources:"
    echo "  • CPU Usage: $(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1"%"}')"
    echo "  • Memory Usage: $(free | grep Mem | awk '{printf("%.1f%%", $3/$2 * 100.0)}')"
    
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "  • GPU Usage:"
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
        while IFS=, read -r gpu_util mem_used mem_total; do
            mem_percent=$(echo "scale=1; $mem_used * 100 / $mem_total" | bc)
            echo "    - GPU: ${gpu_util}%, VRAM: ${mem_percent}%"
        done
    fi
    echo ""
}

# Function to cleanup on exit
cleanup() {
    echo -e "${RED}🧹 Cleaning up...${NC}"
    
    # Kill background processes
    if [[ -n "$MONITOR_PID" ]]; then
        kill $MONITOR_PID 2>/dev/null || true
    fi
    
    if [[ -n "$DASHBOARD_PID" ]]; then
        kill $DASHBOARD_PID 2>/dev/null || true
    fi
    
    # Note: Don't stop Ray here as training might still be running
    echo -e "${YELLOW}⚠️  Ray cluster left running for continued training${NC}"
    echo -e "   To stop Ray: ray stop"
}

# Set up signal handlers
trap cleanup EXIT INT TERM

# Main execution
main() {
    echo -e "${BLUE}🔍 Checking system requirements...${NC}"
    
    # Check dependencies
    if ! command -v ray >/dev/null 2>&1; then
        echo -e "${RED}❌ Ray not found. Please install: pip install ray[default]${NC}"
        exit 1
    fi
    
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo -e "${YELLOW}⚠️  NVIDIA GPU not detected${NC}"
    fi
    
    # Determine node type
    NODE_TYPE=$(check_node_type)
    echo -e "${BLUE}📍 Node type detected: $NODE_TYPE${NC}"
    echo ""
    
    # Start Ray cluster
    if [[ "$NODE_TYPE" == "HEAD" ]]; then
        start_ray_head
        sleep 5  # Give head node time to initialize
        
        # Start monitoring and dashboards on head node
        start_resource_monitor
        start_bot_dashboard
        
        echo ""
        echo -e "${GREEN}🎉 Head node setup complete!${NC}"
        echo -e "   • Ray Dashboard: http://$HEAD_NODE_IP:$DASHBOARD_PORT"
        echo -e "   • Resource Monitor: Running in background"
        echo -e "   • Bot Dashboard: Running in background"
        echo ""
        echo -e "${YELLOW}📋 Next steps:${NC}"
        echo -e "   1. Connect worker nodes with: ray start --address='$HEAD_NODE_IP:10001'"
        echo -e "   2. Press Enter to start training, or Ctrl+C to setup only"
        
        # Wait for user input
        read -p "Press Enter to start training..."
        
        # Show final status before training
        show_cluster_status
        
        # Start training
        start_training
        
    else
        start_ray_worker
        
        echo ""
        echo -e "${GREEN}🎉 Worker node connected!${NC}"
        echo -e "   • Connected to head: $HEAD_NODE_IP"
        echo -e "   • Worker is ready for tasks"
        echo ""
        echo -e "${YELLOW}ℹ️  Worker node will receive tasks from head node${NC}"
        echo -e "   Monitor progress at: http://$HEAD_NODE_IP:$DASHBOARD_PORT"
        
        # Keep worker alive
        echo "   Keeping worker alive... (Ctrl+C to stop)"
        while true; do
            sleep 60
            show_cluster_status
        done
    fi
}

# Run main function
main "$@"
