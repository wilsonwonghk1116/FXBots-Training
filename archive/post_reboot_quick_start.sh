#!/bin/bash
# POST-REBOOT QUICK START SCRIPT
# 黃子華 Style: "One Script to Rule Them All!" 👑

echo "🚀 POST-REBOOT FOREX BOT TRAINING QUICK START"
echo "=============================================="
echo "黃子華話：'Reboot 完最緊要識點重新開始！'"
echo ""

# Check if we're on Head PC or Worker PC
HOSTNAME=$(hostname)
if [[ "$HOSTNAME" == *"w1"* ]]; then
    echo "📍 Detected: HEAD PC 1 (w1)"
    IS_HEAD=true
    PC_IP="192.168.1.10"
    PROJECT_PATH="/home/w1/cursor-to-copilot-backup/TaskmasterForexBots"
elif [[ "$HOSTNAME" == *"w2"* ]]; then
    echo "📍 Detected: WORKER PC 2 (w2)"
    IS_HEAD=false
    PC_IP="192.168.1.11"
    PROJECT_PATH="/home/w2/cursor-to-copilot-backup/TaskmasterForexBots"
else
    echo "❌ Unknown PC! Please run this on w1 or w2"
    exit 1
fi

echo "🌐 Using IP: $PC_IP"
echo "📁 Project Path: $PROJECT_PATH"
echo ""

# Step 1: Network Verification
echo "🔍 Step 1: Network Verification"
if $IS_HEAD; then
    echo "🏠 HEAD PC - Pinging Worker PC..."
    if ping -c 3 192.168.1.11 > /dev/null 2>&1; then
        echo "✅ Worker PC (192.168.1.11) is reachable"
    else
        echo "❌ Cannot reach Worker PC (192.168.1.11)"
        echo "💡 Make sure Worker PC is booted and LAN cable connected"
        exit 1
    fi
else
    echo "👷 WORKER PC - Pinging Head PC..."
    if ping -c 3 192.168.1.10 > /dev/null 2>&1; then
        echo "✅ Head PC (192.168.1.10) is reachable"
    else
        echo "❌ Cannot reach Head PC (192.168.1.10)"
        echo "💡 Make sure Head PC is booted and LAN cable connected"
        exit 1
    fi
fi
echo ""

# Step 2: Change to project directory
echo "🔍 Step 2: Navigating to Project Directory"
cd "$PROJECT_PATH" || {
    echo "❌ Cannot access project directory: $PROJECT_PATH"
    exit 1
}
echo "✅ Changed to: $(pwd)"
echo ""

# Step 3: Activate Conda Environment
echo "🔍 Step 3: Activating Conda Environment"
if conda activate BotsTraining_env 2>/dev/null; then
    echo "✅ Conda environment 'BotsTraining_env' activated"
else
    echo "❌ Failed to activate conda environment"
    echo "💡 Try: conda activate BotsTraining_env"
    exit 1
fi

# Verify Python and packages
PYTHON_VERSION=$(python --version 2>&1)
echo "🐍 Python Version: $PYTHON_VERSION"

if python -c "import ray; print(f'Ray Version: {ray.__version__}')" 2>/dev/null; then
    echo "✅ Ray is available"
else
    echo "❌ Ray not found in environment"
    exit 1
fi

if python -c "import synthetic_env" 2>/dev/null; then
    echo "✅ synthetic_env module available"
else
    echo "❌ synthetic_env module not found"
    exit 1
fi
echo ""

# Step 4: Ray Cluster Setup
echo "🔍 Step 4: Ray Cluster Setup"

# Stop any existing Ray processes
echo "🛑 Stopping any existing Ray processes..."
ray stop 2>/dev/null || true
sleep 2

if $IS_HEAD; then
    echo "🏠 Starting HEAD node on IP: $PC_IP"
    ray start --head --node-ip-address=$PC_IP --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265
    
    if [ $? -eq 0 ]; then
        echo "✅ HEAD node started successfully"
        echo "🌐 Dashboard: http://$PC_IP:8265"
        echo ""
        echo "📋 Next Steps for Worker PC:"
        echo "   1. Boot Worker PC (w2)"
        echo "   2. Run this same script: ./post_reboot_quick_start.sh"
        echo "   3. Worker will auto-connect to this HEAD node"
        echo ""
        echo "⏳ Waiting 10 seconds for cluster to stabilize..."
        sleep 10
        
        echo "🔍 Checking cluster status..."
        ray status
        
        echo ""
        echo "🎯 Ready for Training!"
        echo "   Run: python launch_distributed_training.py"
        echo ""
        echo "黃子華話：'一切準備就緒，就等你話 GO！'"
    else
        echo "❌ Failed to start HEAD node"
        exit 1
    fi
else
    echo "👷 Connecting to HEAD node: 192.168.1.10:6379"
    ray start --address='192.168.1.10:6379'
    
    if [ $? -eq 0 ]; then
        echo "✅ Successfully connected to HEAD node"
        echo ""
        echo "🎯 Worker PC Ready!"
        echo "   Go to HEAD PC and run: python launch_distributed_training.py"
        echo ""
        echo "黃子華話：'Worker 準備就緒，等 Boss 發施號令！'"
    else
        echo "❌ Failed to connect to HEAD node"
        echo "💡 Make sure HEAD PC started first"
        exit 1
    fi
fi

echo ""
echo "🏆 QUICK START COMPLETED!"
echo "================================"
echo "📊 Cluster Configuration:"
echo "   - Expected CPUs: 96 total"
echo "   - Expected GPUs: 2 total (RTX 3090 + RTX 3070)"
echo "   - Head PC: 80 CPUs + 1 GPU (RTX 3090)"
echo "   - Worker PC: 16 CPUs + 1 GPU (RTX 3070)"
echo ""
echo "🎯 Expected Training Performance:"
echo "   - Population: 5,000-8,000 bots/generation"
echo "   - Generations: 300 total"
echo "   - Training Time: 4-6 hours"
echo ""
echo "黃子華終極金句：'Reboot 完最緊要係即刻返番工作狀態！'"
echo "準備好就話俾我知：'我準備好寸你啦！' 😂"
