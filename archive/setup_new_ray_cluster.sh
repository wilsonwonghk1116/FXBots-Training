#!/bin/bash

echo "🚀 Setting up NEW Ray Cluster with Direct LAN Connection"
echo "=================================================="

# NEW IP CONFIGURATION
HEAD_IP="192.168.1.10"
WORKER_IP="192.168.1.11"
RAY_PORT="6379"

echo "📡 Head PC IP: $HEAD_IP"
echo "📡 Worker PC IP: $WORKER_IP"

# Check if we are on Head PC (192.168.1.10) or Worker PC (192.168.1.11)
CURRENT_IP=$(hostname -I | awk '{print $1}')
echo "🖥️  Current machine IP: $CURRENT_IP"

if [[ "$CURRENT_IP" == "$HEAD_IP" ]]; then
    echo "🎯 This is HEAD PC - Starting Ray head node..."
    
    # Kill any existing Ray processes first
    echo "🔪 Killing existing Ray processes..."
    ray stop --force
    pkill -f "ray::" 2>/dev/null || true
    sleep 2
    
    # Start Ray head node with new IP
    echo "🚀 Starting Ray head node on $HEAD_IP:$RAY_PORT..."
    ray start --head \
        --node-ip-address=$HEAD_IP \
        --port=$RAY_PORT \
        --dashboard-host=0.0.0.0 \
        --dashboard-port=8265 \
        --object-manager-port=8076 \
        --gcs-server-port=8077 \
        --raylet-port=8078 \
        --min-worker-port=10002 \
        --max-worker-port=19999 \
        --verbose
    
    echo "✅ Ray head started! Dashboard: http://$HEAD_IP:8265"
    echo "📋 Worker connection command:"
    echo "   ray start --address='$HEAD_IP:$RAY_PORT'"
    
elif [[ "$CURRENT_IP" == "$WORKER_IP" ]]; then
    echo "🎯 This is WORKER PC - Connecting to head node..."
    
    # Kill any existing Ray processes first
    echo "🔪 Killing existing Ray processes..."
    ray stop --force
    pkill -f "ray::" 2>/dev/null || true
    sleep 2
    
    # Connect to head node with new IP
    echo "🔌 Connecting to Ray head at $HEAD_IP:$RAY_PORT..."
    ray start --address="$HEAD_IP:$RAY_PORT" \
        --node-ip-address=$WORKER_IP \
        --object-manager-port=8076 \
        --raylet-port=8078 \
        --min-worker-port=10002 \
        --max-worker-port=19999 \
        --verbose
    
    echo "✅ Worker connected to head node!"
    
else
    echo "❌ ERROR: Unknown IP address $CURRENT_IP"
    echo "   Expected either $HEAD_IP (head) or $WORKER_IP (worker)"
    echo "   Please check your network configuration!"
    exit 1
fi

# Check cluster status
echo ""
echo "🔍 Checking Ray cluster status..."
sleep 3
ray status

echo ""
echo "🎊 Ray cluster setup complete!"
echo "   To start training: RAY_CLUSTER=1 python run_stable_85_percent_trainer.py" 