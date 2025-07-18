#!/bin/bash

# Manual Dual PC Ray Cluster Setup
# Following user's exact 9-step instructions with SSH password 'w'

echo "🚀 Manual Dual PC Ray Cluster Setup"
echo "Following your exact 9-step instructions"
echo "SSH password for PC2: w"
echo "==========================================="

# Configuration
PC1_IP="192.168.1.10"
PC2_IP="192.168.1.11"
RAY_PORT="6379"
DASHBOARD_PORT="8265"

# Function to run SSH commands with password
run_ssh() {
    local command="$1"
    local description="$2"
    
    echo ""
    echo "🔧 $description"
    echo "Command (SSH to $PC2_IP): $command"
    
    # Use sshpass to provide password automatically
    sshpass -p 'w' ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$PC2_IP" "$command"
    return $?
}

echo ""
echo "============================================================"
echo "STEP 1: Activate Training_env on Head PC1"
echo "============================================================"

# First, source conda
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || eval "$(conda shell.bash hook)" 2>/dev/null

# Activate Training_env
conda activate Training_env

if [ $? -eq 0 ]; then
    echo "✅ Step 1 completed: Training_env activated on PC1"
    echo "Current environment: $CONDA_DEFAULT_ENV"
    step1_success=true
else
    echo "❌ Step 1 failed: Could not activate Training_env on PC1"
    echo "Available environments:"
    conda env list
    step1_success=false
fi

echo ""
echo "============================================================"
echo "STEP 2: SSH connect to PC2 IP - 192.168.1.11"
echo "============================================================"

echo "🔧 Testing SSH connection to $PC2_IP with password 'w'..."

# Test SSH connection
sshpass -p 'w' ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$PC2_IP" 'echo "SSH connection successful to PC2"'

if [ $? -eq 0 ]; then
    echo "✅ Step 2 completed: SSH connection to PC2 established"
    step2_success=true
else
    echo "❌ Step 2 failed: Cannot establish SSH connection to PC2"
    echo "💡 Please ensure sshpass is installed: sudo apt-get install sshpass"
    step2_success=false
fi

echo ""
echo "============================================================"
echo "STEP 3: Activate Training_env on Worker PC2 via SSH"
echo "============================================================"

if [ "$step2_success" = true ]; then
    run_ssh "source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null; conda activate Training_env && echo 'Training_env activated on PC2'" "Activating Training_env on Worker PC2"
    
    if [ $? -eq 0 ]; then
        echo "✅ Step 3 completed: Training_env activated on PC2"
        step3_success=true
    else
        echo "❌ Step 3 failed: Could not activate Training_env on PC2"
        step3_success=false
    fi
else
    echo "⏭️ Step 3 skipped: SSH connection not available"
    step3_success=false
fi

echo ""
echo "============================================================"
echo "STEP 4: Check Python version on Head PC1 (should be 3.12.2)"
echo "============================================================"

current_python=$(python --version 2>&1)
echo "🔧 Current Python version on PC1: $current_python"

if [[ "$current_python" == *"3.12.2"* ]]; then
    echo "✅ Step 4 completed: Python 3.12.2 confirmed on PC1"
    step4_success=true
else
    echo "❌ Step 4 failed: Python version mismatch on PC1"
    echo "Expected: 3.12.2, Found: $current_python"
    echo "💡 ERROR: Please ensure Python 3.12.2 is active in Training_env"
    step4_success=false
fi

echo ""
echo "============================================================"
echo "STEP 5: Check Python version on Worker PC2 (should be 3.12.2)"
echo "============================================================"

if [ "$step2_success" = true ]; then
    pc2_python=$(run_ssh "python --version" "Checking Python version on Worker PC2" 2>&1)
    echo "🔧 Python version on PC2: $pc2_python"
    
    if [[ "$pc2_python" == *"3.12.2"* ]]; then
        echo "✅ Step 5 completed: Python 3.12.2 confirmed on PC2"
        step5_success=true
    else
        echo "❌ Step 5 failed: Python version mismatch on PC2"
        echo "Expected: 3.12.2, Found: $pc2_python"
        echo "💡 ERROR: Please ensure Python 3.12.2 is active in Training_env on PC2"
        step5_success=false
    fi
else
    echo "⏭️ Step 5 skipped: SSH connection not available"
    step5_success=false
fi

echo ""
echo "============================================================"
echo "STEP 6: Stop Ray on Head PC1"
echo "============================================================"

echo "🔧 Stopping Ray on Head PC1..."
ray stop --force
echo "✅ Step 6 completed: Ray stopped on PC1"

echo ""
echo "============================================================"
echo "STEP 7: Start Ray head on Head PC1"
echo "============================================================"

echo "🔧 Starting Ray head on PC1 with specified configuration..."
ray start --head \
  --node-ip-address="$PC1_IP" \
  --port="$RAY_PORT" \
  --dashboard-host=0.0.0.0 \
  --dashboard-port="$DASHBOARD_PORT" \
  --object-manager-port=10001 \
  --ray-client-server-port=10201 \
  --min-worker-port=10300 \
  --max-worker-port=10399

if [ $? -eq 0 ]; then
    echo "✅ Step 7 completed: Ray head started on PC1"
    echo "🌐 Ray Dashboard: http://$PC1_IP:$DASHBOARD_PORT"
    sleep 5  # Give Ray head time to initialize
    step7_success=true
else
    echo "❌ Step 7 failed: Could not start Ray head on PC1"
    step7_success=false
fi

echo ""
echo "============================================================"
echo "STEP 8: Stop Ray on Worker PC2 via SSH"
echo "============================================================"

if [ "$step2_success" = true ]; then
    run_ssh "ray stop --force" "Stopping Ray on Worker PC2"
    echo "✅ Step 8 completed: Ray stopped on PC2"
else
    echo "⏭️ Step 8 skipped: SSH connection not available"
fi

echo ""
echo "============================================================"
echo "STEP 9: Start Ray worker on Worker PC2 via SSH"
echo "============================================================"

if [ "$step2_success" = true ] && [ "$step7_success" = true ]; then
    echo "🔧 Starting Ray worker on PC2..."
    run_ssh "ray start --address='$PC1_IP:$RAY_PORT' --node-ip-address=$PC2_IP" "Starting Ray worker on Worker PC2"
    
    if [ $? -eq 0 ]; then
        echo "✅ Step 9 completed: Ray worker started on PC2"
        sleep 3  # Give Ray worker time to connect
        step9_success=true
    else
        echo "❌ Step 9 failed: Could not start Ray worker on PC2"
        step9_success=false
    fi
else
    echo "⏭️ Step 9 skipped: Prerequisites not met (SSH or Ray head failed)"
    step9_success=false
fi

echo ""
echo "============================================================"
echo "FINAL VERIFICATION"
echo "============================================================"

echo "🔧 Checking Ray cluster status..."
ray status

cluster_status=$(ray status 2>&1)
node_count=$(echo "$cluster_status" | grep -c "node_")

echo ""
echo "============================================================"
echo "SETUP SUMMARY"
echo "============================================================"

echo "Setup Results:"
echo "- Step 1 (Conda PC1): $( [ "$step1_success" = true ] && echo "✅ Success" || echo "❌ Failed" )"
echo "- Step 2 (SSH PC2): $( [ "$step2_success" = true ] && echo "✅ Success" || echo "❌ Failed" )"
echo "- Step 3 (Conda PC2): $( [ "$step3_success" = true ] && echo "✅ Success" || echo "❌ Failed" )"
echo "- Step 4 (Python PC1): $( [ "$step4_success" = true ] && echo "✅ Success" || echo "❌ Version mismatch" )"
echo "- Step 5 (Python PC2): $( [ "$step5_success" = true ] && echo "✅ Success" || echo "❌ Version mismatch" )"
echo "- Step 6 (Ray stop PC1): ✅ Success"
echo "- Step 7 (Ray head PC1): $( [ "$step7_success" = true ] && echo "✅ Success" || echo "❌ Failed" )"
echo "- Step 8 (Ray stop PC2): $( [ "$step2_success" = true ] && echo "✅ Success" || echo "⏭️ Skipped" )"
echo "- Step 9 (Ray worker PC2): $( [ "$step9_success" = true ] && echo "✅ Success" || echo "❌ Failed" )"

if [ "$step9_success" = true ] && [ $node_count -ge 2 ]; then
    echo ""
    echo "🎉 SUCCESS: Dual PC Ray cluster is ready!"
    echo "📊 Cluster nodes: $node_count"
    echo "🌐 Dashboard: http://$PC1_IP:$DASHBOARD_PORT"
    echo "💻 Head Node: $PC1_IP:$RAY_PORT"
    echo "👷 Worker Node: $PC2_IP"
    echo ""
    echo "✅ Ready for 75% distributed training across both PCs!"
    
    # Launch the training system
    echo ""
    echo "🚀 Launching training system..."
    python fixed_integrated_training_75_percent.py &
    echo "Training system started in background"
    
    exit 0
elif [ "$step7_success" = true ]; then
    echo ""
    echo "⚠️ PARTIAL SUCCESS: Single PC Ray cluster ready"
    echo "🌐 Dashboard: http://$PC1_IP:$DASHBOARD_PORT"
    echo "💡 PC2 connection issues - training will run on PC1 only"
    
    # Check for specific issues
    if [ "$step4_success" != true ] || [ "$step5_success" != true ]; then
        echo ""
        echo "🔴 CRITICAL: Python version mismatch detected!"
        echo "Please ensure both PCs have Python 3.12.2 in Training_env"
    fi
    
    echo ""
    echo "🚀 Launching single PC training system..."
    python fixed_integrated_training_75_percent.py &
    echo "Training system started in background (single PC mode)"
    
    exit 0
else
    echo ""
    echo "❌ SETUP FAILED: Critical errors in Ray cluster setup"
    echo ""
    echo "💡 Please resolve these issues:"
    
    if [ "$step1_success" != true ]; then
        echo "  - Training_env conda environment not found on PC1"
    fi
    
    if [ "$step2_success" != true ]; then
        echo "  - SSH connection to PC2 failed (check network/sshpass)"
    fi
    
    if [ "$step4_success" != true ]; then
        echo "  - Python 3.12.2 not found on PC1"
    fi
    
    if [ "$step5_success" != true ]; then
        echo "  - Python 3.12.2 not found on PC2"
    fi
    
    if [ "$step7_success" != true ]; then
        echo "  - Ray head failed to start on PC1"
    fi
    
    exit 1
fi
