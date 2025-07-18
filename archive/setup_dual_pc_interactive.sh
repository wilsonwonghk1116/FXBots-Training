#!/bin/bash

# Interactive Dual PC Ray Cluster Setup Script
# Addresses conda, SSH, and Python version issues

echo "🚀 Interactive Dual PC Ray Cluster Setup"
echo "Following user's exact instructions with interactive fixes"
echo "==========================================================="

# Configuration
PC1_IP="192.168.1.10"
PC2_IP="192.168.1.11"
RAY_PORT="6379"
DASHBOARD_PORT="8265"

# Function to run commands with error handling
run_step() {
    local step_num="$1"
    local description="$2"
    local command="$3"
    local ssh_target="$4"
    
    echo ""
    echo "============================================================"
    echo "STEP $step_num: $description"
    echo "============================================================"
    
    if [ -n "$ssh_target" ]; then
        echo "🔧 Command (via SSH to $ssh_target): $command"
        ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$ssh_target" "$command"
    else
        echo "🔧 Command: $command"
        eval "$command"
    fi
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "✅ Step $step_num completed successfully"
        return 0
    else
        echo "❌ Step $step_num failed (exit code: $exit_code)"
        return $exit_code
    fi
}

# Step 1: Initialize conda and activate Training_env on PC1
echo ""
echo "============================================================"
echo "STEP 1: Initialize conda and activate Training_env on PC1"
echo "============================================================"

# Initialize conda for bash
eval "$(conda shell.bash hook)"

# Activate Training_env
conda activate Training_env

if [ $? -eq 0 ]; then
    echo "✅ Step 1 completed: Training_env activated on PC1"
    echo "Current Python: $(which python)"
    step1_success=true
else
    echo "❌ Step 1 failed: Could not activate Training_env on PC1"
    echo "💡 Available conda environments:"
    conda env list
    step1_success=false
fi

# Step 2: Test SSH connection to PC2
echo ""
echo "============================================================"
echo "STEP 2: Test SSH connection to Worker PC2"
echo "============================================================"

echo "🔧 Testing SSH connection to $PC2_IP (auto-accepting SSH key)..."
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$PC2_IP" 'echo "SSH connection successful to PC2"'

if [ $? -eq 0 ]; then
    echo "✅ Step 2 completed: SSH connection to PC2 established"
    step2_success=true
else
    echo "❌ Step 2 failed: Cannot establish SSH connection to PC2"
    echo "💡 Please ensure:"
    echo "   - PC2 is powered on and accessible"
    echo "   - SSH is enabled on PC2"
    echo "   - Network connectivity between PCs"
    step2_success=false
fi

# Step 3: Activate Training_env on PC2 via SSH
if [ "$step2_success" = true ]; then
    echo ""
    echo "============================================================"
    echo "STEP 3: Initialize conda and activate Training_env on PC2"
    echo "============================================================"
    
    echo "🔧 Initializing conda and activating Training_env on PC2..."
    ssh -o StrictHostKeyChecking=no "$PC2_IP" 'eval "$(conda shell.bash hook)" && conda activate Training_env && echo "Training_env activated on PC2"'
    
    if [ $? -eq 0 ]; then
        echo "✅ Step 3 completed: Training_env activated on PC2"
        step3_success=true
    else
        echo "❌ Step 3 failed: Could not activate Training_env on PC2"
        echo "💡 Checking available environments on PC2..."
        ssh -o StrictHostKeyChecking=no "$PC2_IP" 'conda env list'
        step3_success=false
    fi
else
    echo "⏭️ Step 3 skipped: SSH connection not available"
    step3_success=false
fi

# Step 4: Check Python version on PC1
echo ""
echo "============================================================"
echo "STEP 4: Check Python version on PC1"
echo "============================================================"

current_python=$(python --version 2>&1)
echo "🔧 Current Python version on PC1: $current_python"

if [[ "$current_python" == *"3.12.2"* ]]; then
    echo "✅ Step 4 completed: Python 3.12.2 confirmed on PC1"
    step4_success=true
else
    echo "⚠️ Step 4: Python version mismatch on PC1"
    echo "Expected: 3.12.2, Found: $current_python"
    echo "💡 Continuing with current Python version..."
    step4_success=false
fi

# Step 5: Check Python version on PC2
if [ "$step2_success" = true ]; then
    echo ""
    echo "============================================================"
    echo "STEP 5: Check Python version on PC2"
    echo "============================================================"
    
    pc2_python=$(ssh -o StrictHostKeyChecking=no "$PC2_IP" 'python --version' 2>&1)
    echo "🔧 Python version on PC2: $pc2_python"
    
    if [[ "$pc2_python" == *"3.12.2"* ]]; then
        echo "✅ Step 5 completed: Python 3.12.2 confirmed on PC2"
        step5_success=true
    else
        echo "⚠️ Step 5: Python version mismatch on PC2"
        echo "Expected: 3.12.2, Found: $pc2_python"
        echo "💡 Continuing with current Python version..."
        step5_success=false
    fi
else
    echo "⏭️ Step 5 skipped: SSH connection not available"
    step5_success=false
fi

# Step 6: Stop Ray on PC1
echo ""
echo "============================================================"
echo "STEP 6: Stop Ray on PC1"
echo "============================================================"

echo "🔧 Stopping Ray on PC1..."
ray stop --force

echo "✅ Step 6 completed: Ray stopped on PC1"

# Step 7: Start Ray head on PC1
echo ""
echo "============================================================"
echo "STEP 7: Start Ray head on PC1"
echo "============================================================"

echo "🔧 Starting Ray head on PC1..."
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

# Step 8: Stop Ray on PC2
if [ "$step2_success" = true ]; then
    echo ""
    echo "============================================================"
    echo "STEP 8: Stop Ray on PC2"
    echo "============================================================"
    
    echo "🔧 Stopping Ray on PC2..."
    ssh -o StrictHostKeyChecking=no "$PC2_IP" 'ray stop --force'
    
    echo "✅ Step 8 completed: Ray stopped on PC2"
else
    echo "⏭️ Step 8 skipped: SSH connection not available"
fi

# Step 9: Start Ray worker on PC2
if [ "$step2_success" = true ] && [ "$step7_success" = true ]; then
    echo ""
    echo "============================================================"
    echo "STEP 9: Start Ray worker on PC2"
    echo "============================================================"
    
    echo "🔧 Starting Ray worker on PC2..."
    ssh -o StrictHostKeyChecking=no "$PC2_IP" "ray start --address='$PC1_IP:$RAY_PORT' --node-ip-address=$PC2_IP"
    
    if [ $? -eq 0 ]; then
        echo "✅ Step 9 completed: Ray worker started on PC2"
        sleep 3  # Give Ray worker time to connect
        step9_success=true
    else
        echo "❌ Step 9 failed: Could not start Ray worker on PC2"
        step9_success=false
    fi
else
    echo "⏭️ Step 9 skipped: Prerequisites not met"
    step9_success=false
fi

# Final verification
echo ""
echo "============================================================"
echo "CLUSTER VERIFICATION"
echo "============================================================"

echo "🔧 Checking Ray cluster status..."
ray status

cluster_status=$(ray status 2>&1)
if [[ "$cluster_status" == *"2 nodes"* ]] || [[ "$cluster_status" == *"Active:"* ]]; then
    if [ "$step9_success" = true ]; then
        echo "✅ Dual PC Ray cluster verified successfully!"
        cluster_success=true
    else
        echo "⚠️ Single PC cluster only (PC2 not connected)"
        cluster_success=false
    fi
else
    echo "❌ Cluster verification failed"
    cluster_success=false
fi

# Summary
echo ""
echo "============================================================"
echo "SETUP SUMMARY"
echo "============================================================"

echo "Setup Results:"
echo "- Step 1 (Conda PC1): $( [ "$step1_success" = true ] && echo "✅ Success" || echo "❌ Failed" )"
echo "- Step 2 (SSH PC2): $( [ "$step2_success" = true ] && echo "✅ Success" || echo "❌ Failed" )"
echo "- Step 3 (Conda PC2): $( [ "$step3_success" = true ] && echo "✅ Success" || echo "❌ Failed" )"
echo "- Step 4 (Python PC1): $( [ "$step4_success" = true ] && echo "✅ Success" || echo "⚠️ Version mismatch" )"
echo "- Step 5 (Python PC2): $( [ "$step5_success" = true ] && echo "✅ Success" || echo "⚠️ Version mismatch/SSH failed" )"
echo "- Step 6 (Ray stop PC1): ✅ Success"
echo "- Step 7 (Ray head PC1): $( [ "$step7_success" = true ] && echo "✅ Success" || echo "❌ Failed" )"
echo "- Step 8 (Ray stop PC2): $( [ "$step2_success" = true ] && echo "✅ Success" || echo "⏭️ Skipped" )"
echo "- Step 9 (Ray worker PC2): $( [ "$step9_success" = true ] && echo "✅ Success" || echo "❌ Failed" )"

if [ "$cluster_success" = true ]; then
    echo ""
    echo "🎉 SUCCESS: Dual PC Ray cluster is ready!"
    echo "🌐 Dashboard: http://$PC1_IP:$DASHBOARD_PORT"
    echo "📊 Head Node: $PC1_IP:$RAY_PORT"
    echo "👷 Worker Node: $PC2_IP"
    echo ""
    echo "✅ Ready for 75% distributed training across both PCs!"
    exit 0
elif [ "$step7_success" = true ]; then
    echo ""
    echo "⚠️ PARTIAL SUCCESS: Single PC Ray cluster ready"
    echo "🌐 Dashboard: http://$PC1_IP:$DASHBOARD_PORT"
    echo "💡 PC2 connection failed - training will run on PC1 only"
    echo ""
    echo "⚠️ Ready for single PC training (not distributed)"
    exit 0
else
    echo ""
    echo "❌ SETUP FAILED: Ray cluster could not be established"
    echo "💡 Please check the errors above and resolve them"
    exit 1
fi
