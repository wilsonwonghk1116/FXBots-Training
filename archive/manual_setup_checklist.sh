#!/bin/bash

# Step-by-Step Manual Ray Cluster Setup
# Run each step manually with SSH password 'w'

echo "🚀 Step-by-Step Dual PC Ray Cluster Setup"
echo "SSH password for PC2: w"
echo "==========================================="

echo ""
echo "STEP 1: Activate Training_env on Head PC1"
echo "Run: conda activate Training_env"
echo ""

echo "STEP 2: Test SSH to PC2 (password: w)"
echo "Run: ssh 192.168.1.11"
echo "Enter password: w"
echo "Test command: echo 'SSH works'"
echo "Exit: exit"
echo ""

echo "STEP 3: SSH to PC2 and activate Training_env"
echo "Run: ssh 192.168.1.11"
echo "Enter password: w"
echo "Run on PC2: conda activate Training_env"
echo "Keep this SSH session open for next steps"
echo ""

echo "STEP 4: Check Python version on PC1"
echo "Run: python --version"
echo "Should show: Python 3.12.2"
echo ""

echo "STEP 5: Check Python version on PC2 (in SSH session)"
echo "Run on PC2: python --version"
echo "Should show: Python 3.12.2"
echo ""

echo "STEP 6: Stop Ray on PC1"
echo "Run: ray stop"
echo ""

echo "STEP 7: Start Ray head on PC1"
echo "Run:"
echo "ray start --head \\"
echo "  --node-ip-address=192.168.1.10 \\"
echo "  --port=6379 \\"
echo "  --dashboard-host=0.0.0.0 \\"
echo "  --dashboard-port=8265 \\"
echo "  --object-manager-port=10001 \\"
echo "  --ray-client-server-port=10201 \\"
echo "  --min-worker-port=10300 \\"
echo "  --max-worker-port=10399"
echo ""

echo "STEP 8: Stop Ray on PC2 (in SSH session)"
echo "Run on PC2: ray stop"
echo ""

echo "STEP 9: Start Ray worker on PC2 (in SSH session)"
echo "Run on PC2:"
echo "ray start \\"
echo "  --address='192.168.1.10:6379' \\"
echo "  --node-ip-address=192.168.1.11"
echo ""

echo "VERIFICATION: Check cluster status"
echo "Run: ray status"
echo "Should show 2 nodes"
echo ""

echo "LAUNCH TRAINING:"
echo "Run: python fixed_integrated_training_75_percent.py"
echo ""

echo "📋 Manual execution checklist created!"
echo "Follow each step in order, using SSH password 'w' for PC2"
