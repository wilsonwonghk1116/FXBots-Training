#!/bin/bash
# Manual Cluster Test Commands
# ============================
# Run these commands one by one in your terminal

echo "=== STEP 1: Basic Connectivity ==="
echo "Test PC2 network connectivity:"
echo "ping -c 2 192.168.1.11"
echo ""

echo "=== STEP 2: SSH Test ==="
echo "Test SSH to PC2 with password 'w':"
echo "sshpass -p 'w' ssh w1@192.168.1.11 'echo SSH_SUCCESS'"
echo ""

echo "=== STEP 3: Conda Environment Check ==="
echo "Check Training_env on PC1:"
echo "conda env list | grep Training_env"
echo ""
echo "Check Training_env on PC2:"
echo "sshpass -p 'w' ssh w1@192.168.1.11 'conda env list | grep Training_env'"
echo ""

echo "=== STEP 4: Python Version Check ==="
echo "PC1 Python version:"
echo "source ~/miniconda3/etc/profile.d/conda.sh && conda activate Training_env && python --version"
echo ""
echo "PC2 Python version:"
echo "sshpass -p 'w' ssh w1@192.168.1.11 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate Training_env && python --version'"
echo ""

echo "=== STEP 5: Ray Status ==="
echo "Current Ray cluster status:"
echo "source ~/miniconda3/etc/profile.d/conda.sh && conda activate Training_env && ray status"
echo ""

echo "=== STEP 6: Launch Training ==="
echo "If all above tests pass, run the automated training:"
echo "python automated_cluster_training.py"
echo ""
