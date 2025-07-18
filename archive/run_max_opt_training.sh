#!/bin/bash
#
# Script to run the fully optimized dual PC training.
# This script executes the standalone Python script which connects to
# an existing Ray cluster.
#
# Ensure the Ray cluster is running on both PCs before executing this.
# Head PC: ray start --head --node-ip-address=<HEAD_PC_IP>
# Worker PC: ray start --address=<HEAD_PC_IP>:6379
#

echo "🚀 Starting Full Training with Maximum GPU Optimizations 🚀"
echo "=========================================================="
echo "Executing script: run_dual_pc_training_standalone.py"
echo "Make sure your Ray cluster is active on both PCs."
echo "=========================================================="
echo ""

# Exit immediately if a command exits with a non-zero status.
set -e

# Run the fully optimized Python training script
python3 run_dual_pc_training_standalone.py

echo ""
echo "✅ Training script finished." 