#!/bin/bash
"""
Complete Dual PC Training Launcher
Sets up Ray cluster then launches training with 75% utilization
"""

echo "🚀 Complete Dual PC Training System Launcher"
echo "============================================="

# Step 1: Setup the Ray cluster
echo "📡 Setting up Ray cluster across PC1 & PC2..."
python setup_dual_pc_ray_cluster.py

# Check if setup was successful
if [ $? -eq 0 ]; then
    echo "✅ Ray cluster setup completed successfully!"
    echo ""
    echo "🎯 Launching training system with 75% utilization..."
    sleep 2
    
    # Step 2: Launch the training system
    python fixed_integrated_training_75_percent.py
    
else
    echo "❌ Ray cluster setup failed!"
    echo "Please check the setup errors above and try again."
    exit 1
fi
