#!/bin/bash
"""
Fix Ray Cluster Python Version Mismatch
This script will restart the Ray cluster with the correct Python version
"""

echo "="*60
echo "FIXING RAY CLUSTER PYTHON VERSION MISMATCH"
echo "="*60

# Stop existing Ray cluster
echo "Shutting down existing Ray cluster..."
ray stop

# Wait a moment
sleep 2

# Start Ray head with current Python version
echo "Starting Ray head with Python $(python --version)..."
ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265 --disable-usage-stats

echo ""
echo "Ray cluster restarted successfully!"
echo "Head node dashboard: http://$(hostname -I | awk '{print $1}'):8265"
echo ""
echo "To connect worker nodes, run this command on each worker machine:"
echo "ray start --address='$(hostname -I | awk '{print $1}'):6379'"
echo ""
echo "Test the cluster with: python quick_ray_test.py"
