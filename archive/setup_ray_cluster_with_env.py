#!/usr/bin/env python3
"""
Ray Cluster Setup with Proper Environment Configuration
This script ensures Ray workers have access to all required modules
"""

import os
import sys
import subprocess
import socket
import time
import json
from pathlib import Path

def get_local_ip():
    """Get the local IP address"""
    try:
        # Connect to a remote address to determine local IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"

def check_conda_env():
    """Check if we're in the correct conda environment"""
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', '')
    if conda_env != 'BotsTraining_env':
        print(f"❌ WARNING: Not in BotsTraining_env (current: {conda_env})")
        print("Please activate: conda activate BotsTraining_env")
        return False
    else:
        print(f"✅ Conda environment: {conda_env}")
        return True

def setup_environment_variables():
    """Setup environment variables for Ray workers"""
    project_path = "/home/w1/cursor-to-copilot-backup/TaskmasterForexBots"
    worker_path = "/home/w2/cursor-to-copilot-backup/TaskmasterForexBots"
    
    # Set PYTHONPATH
    python_paths = [project_path, worker_path]
    existing_path = os.environ.get('PYTHONPATH', '')
    if existing_path:
        python_paths.append(existing_path)
    
    pythonpath = ":".join(python_paths)
    os.environ['PYTHONPATH'] = pythonpath
    
    print(f"✅ PYTHONPATH: {pythonpath}")
    return pythonpath

def create_ray_worker_script():
    """Create a script for worker nodes to ensure proper environment"""
    worker_script = """#!/bin/bash
# Ray Worker Environment Setup Script
cd /home/w2/cursor-to-copilot-backup/TaskmasterForexBots

# Activate conda environment
source /home/w2/miniconda3/etc/profile.d/conda.sh
conda activate BotsTraining_env

# Set environment variables
export PYTHONPATH="/home/w1/cursor-to-copilot-backup/TaskmasterForexBots:/home/w2/cursor-to-copilot-backup/TaskmasterForexBots:$PYTHONPATH"
export RAY_CLUSTER=1

# Test imports
echo "Testing Python imports..."
python3 -c "
import sys
print('Python executable:', sys.executable)
print('Python path:', sys.path[:3])

try:
    import synthetic_env
    print('✅ synthetic_env imported successfully')
except ImportError as e:
    print('❌ synthetic_env import failed:', e)

try:
    import ray
    print('✅ ray imported successfully')
    print('Ray version:', ray.__version__)
except ImportError as e:
    print('❌ ray import failed:', e)

try:
    import torch
    print('✅ torch imported successfully')
    print('Torch version:', torch.__version__)
    print('CUDA available:', torch.cuda.is_available())
except ImportError as e:
    print('❌ torch import failed:', e)
"

echo "Environment setup complete. Ready to start Ray worker."
"""
    
    script_path = "setup_worker_env.sh"
    with open(script_path, 'w') as f:
        f.write(worker_script)
    os.chmod(script_path, 0o755)
    print(f"✅ Created worker setup script: {script_path}")
    return script_path

def start_ray_head():
    """Start Ray head node with proper configuration"""
    local_ip = get_local_ip()
    
    # Kill any existing Ray processes
    print("🔄 Stopping any existing Ray processes...")
    subprocess.run(["ray", "stop"], capture_output=True)
    time.sleep(2)
    
    # Setup environment
    pythonpath = setup_environment_variables()
    
    # Ray head configuration
    ray_cmd = [
        "ray", "start", "--head",
        f"--node-ip-address={local_ip}",
        "--port=6379",
        "--dashboard-host=0.0.0.0",
        "--dashboard-port=8265",
        "--object-manager-port=8076",
        "--temp-dir=/tmp/ray"
    ]
    
    print(f"🚀 Starting Ray head node on {local_ip}:6379...")
    print(f"Command: {' '.join(ray_cmd)}")
    
    try:
        result = subprocess.run(ray_cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("✅ Ray head node started successfully!")
            print(f"Dashboard: http://{local_ip}:8265")
            print(f"Connection command for workers:")
            print(f"ray start --address='{local_ip}:6379'")
            return True
        else:
            print(f"❌ Failed to start Ray head: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("❌ Ray head startup timed out")
        return False

def check_ray_cluster():
    """Check Ray cluster status"""
    try:
        import ray
        if not ray.is_initialized():
            ray.init(address='auto')
        
        resources = ray.cluster_resources()
        nodes = ray.nodes()
        
        print("\n📊 Ray Cluster Status:")
        print(f"Total CPUs: {resources.get('CPU', 0)}")
        print(f"Total GPUs: {resources.get('GPU', 0)}")
        print(f"Total Memory: {resources.get('memory', 0) / 1e9:.1f} GB")
        print(f"Active nodes: {len([n for n in nodes if n['Alive']])}")
        
        for i, node in enumerate(nodes):
            if node['Alive']:
                node_ip = node['NodeManagerAddress']
                node_resources = node['Resources']
                print(f"  Node {i+1} ({node_ip}): "
                      f"CPU={node_resources.get('CPU', 0)}, "
                      f"GPU={node_resources.get('GPU', 0)}, "
                      f"Memory={node_resources.get('memory', 0)/1e9:.1f}GB")
        
        ray.shutdown()
        return True
        
    except Exception as e:
        print(f"❌ Error checking cluster: {e}")
        return False

def create_worker_connection_commands():
    """Create connection commands for worker PC"""
    local_ip = get_local_ip()
    
    commands = f"""
# Commands to run on Worker PC (192.168.1.11):

# 1. First, copy the project files if not already done:
scp -r w1@192.168.1.10:/home/w1/cursor-to-copilot-backup/TaskmasterForexBots /home/w2/cursor-to-copilot-backup/

# 2. Activate conda environment:
conda activate BotsTraining_env

# 3. Set environment variables:
export PYTHONPATH="/home/w1/cursor-to-copilot-backup/TaskmasterForexBots:/home/w2/cursor-to-copilot-backup/TaskmasterForexBots:$PYTHONPATH"
export RAY_CLUSTER=1

# 4. Connect to Ray cluster:
ray start --address='{local_ip}:6379' --temp-dir=/tmp/ray

# 5. Check connection:
ray status
"""
    
    with open("worker_connection_commands.txt", 'w') as f:
        f.write(commands)
    
    print("✅ Worker connection commands saved to: worker_connection_commands.txt")
    print(commands)

def main():
    print("🔧 Ray Cluster Setup with Environment Configuration")
    print("=" * 60)
    
    # Check environment
    if not check_conda_env():
        return False
    
    # Create worker setup script
    create_ray_worker_script()
    
    # Start Ray head
    if start_ray_head():
        time.sleep(3)  # Allow head to fully initialize
        
        # Create worker connection commands
        create_worker_connection_commands()
        
        # Check cluster status
        check_ray_cluster()
        
        print("\n🎯 Next Steps:")
        print("1. Copy worker_connection_commands.txt to Worker PC")
        print("2. Run the commands on Worker PC to connect")
        print("3. Run: RAY_CLUSTER=1 python run_stable_85_percent_trainer.py")
        
        return True
    else:
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 