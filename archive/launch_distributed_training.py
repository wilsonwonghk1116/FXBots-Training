#!/usr/bin/env python3
"""
Distributed Training Launcher
Simple script to launch the distributed forex bot training
"""

import os
import sys
import subprocess
import time

def check_environment():
    """Check if environment is ready for training"""
    print("🔍 Checking Environment...")
    
    # Check conda environment
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', '')
    if conda_env != 'BotsTraining_env':
        print(f"❌ Wrong conda environment: {conda_env}")
        print("Please run: conda activate BotsTraining_env")
        return False
    else:
        print(f"✅ Conda environment: {conda_env}")
    
    # Check Ray availability
    try:
        import ray
        print(f"✅ Ray available: {ray.__version__}")
    except ImportError:
        print("❌ Ray not available")
        return False
    
    # Check project modules
    try:
        import synthetic_env
        print("✅ synthetic_env available")
    except ImportError as e:
        print(f"❌ synthetic_env not available: {e}")
        return False
    
    return True

def check_ray_cluster():
    """Check if Ray cluster is running and connected"""
    print("\n🔗 Checking Ray Cluster...")
    
    try:
        import ray
        
        # Try to connect to existing cluster
        ray.init(address='auto')
        
        resources = ray.cluster_resources()
        nodes = ray.nodes()
        active_nodes = [n for n in nodes if n['Alive']]
        
        print(f"✅ Connected to Ray cluster")
        print(f"   Total CPUs: {resources.get('CPU', 0)}")
        print(f"   Total GPUs: {resources.get('GPU', 0)}")
        print(f"   Active nodes: {len(active_nodes)}")
        
        for i, node in enumerate(active_nodes):
            node_ip = node['NodeManagerAddress']
            node_resources = node['Resources']
            print(f"   Node {i+1} ({node_ip}): "
                  f"CPU={node_resources.get('CPU', 0)}, "
                  f"GPU={node_resources.get('GPU', 0)}")
        
        ray.shutdown()
        
        if len(active_nodes) < 2:
            print(f"⚠️  Warning: Only {len(active_nodes)} node(s) active. Need 2 for distributed training.")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to connect to Ray cluster: {e}")
        print("   Make sure Ray head is running with: python setup_ray_cluster_with_env.py")
        return False

def start_training():
    """Start the distributed training"""
    print("\n🚀 Starting Distributed Training...")
    
    # Set environment variables
    os.environ['RAY_CLUSTER'] = '1'
    
    # Add project path to PYTHONPATH
    project_paths = [
        "/home/w1/cursor-to-copilot-backup/TaskmasterForexBots",
        "/home/w2/cursor-to-copilot-backup/TaskmasterForexBots"
    ]
    
    pythonpath = ":".join(project_paths)
    if 'PYTHONPATH' in os.environ:
        os.environ['PYTHONPATH'] = f"{pythonpath}:{os.environ['PYTHONPATH']}"
    else:
        os.environ['PYTHONPATH'] = pythonpath
    
    print(f"✅ Environment variables set")
    print(f"   RAY_CLUSTER=1")
    print(f"   PYTHONPATH={os.environ['PYTHONPATH']}")
    
    # Launch training
    cmd = [sys.executable, "run_stable_85_percent_trainer.py"]
    print(f"\n🏃 Executing: {' '.join(cmd)}")
    
    try:
        # Run the training script
        result = subprocess.run(cmd, cwd=os.getcwd())
        
        if result.returncode == 0:
            print("\n🎉 Training completed successfully!")
        else:
            print(f"\n❌ Training failed with exit code: {result.returncode}")
            
        return result.returncode == 0
        
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        return False
    except Exception as e:
        print(f"\n💥 Training crashed: {e}")
        return False

def main():
    print("🤖 Distributed Forex Bot Training Launcher")
    print("=" * 50)
    
    # Check environment
    if not check_environment():
        print("\n❌ Environment check failed. Please fix issues above.")
        return 1
    
    # Check Ray cluster
    if not check_ray_cluster():
        print("\n❌ Ray cluster check failed. Please ensure:")
        print("   1. Head PC has Ray head running: python setup_ray_cluster_with_env.py")
        print("   2. Worker PC has connected to Ray cluster")
        print("   3. Both nodes show as active")
        return 1
    
    # Start training
    print("\n✅ All checks passed. Starting training...")
    time.sleep(2)  # Brief pause before starting
    
    success = start_training()
    
    if success:
        print("\n🏆 Distributed training completed successfully!")
        print("   Check for CHAMPION_BOT_*.pth files in the current directory")
        return 0
    else:
        print("\n🔥 Training failed or was interrupted")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 