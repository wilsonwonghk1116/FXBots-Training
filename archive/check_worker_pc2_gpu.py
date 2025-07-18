#!/usr/bin/env python3
"""
Worker PC 2 GPU Diagnostic Script
=================================

This script checks GPU availability on Worker PC 2 to diagnose
why the Ray worker can't access the RTX 3070.
"""

import subprocess
import os
import sys

def run_command(cmd, description):
    """Run a command and return the result"""
    print(f"\n🔍 {description}")
    print(f"Command: {cmd}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        print(f"Return code: {result.returncode}")
        if result.stdout:
            print(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"STDERR:\n{result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return False

def main():
    print("🔍 WORKER PC 2 GPU DIAGNOSTIC TOOL")
    print("=" * 60)
    print(f"Python executable: {sys.executable}")
    print(f"Current user: {os.getenv('USER', 'unknown')}")
    print(f"Hostname: {os.uname().nodename}")
    
    # Check basic system info
    run_command("whoami", "Current user")
    run_command("hostname", "Hostname")
    run_command("uname -a", "System information")
    
    # Check NVIDIA driver
    run_command("nvidia-smi", "NVIDIA System Management Interface")
    run_command("nvidia-smi -L", "List NVIDIA GPUs")
    
    # Check CUDA installation
    run_command("nvcc --version", "CUDA compiler version")
    run_command("which nvcc", "CUDA compiler location")
    
    # Check environment variables
    print("\n🔍 Environment Variables")
    print("-" * 50)
    cuda_vars = ['CUDA_HOME', 'CUDA_PATH', 'CUDA_VISIBLE_DEVICES', 'PATH', 'LD_LIBRARY_PATH']
    for var in cuda_vars:
        value = os.environ.get(var, 'Not set')
        print(f"{var}: {value}")
    
    # Check PyTorch CUDA
    print("\n🔍 PyTorch CUDA Check")
    print("-" * 50)
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        
        if torch.cuda.is_available():
            print(f"Device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("❌ CUDA not available to PyTorch")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
    
    # Check Ray setup
    print("\n🔍 Ray Environment Check")
    print("-" * 50)
    try:
        import ray
        print(f"Ray version: {ray.__version__}")
        
        # Check if Ray can detect GPUs
        ray.init(ignore_reinit_error=True)
        resources = ray.cluster_resources()
        print(f"Ray cluster resources: {resources}")
        print(f"GPUs detected by Ray: {resources.get('GPU', 0)}")
        ray.shutdown()
        
    except ImportError as e:
        print(f"❌ Ray import failed: {e}")
    except Exception as e:
        print(f"❌ Ray error: {e}")
    
    # Check GPU device files
    print("\n🔍 GPU Device Files")
    print("-" * 50)
    gpu_devices = ['/dev/nvidia0', '/dev/nvidia1', '/dev/nvidiactl', '/dev/nvidia-uvm']
    for device in gpu_devices:
        if os.path.exists(device):
            stat = os.stat(device)
            print(f"✅ {device} exists (permissions: {oct(stat.st_mode)[-3:]})")
        else:
            print(f"❌ {device} does not exist")
    
    # Check permissions
    run_command("ls -la /dev/nvidia*", "NVIDIA device permissions")
    run_command("groups", "User groups")
    
    print("\n" + "=" * 60)
    print("🎯 DIAGNOSTIC COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
