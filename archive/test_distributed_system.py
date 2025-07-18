#!/usr/bin/env python3
"""
Simplified launcher for the distributed training system
Tests the Ray cluster and basic functionality without GUI
"""

import os
import sys
import ray
import time
import torch
import gc
import GPUtil
import subprocess

def cleanup_gpu_vram():
    """Clean up GPU VRAM"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print("GPU VRAM cleaned up")

def check_ray_cluster():
    """Check Ray cluster status and resources"""
    try:
        if not ray.is_initialized():
            print("Ray not initialized. Connecting to cluster...")
            ray.init(address="192.168.1.10:6379")
        
        resources = ray.available_resources()
        print(f"Ray cluster resources: {resources}")
        
        # Check if we have both PCs
        nodes = ray.nodes()
        print(f"Ray cluster nodes: {len(nodes)}")
        for i, node in enumerate(nodes):
            print(f"Node {i}: {node['NodeManagerAddress']}")
        
        return True
    except Exception as e:
        print(f"Ray cluster error: {e}")
        return False

def test_distributed_task():
    """Test a simple distributed task"""
    try:
        @ray.remote
        def cpu_bound_task(n):
            """Simple CPU-bound task for testing"""
            import time
            start = time.time()
            # Simulate work
            result = sum(i * i for i in range(n))
            end = time.time()
            return result, end - start
        
        print("Testing distributed tasks...")
        
        # Run tasks on both PCs
        futures = []
        for i in range(4):  # 4 tasks to distribute across nodes
            future = cpu_bound_task.remote(1000000)
            futures.append(future)
        
        results = ray.get(futures)
        print(f"✓ Completed {len(results)} distributed tasks")
        
        for i, (result, duration) in enumerate(results):
            print(f"Task {i}: result={result}, duration={duration:.3f}s")
        
        return True
    except Exception as e:
        print(f"Distributed task error: {e}")
        return False

def monitor_resources():
    """Monitor system resources"""
    try:
        # GPU info
        if torch.cuda.is_available():
            print(f"CUDA devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # GPUtil info
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            print(f"GPU {gpu.id}: {gpu.name}, Memory: {gpu.memoryUsed}/{gpu.memoryTotal}MB ({gpu.memoryUtil*100:.1f}%)")
        
        return True
    except Exception as e:
        print(f"Resource monitoring error: {e}")
        return False

def main():
    print("=== Distributed Training System Test ===")
    print(f"Working directory: {os.getcwd()}")
    print(f"Python version: {sys.version}")
    
    # Cleanup GPU first
    cleanup_gpu_vram()
    
    # Check Ray cluster
    print("\n1. Checking Ray cluster...")
    if not check_ray_cluster():
        print("❌ Ray cluster check failed")
        return False
    
    # Monitor resources
    print("\n2. Monitoring resources...")
    monitor_resources()
    
    # Test distributed tasks
    print("\n3. Testing distributed tasks...")
    if not test_distributed_task():
        print("❌ Distributed task test failed")
        return False
    
    print("\n✅ All tests passed! The distributed training system is ready.")
    print("\nTo launch the full GUI version, run:")
    print("python fixed_integrated_training_75_percent.py")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n🎉 System is ready for full training!")
        else:
            print("\n⚠️ Some tests failed. Check the errors above.")
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup_gpu_vram()
