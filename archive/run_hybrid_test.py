#!/usr/bin/env python3
"""
Quick test of the hybrid system
"""

import torch
import multiprocessing
import psutil
import GPUtil

def test_hybrid_system():
    print("🚀 TESTING HYBRID SYSTEM 🚀")
    
    # Test CUDA
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name()}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Test CPU
    cpu_count = multiprocessing.cpu_count()
    print(f"CPU Cores: {cpu_count}")
    print(f"Optimal threads: {min(60, cpu_count * 4)}")
    
    # Test GPU status
    if torch.cuda.is_available():
        gpu = GPUtil.getGPUs()[0]
        print(f"GPU Memory: {gpu.memoryTotal}MB total, {gpu.memoryUsed}MB used")
        print(f"GPU Utilization: {gpu.load * 100:.1f}%")
    
    # Test tensor creation
    print("\nTesting tensor operations...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        test_tensor = torch.randn(1000, 1000).to(device)
        result = torch.matmul(test_tensor, test_tensor)
        print("✅ Tensor operations working")
        
        # Clean up
        del test_tensor, result
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"❌ Tensor operation failed: {e}")
    
    print("✅ Hybrid system test complete!")

if __name__ == "__main__":
    test_hybrid_system() 