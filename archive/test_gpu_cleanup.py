#!/usr/bin/env python3
"""
Test script to verify GPU VRAM cleanup functionality
"""

import torch
import gc
import time
import subprocess

def check_gpu_memory():
    """Check current GPU memory usage"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"GPU {i}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
    else:
        print("No CUDA GPUs available")

def test_cleanup_function():
    """Test the GPU cleanup functionality"""
    print("🧪 Testing GPU VRAM cleanup functionality...")
    
    # Check initial state
    print("\n📊 Initial GPU memory state:")
    check_gpu_memory()
    
    if torch.cuda.is_available():
        # Allocate some GPU memory
        print("\n🔄 Allocating test tensors...")
        tensors = []
        for i in range(5):
            tensor = torch.randn(1000, 1000, device='cuda')
            tensors.append(tensor)
        
        print("📊 After allocation:")
        check_gpu_memory()
        
        # Test cleanup
        print("\n🧹 Testing cleanup function...")
        
        # Delete tensors
        del tensors
        
        # Run comprehensive cleanup
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(i)
            torch.cuda.set_per_process_memory_fraction(1.0, device=i)
            
            try:
                torch.cuda.reset_accumulated_memory_stats(i)
            except:
                pass
        
        gc.collect()
        
        print("📊 After cleanup:")
        check_gpu_memory()
        
        print("✅ GPU cleanup test completed!")
    else:
        print("❌ No CUDA GPUs available for testing")

if __name__ == "__main__":
    test_cleanup_function()
