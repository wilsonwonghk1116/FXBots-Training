#!/usr/bin/env python3
"""
QUICK TEST FOR RTX 3070 OPTIMIZATION
===================================
This script quickly tests if the new VRAM optimizations work on your RTX 3070.
Run this BEFORE the full training to verify everything is working properly.
"""

import os
import torch
import time
import gc
from datetime import datetime

# Apply the same environment optimizations
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:64,roundup_power2_divisions:16'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

print("🎯 RTX 3070 VRAM OPTIMIZATION - QUICK TEST")
print("=" * 50)

def clear_cuda_memory():
    """Clear all CUDA memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

def get_memory_info():
    """Get GPU memory information"""
    if not torch.cuda.is_available():
        return None
    
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    allocated = torch.cuda.memory_allocated(0) / (1024**3)
    cached = torch.cuda.memory_reserved(0) / (1024**3)
    free = total - allocated
    utilization = (allocated / total) * 100
    
    return {
        'total': total,
        'allocated': allocated,
        'cached': cached,
        'free': free,
        'utilization': utilization,
        'device_name': torch.cuda.get_device_name(0)
    }

def test_conservative_allocation():
    """Test conservative VRAM allocation (1.5GB)"""
    print("\n🔧 Testing Conservative 1.5GB Allocation...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    try:
        device = torch.device("cuda:0")
        
        # Set conservative memory fraction (75%)
        torch.cuda.set_per_process_memory_fraction(0.75, 0)
        
        # Clear memory first
        clear_cuda_memory()
        
        # Test allocation of 1.5GB
        target_gb = 1.5
        bytes_to_allocate = int(target_gb * 1024**3)
        
        # Use float16 for efficiency
        tensor = torch.empty(bytes_to_allocate // 2, dtype=torch.float16, device=device)
        
        memory_info = get_memory_info()
        if memory_info:
            print(f"✅ Successfully allocated {target_gb}GB")
            print(f"   Device: {memory_info['device_name']}")
            print(f"   VRAM Utilization: {memory_info['utilization']:.1f}%")
            print(f"   Free Memory: {memory_info['free']:.2f}GB")
        else:
            print(f"✅ Successfully allocated {target_gb}GB")
            print("   Memory info unavailable")
        
        # Test operations
        print("\n🧪 Testing Operations...")
        with torch.cuda.amp.autocast():
            a = torch.randn(1024, 1024, device=device, dtype=torch.float16)
            b = torch.randn(1024, 1024, device=device, dtype=torch.float16)
            c = torch.matmul(a, b)
            result = torch.sum(c).item()
            
        print(f"✅ Matrix operations successful: {result:.2f}")
        
        # Cleanup
        del tensor, a, b, c
        clear_cuda_memory()
        
        final_memory = get_memory_info()
        if final_memory:
            print(f"✅ Memory cleanup successful: {final_memory['utilization']:.1f}% utilization")
        else:
            print(f"✅ Memory cleanup completed")
        
        return True
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"❌ OOM Error: {e}")
            print("💡 Try reducing allocation or using emergency mode")
        else:
            print(f"❌ CUDA Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        return False

def test_progressive_allocation():
    """Test progressive memory allocation"""
    print("\n🔧 Testing Progressive Allocation...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    device = torch.device("cuda:0")
    
    try:
        # Progressive allocation test
        steps = [0.3, 0.6, 0.9, 1.2, 1.5]  # GB
        
        for i, target_gb in enumerate(steps):
            clear_cuda_memory()
            
            bytes_to_allocate = int(target_gb * 1024**3)
            tensor = torch.empty(bytes_to_allocate // 2, dtype=torch.float16, device=device)
            
            memory_info = get_memory_info()
            if memory_info:
                print(f"  Step {i+1}: {target_gb}GB allocated, {memory_info['utilization']:.1f}% utilization")
            else:
                print(f"  Step {i+1}: {target_gb}GB allocated")
            
            del tensor
        
        print("✅ Progressive allocation test successful")
        return True
        
    except Exception as e:
        print(f"❌ Progressive allocation failed at {target_gb}GB: {e}")
        return False

def test_emergency_fallback():
    """Test emergency fallback operations"""
    print("\n🔧 Testing Emergency Fallback...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    device = torch.device("cuda:0")
    
    try:
        clear_cuda_memory()
        
        # Emergency minimal operations (128x128 matrices)
        with torch.cuda.amp.autocast():
            a = torch.randn(128, 128, device=device, dtype=torch.float16)
            b = torch.randn(128, 128, device=device, dtype=torch.float16)
            c = torch.matmul(a, b)
            result = torch.sum(c).item()
            
        memory_info = get_memory_info()
        print(f"✅ Emergency operations successful: {result:.2f}")
        print(f"   Minimal VRAM usage: {memory_info['utilization']:.1f}%")
        
        del a, b, c
        clear_cuda_memory()
        
        return True
        
    except Exception as e:
        print(f"❌ Emergency fallback failed: {e}")
        return False

def main():
    """Run all tests"""
    start_time = time.time()
    
    # Initial system check
    print(f"⏰ Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Please check your GPU setup.")
        return
    
    initial_memory = get_memory_info()
    print(f"🖥️ GPU: {initial_memory['device_name']}")
    print(f"💾 Total VRAM: {initial_memory['total']:.2f}GB")
    print(f"📊 Initial Utilization: {initial_memory['utilization']:.1f}%")
    
    # Run tests
    tests = [
        ("Conservative Allocation", test_conservative_allocation),
        ("Progressive Allocation", test_progressive_allocation),
        ("Emergency Fallback", test_emergency_fallback)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n" + "="*50)
        success = test_func()
        results[test_name] = success
        
        if success:
            print(f"✅ {test_name}: PASSED")
        else:
            print(f"❌ {test_name}: FAILED")
    
    # Summary
    print("\n" + "="*50)
    print("📋 TEST SUMMARY")
    print("="*50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Your RTX 3070 is ready for optimized training!")
        print("\nNext step: Run the full trainer:")
        print("python rtx3070_optimized_trainer.py --duration=5")
    else:
        print("⚠️ Some tests failed. Check the errors above.")
        print("\nTroubleshooting:")
        print("1. Make sure Ray cluster is running")
        print("2. Check available VRAM (close other GPU applications)")
        print("3. Try reducing memory fraction to 0.65")
    
    total_time = time.time() - start_time
    print(f"\n⏱️ Test completed in {total_time:.1f} seconds")

if __name__ == "__main__":
    main() 