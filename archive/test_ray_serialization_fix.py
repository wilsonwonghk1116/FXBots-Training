#!/usr/bin/env python3
"""
Test script to verify Ray serialization fix for massive scale training
"""

import ray
import sys
import time

def test_ray_serialization():
    """Test that Ray can properly serialize our actors"""
    
    print("🧪 TESTING RAY SERIALIZATION FIX")
    print("=" * 50)
    
    # Initialize Ray
    try:
        if not ray.is_initialized():
            ray.init(address='ray://192.168.1.10:10001')
        print("✅ Ray connection established")
    except Exception as e:
        print(f"❌ Ray connection failed: {e}")
        return False
    
    # Import the fixed classes
    try:
        from massive_scale_distributed_training import MassiveScaleCoordinator, DistributedForexTrainer
        print("✅ Successfully imported fixed classes")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Test coordinator creation (this was failing before)
    try:
        print("🔄 Testing MassiveScaleCoordinator creation...")
        coordinator = MassiveScaleCoordinator.remote()
        print("✅ MassiveScaleCoordinator created successfully!")
        
        # Test that we can call a method on it
        ray.get(coordinator.__ray_ready__.remote())
        print("✅ Coordinator is ready and responsive")
        
    except Exception as e:
        print(f"❌ Coordinator creation failed: {e}")
        return False
    
    # Test trainer creation
    try:
        print("🔄 Testing DistributedForexTrainer creation...")
        trainer = DistributedForexTrainer.remote(worker_id=0)
        print("✅ DistributedForexTrainer created successfully!")
        
        # Test that trainer is ready
        ray.get(trainer.__ray_ready__.remote())
        print("✅ Trainer is ready and responsive")
        
    except Exception as e:
        print(f"❌ Trainer creation failed: {e}")
        return False
    
    print("\n🎉 ALL SERIALIZATION TESTS PASSED!")
    print("✅ The Ray serialization fix is working correctly")
    print("✅ Massive scale training can now be launched without errors")
    
    return True

if __name__ == "__main__":
    success = test_ray_serialization()
    
    if success:
        print("\n🚀 READY FOR MASSIVE SCALE TRAINING!")
        print("Run 'python launch_massive_training.py' and confirm with 'y' to start")
        sys.exit(0)
    else:
        print("\n❌ Serialization test failed")
        sys.exit(1)
