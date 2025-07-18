#!/usr/bin/env python3
"""
Quick Test for Fixed RTX 3070 Optimized Trainer
==============================================

Tests the fixed trainer configuration that addresses:
1. GPU resource overallocation issues
2. Ray scheduling conflicts
3. Memory management improvements

Usage: python test_fixed_rtx3070.py
"""

import logging
import time
import ray
import torch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_cluster_resources():
    """Test Ray cluster connection and resource availability"""
    try:
        logger.info("🔍 Testing Ray cluster connection...")
        ray.init(address='auto', ignore_reinit_error=True)
        
        cluster_resources = ray.cluster_resources()
        available_cpus = cluster_resources.get('CPU', 0)
        available_gpus = cluster_resources.get('GPU', 0)
        
        logger.info(f"✅ Cluster Resources:")
        logger.info(f"   Available CPUs: {available_cpus}")
        logger.info(f"   Available GPUs: {available_gpus}")
        
        if available_gpus >= 2:
            logger.info("✅ Sufficient GPUs for dual-PC training (2+ required)")
            return True
        else:
            logger.warning(f"⚠️ Insufficient GPUs: {available_gpus} (2 required)")
            return False
            
    except Exception as e:
        logger.error(f"❌ Ray cluster test failed: {e}")
        return False

def test_gpu_allocation():
    """Test GPU allocation strategy"""
    try:
        logger.info("🔍 Testing GPU allocation strategy...")
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            logger.info(f"✅ CUDA GPUs detected: {gpu_count}")
            
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024**3)
                logger.info(f"   GPU {i}: {props.name} ({memory_gb:.1f}GB VRAM)")
            
            # Test allocation on each GPU
            with torch.cuda.device(0):
                test_tensor_0 = torch.randn(512, 512, device='cuda:0')
                logger.info("✅ GPU 0 allocation test passed")
                
            if gpu_count > 1:
                with torch.cuda.device(1):
                    test_tensor_1 = torch.randn(512, 512, device='cuda:1')
                    logger.info("✅ GPU 1 allocation test passed")
            
            return True
            
        else:
            logger.error("❌ No CUDA GPUs available")
            return False
            
    except Exception as e:
        logger.error(f"❌ GPU allocation test failed: {e}")
        return False

def test_fixed_configuration():
    """Test the fixed configuration parameters"""
    logger.info("🔍 Testing fixed configuration...")
    
    try:
        from rtx3070_optimized_trainer import RTX3070OptimizedConfig
        config = RTX3070OptimizedConfig()
        
        # Test reduced worker counts
        assert config.PC1_WORKERS == 1, f"Expected 1 PC1 worker, got {config.PC1_WORKERS}"
        assert config.PC2_WORKERS == 1, f"Expected 1 PC2 worker, got {config.PC2_WORKERS}"
        
        # Test increased per-worker resources
        assert config.PC1_CPUS_PER_WORKER == 40, f"Expected 40 CPUs per PC1 worker, got {config.PC1_CPUS_PER_WORKER}"
        assert config.PC2_CPUS_PER_WORKER == 20, f"Expected 20 CPUs per PC2 worker, got {config.PC2_CPUS_PER_WORKER}"
        
        # Test increased VRAM allocation
        assert config.PC1_VRAM_PER_WORKER_GB == 14.0, f"Expected 14GB per PC1 worker, got {config.PC1_VRAM_PER_WORKER_GB}"
        assert config.PC2_VRAM_PER_WORKER_GB == 4.0, f"Expected 4GB per PC2 worker, got {config.PC2_VRAM_PER_WORKER_GB}"
        
        logger.info("✅ Configuration parameters validated")
        
        # Calculate total resource requirements
        total_cpu_needed = (config.PC1_WORKERS * config.PC1_CPUS_PER_WORKER + 
                           config.PC2_WORKERS * config.PC2_CPUS_PER_WORKER)
        total_gpu_needed = config.PC1_WORKERS + config.PC2_WORKERS  # 1.0 GPU each
        
        logger.info(f"📊 Resource Requirements:")
        logger.info(f"   Total CPUs needed: {total_cpu_needed}")
        logger.info(f"   Total GPUs needed: {total_gpu_needed}")
        logger.info(f"   PC1 VRAM per worker: {config.PC1_VRAM_PER_WORKER_GB}GB")
        logger.info(f"   PC2 VRAM per worker: {config.PC2_VRAM_PER_WORKER_GB}GB")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("🚀 Starting RTX 3070 Fixed Trainer Tests")
    logger.info("=" * 50)
    
    tests = [
        ("Cluster Resources", test_cluster_resources),
        ("GPU Allocation", test_gpu_allocation),
        ("Fixed Configuration", test_fixed_configuration),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running: {test_name}")
        try:
            if test_func():
                logger.info(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                logger.error(f"❌ {test_name}: FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {e}")
    
    logger.info("\n" + "=" * 50)
    logger.info(f"🏁 Test Results: {passed}/{total} passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! The fixed trainer should work correctly.")
        logger.info("   Ready to run: python rtx3070_optimized_trainer.py --duration=5")
    else:
        logger.warning("⚠️ Some tests failed. Review the issues before running full training.")
    
    try:
        ray.shutdown()
    except:
        pass

if __name__ == "__main__":
    main() 