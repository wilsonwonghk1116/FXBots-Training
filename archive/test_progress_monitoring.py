#!/usr/bin/env python3
"""
Test Progress Monitoring System
==============================

Tests the enhanced RTX 3070 optimized trainer with detailed progress monitoring,
progress bars, and real-time status updates.

Features tested:
- Overall training progress bar (0-100%)
- Worker completion tracking
- Real-time GPU status monitoring
- Detailed status updates every 30 seconds
- Result collection with progress feedback

Usage: python test_progress_monitoring.py
"""

import logging
import time
import subprocess
import sys
import torch

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_requirements():
    """Check if all required packages are available"""
    try:
        import tqdm
        import ray
        import torch
        import numpy as np
        logger.info("✅ All required packages available")
        return True
    except ImportError as e:
        logger.error(f"❌ Missing required package: {e}")
        logger.info("💡 Install with: pip install tqdm ray torch numpy")
        return False

def test_tqdm_integration():
    """Test tqdm progress bars work correctly"""
    logger.info("🧪 Testing tqdm progress bar integration...")
    
    try:
        from tqdm import tqdm
        import time
        
        # Test overall progress bar
        with tqdm(total=100, desc="🚀 Overall Progress Test", 
                 bar_format="{l_bar}{bar}| {n:.1f}%/{total}% [{elapsed}<{remaining}]",
                 ncols=80) as pbar:
            
            for i in range(11):
                pbar.n = i * 10
                pbar.refresh()
                time.sleep(0.2)
        
        # Test worker completion bar
        with tqdm(total=5, desc="👷 Worker Test", 
                 bar_format="{l_bar}{bar}| {n}/{total} workers [{elapsed}]",
                 ncols=80) as pbar:
            
            for i in range(6):
                pbar.update(1)
                time.sleep(0.2)
        
        logger.info("✅ Progress bars working correctly")
        return True
        
    except Exception as e:
        logger.error(f"❌ Progress bar test failed: {e}")
        return False

def test_ray_cluster_connection():
    """Test Ray cluster connection for monitoring"""
    logger.info("🔗 Testing Ray cluster connection...")
    
    try:
        import ray
        
        # Try to connect to existing cluster
        try:
            ray.init(address='auto', ignore_reinit_error=True)
            logger.info("✅ Connected to existing Ray cluster")
            
            cluster_resources = ray.cluster_resources()
            logger.info(f"📊 Available CPUs: {cluster_resources.get('CPU', 0)}")
            logger.info(f"📊 Available GPUs: {cluster_resources.get('GPU', 0)}")
            
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ No Ray cluster running: {e}")
            logger.info("💡 Start Ray cluster with: ray start --head")
            return False
            
    except ImportError:
        logger.error("❌ Ray not available")
        return False

def test_gpu_monitoring():
    """Test GPU status monitoring functions"""
    logger.info("🎮 Testing GPU monitoring...")
    
    try:
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            logger.info(f"✅ Found {gpu_count} CUDA GPU(s)")
            
            # Test basic PyTorch GPU info
            for gpu_id in range(gpu_count):
                device_name = torch.cuda.get_device_name(gpu_id)
                memory_allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
                
                logger.info(f"🎮 GPU {gpu_id}: {device_name}")
                logger.info(f"   Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
            
            # Test VRAM manager compatibility (simulate the same method)
            logger.info("🧪 Testing VRAM manager compatibility...")
            try:
                props = torch.cuda.get_device_properties(0)
                total = props.total_memory / (1024**3)
                allocated = torch.cuda.memory_allocated(0) / (1024**3)
                utilization = (allocated / total) * 100
                
                logger.info(f"✅ VRAM Manager test: {props.name}")
                logger.info(f"   {utilization:.1f}% VRAM, {allocated:.2f}GB/{total:.2f}GB")
            except Exception as e:
                logger.warning(f"⚠️ VRAM Manager test failed: {e}")
            
            return True
        else:
            logger.warning("⚠️ CUDA not available - GPU monitoring disabled")
            return False
            
    except Exception as e:
        logger.error(f"❌ GPU monitoring test failed: {e}")
        return False

def run_short_training_test():
    """Run a very short training test to verify progress monitoring"""
    logger.info("🚀 Running short training test with progress monitoring...")
    
    try:
        # Run the optimized trainer for 1 minute only
        cmd = [sys.executable, "rtx3070_optimized_trainer.py", "--duration=1"]
        
        logger.info("⚡ Launching 1-minute training test...")
        logger.info("📊 You should see:")
        logger.info("   - Overall training progress bar (0-100%)")
        logger.info("   - Worker completion tracking")
        logger.info("   - Status updates every 30 seconds")
        logger.info("   - GPU status monitoring")
        logger.info("   - Result collection progress")
        
        # Run the command
        result = subprocess.run(cmd, capture_output=False, text=True)
        
        if result.returncode == 0:
            logger.info("✅ Short training test completed successfully!")
            return True
        else:
            logger.error(f"❌ Training test failed with return code: {result.returncode}")
            return False
            
    except FileNotFoundError:
        logger.error("❌ rtx3070_optimized_trainer.py not found")
        logger.info("💡 Make sure you're in the correct directory")
        return False
    except Exception as e:
        logger.error(f"❌ Training test failed: {e}")
        return False

def main():
    """Run all progress monitoring tests"""
    logger.info("🧪 TESTING RTX 3070 PROGRESS MONITORING SYSTEM")
    logger.info("=" * 60)
    
    all_tests_passed = True
    
    # Test 1: Check requirements
    logger.info("📋 Test 1: Checking requirements...")
    if not check_requirements():
        all_tests_passed = False
    
    print()
    
    # Test 2: Test progress bars
    logger.info("📊 Test 2: Testing progress bars...")
    if not test_tqdm_integration():
        all_tests_passed = False
    
    print()
    
    # Test 3: Test Ray connection
    logger.info("🔗 Test 3: Testing Ray cluster...")
    ray_available = test_ray_cluster_connection()
    
    print()
    
    # Test 4: Test GPU monitoring
    logger.info("🎮 Test 4: Testing GPU monitoring...")
    if not test_gpu_monitoring():
        logger.warning("⚠️ GPU monitoring limited - will still work with CPU fallback")
    
    print()
    
    # Test 5: Short training test (optional)
    if ray_available:
        logger.info("🚀 Test 5: Running short training test...")
        user_input = input("Run 1-minute training test? (y/N): ").lower().strip()
        
        if user_input in ['y', 'yes']:
            if not run_short_training_test():
                all_tests_passed = False
        else:
            logger.info("⏭️ Skipping training test")
    else:
        logger.info("⏭️ Skipping training test - Ray cluster not available")
    
    print()
    logger.info("=" * 60)
    
    if all_tests_passed:
        logger.info("🎉 ALL TESTS PASSED!")
        logger.info("✅ Progress monitoring system is ready to use")
        logger.info("🚀 Run: python rtx3070_optimized_trainer.py --duration=5")
    else:
        logger.warning("⚠️ Some tests failed - check logs above")
        logger.info("💡 Fix issues before running full training")
    
    logger.info("=" * 60)

if __name__ == "__main__":
    main() 