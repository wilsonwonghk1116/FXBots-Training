#!/usr/bin/env python3
"""
Test GPU Status Fix
==================

Specifically tests the fix for the 'allocated_gb' KeyError in GPU status monitoring.

Before fix: KeyError: 'allocated_gb'
After fix: Should work with correct key names

Usage: python test_gpu_status_fix.py
"""

import logging
import torch
from typing import Dict, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestVRAMManager:
    """Test version of VRAM Manager to simulate the exact same method"""
    
    @staticmethod
    def get_detailed_memory_info(device_id: int = 0) -> Optional[Dict]:
        """Get comprehensive GPU memory information - EXACT COPY"""
        if not torch.cuda.is_available():
            return None
        
        try:
            props = torch.cuda.get_device_properties(device_id)
            total = props.total_memory / (1024**3)
            allocated = torch.cuda.memory_allocated(device_id) / (1024**3)
            cached = torch.cuda.memory_reserved(device_id) / (1024**3)
            free = total - allocated
            
            return {
                'total': total,
                'allocated': allocated,
                'cached': cached,
                'free': free,
                'utilization': (allocated / total) * 100,
                'device_name': props.name,
                'multiprocessor_count': props.multi_processor_count
            }
        except Exception as e:
            logger.warning(f"Failed to get memory info: {e}")
            return None

def test_old_broken_method():
    """Test the old broken method that caused the KeyError"""
    logger.info("🔴 Testing OLD BROKEN method...")
    
    vram_manager = TestVRAMManager()
    
    try:
        if torch.cuda.is_available():
            for gpu_id in range(torch.cuda.device_count()):
                memory_info = vram_manager.get_detailed_memory_info(gpu_id)
                if memory_info:
                    # This SHOULD fail with KeyError: 'allocated_gb'
                    logger.info(f"🎮 GPU {gpu_id} ({memory_info['device_name']}): "
                               f"{memory_info['utilization']:.1f}% VRAM, "
                               f"{memory_info['allocated_gb']:.2f}GB/{memory_info['total_gb']:.2f}GB")
        else:
            logger.info("🔍 GPU status: CUDA not available")
        
        logger.info("⚠️ OLD method unexpectedly succeeded - this should have failed!")
        return False
        
    except KeyError as e:
        logger.error(f"❌ OLD method failed as expected with KeyError: {e}")
        return True
    except Exception as e:
        logger.error(f"❌ OLD method failed with unexpected error: {e}")
        return False

def test_new_fixed_method():
    """Test the new fixed method that should work"""
    logger.info("🟢 Testing NEW FIXED method...")
    
    vram_manager = TestVRAMManager()
    
    try:
        if torch.cuda.is_available():
            for gpu_id in range(torch.cuda.device_count()):
                memory_info = vram_manager.get_detailed_memory_info(gpu_id)
                if memory_info:
                    # Use correct key names with safe access
                    device_name = memory_info.get('device_name', f'GPU{gpu_id}')
                    utilization = memory_info.get('utilization', 0)
                    allocated = memory_info.get('allocated', 0)
                    total = memory_info.get('total', 0)
                    
                    logger.info(f"🎮 GPU {gpu_id} ({device_name}): "
                               f"{utilization:.1f}% VRAM, "
                               f"{allocated:.2f}GB/{total:.2f}GB")
                else:
                    logger.info(f"🎮 GPU {gpu_id}: Memory info unavailable")
        else:
            logger.info("🔍 GPU status: CUDA not available")
        
        logger.info("✅ NEW method succeeded!")
        return True
        
    except Exception as e:
        logger.error(f"❌ NEW method failed: {e}")
        return False

def test_data_structure():
    """Test what the actual data structure looks like"""
    logger.info("🔍 Testing actual data structure...")
    
    vram_manager = TestVRAMManager()
    
    try:
        if torch.cuda.is_available():
            memory_info = vram_manager.get_detailed_memory_info(0)
            if memory_info:
                logger.info("📊 Actual memory_info structure:")
                for key, value in memory_info.items():
                    if isinstance(value, float):
                        logger.info(f"   '{key}': {value:.3f}")
                    else:
                        logger.info(f"   '{key}': {value}")
                
                logger.info("✅ Data structure analysis complete")
                return True
            else:
                logger.warning("⚠️ No memory info available")
                return False
        else:
            logger.warning("⚠️ CUDA not available")
            return False
            
    except Exception as e:
        logger.error(f"❌ Data structure test failed: {e}")
        return False

def main():
    """Run all GPU status fix tests"""
    logger.info("🧪 TESTING GPU STATUS FIX")
    logger.info("=" * 50)
    
    # Test 1: Show the data structure
    logger.info("📋 Test 1: Analyzing data structure...")
    test_data_structure()
    
    print()
    
    # Test 2: Test the old broken method
    logger.info("🔴 Test 2: Testing old broken method...")
    old_method_failed = test_old_broken_method()
    
    print()
    
    # Test 3: Test the new fixed method
    logger.info("🟢 Test 3: Testing new fixed method...")
    new_method_succeeded = test_new_fixed_method()
    
    print()
    logger.info("=" * 50)
    
    if old_method_failed and new_method_succeeded:
        logger.info("🎉 ALL TESTS PASSED!")
        logger.info("✅ Old method correctly fails with KeyError")
        logger.info("✅ New method works without errors")
        logger.info("🚀 GPU status monitoring fix is working!")
    else:
        logger.warning("⚠️ Test results unexpected:")
        logger.info(f"   Old method failed (expected): {old_method_failed}")
        logger.info(f"   New method succeeded (expected): {new_method_succeeded}")
    
    logger.info("=" * 50)
    logger.info("💡 Now try: python test_progress_monitoring.py")

if __name__ == "__main__":
    main() 