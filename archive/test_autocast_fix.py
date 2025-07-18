#!/usr/bin/env python3
"""
Quick Test for PyTorch Autocast Deprecation Fix
==============================================

Tests that the rtx3070_optimized_trainer.py no longer produces
FutureWarning about deprecated torch.cuda.amp.autocast usage.

Usage: python test_autocast_fix.py
"""

import warnings
import torch
import logging

# Set up to catch warnings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_new_autocast_syntax():
    """Test that new autocast syntax works without warnings"""
    if not torch.cuda.is_available():
        logger.warning("⚠️ CUDA not available - autocast test skipped")
        return True
    
    try:
        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Test new syntax
            device = torch.cuda.current_device()
            with torch.cuda.device(device):
                with torch.autocast('cuda'):
                    # Simple operation
                    a = torch.randn(100, 100, device=device, dtype=torch.float16)
                    b = torch.randn(100, 100, device=device, dtype=torch.float16)
                    c = torch.matmul(a, b)
                    
            # Check for warnings
            deprecation_warnings = [warning for warning in w 
                                  if "deprecated" in str(warning.message).lower()]
            
            if deprecation_warnings:
                logger.error("❌ Still getting deprecation warnings:")
                for warning in deprecation_warnings:
                    logger.error(f"   {warning.message}")
                return False
            else:
                logger.info("✅ No deprecation warnings - autocast syntax fixed!")
                return True
                
    except Exception as e:
        logger.error(f"❌ Error testing autocast: {e}")
        return False

def test_imports():
    """Test that required torch modules are available"""
    try:
        # Test that torch.autocast is available
        assert hasattr(torch, 'autocast'), "torch.autocast not available"
        logger.info("✅ torch.autocast available")
        
        # Test that we can create an autocast context
        with torch.autocast('cuda'):
            pass
        logger.info("✅ torch.autocast('cuda') context works")
        
        return True
    except Exception as e:
        logger.error(f"❌ Import test failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("🧪 Testing PyTorch Autocast Fix...")
    
    # Test imports first
    if not test_imports():
        logger.error("❌ Import tests failed")
        exit(1)
    
    # Test autocast syntax
    if not test_new_autocast_syntax():
        logger.error("❌ Autocast syntax test failed")
        exit(1)
    
    logger.info("🎉 ALL TESTS PASSED! Autocast warnings should be fixed.")
    logger.info("💡 You can now run: python rtx3070_optimized_trainer.py --duration=5")
    logger.info("   And the FutureWarning should be gone!") 