#!/usr/bin/env python3
"""
Test script to verify all imports are working correctly
"""

print("Testing imports...")

try:
    import sys
    print("✓ sys imported")
    
    import os
    print("✓ os imported")
    
    import ray
    print("✓ ray imported")
    
    import torch
    print("✓ torch imported")
    
    import gc
    print("✓ gc imported")
    
    import GPUtil
    print("✓ GPUtil imported")
    
    from PyQt6.QtWidgets import QApplication
    print("✓ PyQt6.QtWidgets imported")
    
    import numpy as np
    print("✓ numpy imported")
    
    import pandas as pd
    print("✓ pandas imported")
    
    print("\n✅ All imports successful!")
    
    # Test Ray connection
    if ray.is_initialized():
        print("✓ Ray is already initialized")
        print(f"Ray cluster resources: {ray.available_resources()}")
    else:
        print("Ray is not initialized")
        
except Exception as e:
    print(f"❌ Import error: {e}")
    import traceback
    traceback.print_exc()

print("Import test completed.")
