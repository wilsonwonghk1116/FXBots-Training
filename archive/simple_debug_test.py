#!/usr/bin/env python3
"""
Simple debug test to check if Python execution works
"""

print("=== SIMPLE DEBUG TEST ===")
print("Python is working!")

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
except ImportError as e:
    print(f"PyTorch import error: {e}")

try:
    import numpy as np
    print(f"NumPy version: {np.__version__}")
except ImportError as e:
    print(f"NumPy import error: {e}")

try:
    import pandas as pd
    print(f"Pandas version: {pd.__version__}")
except ImportError as e:
    print(f"Pandas import error: {e}")

print("=== END DEBUG TEST ===")
