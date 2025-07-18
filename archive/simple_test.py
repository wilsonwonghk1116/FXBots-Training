#!/usr/bin/env python3
"""
Simple Test for Bug Fix
"""
import sys
import os
import torch
import numpy as np

# Add current directory to path
sys.path.insert(0, os.getcwd())

# Test the import first
try:
    print("Testing imports...")
    
    # Try importing the main script
    import importlib.util
    spec = importlib.util.spec_from_file_location("main", "run_smart_real_training.py")
    main_module = importlib.util.module_from_spec(spec)
    print("Spec created successfully")
    
    spec.loader.exec_module(main_module)
    print("Module loaded successfully")
    
    # Test creating the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model = main_module.SmartTradingBot().to(device)
    print("Model created successfully")
    
    # Test a single forward pass
    obs = torch.randn(1, 26).to(device)
    print(f"Input shape: {obs.shape}")
    
    with torch.no_grad():
        action_probs, position_size = model(obs)
        print(f"Action probs: {action_probs}")
        print(f"Position size: {position_size}")
        print(f"Action probs shape: {action_probs.shape}")
        print(f"Action probs sum: {torch.sum(action_probs)}")
        
        # Check if probabilities are valid
        if torch.any(torch.isnan(action_probs)) or torch.any(torch.isinf(action_probs)):
            print("❌ Invalid probabilities detected!")
        elif torch.allclose(action_probs, torch.tensor([[1.0, 1.0, 1.0]], device=device)):
            print("❌ Still getting [1, 1, 1] probabilities!")
        else:
            print("✅ Probabilities look valid!")
            
        action = torch.argmax(action_probs).item()
        print(f"Selected action: {action}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
