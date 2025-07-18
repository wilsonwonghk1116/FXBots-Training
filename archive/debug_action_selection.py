#!/usr/bin/env python3
"""
Focused Action Selection Debug Tool
===================================

This tool specifically debugs the action selection mechanism to find why 
torch.argmax always returns 0 (HOLD) even when probabilities are balanced.
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
import logging

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.model import SmartTradingBot
from src.env import SmartForexEnvironment

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_action_selection():
    """Debug the exact action selection process"""
    print("="*80)
    print("FOCUSED ACTION SELECTION DEBUG")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmartTradingBot().to(device)
    model.eval()
    
    print(f"Device: {device}")
    print(f"Model: {model}")
    
    # Test multiple random inputs
    num_tests = 20
    action_counts = [0, 0, 0]  # hold, buy, sell
    
    print("\nTesting action selection with random inputs:")
    print("-" * 80)
    
    for i in range(num_tests):
        # Create random observation
        obs = np.random.randn(26)
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        
        print(f"\nTest {i+1}:")
        print(f"  Input shape: {obs_tensor.shape}")
        print(f"  Input sample: {obs_tensor[0][:5].cpu().numpy()}")  # First 5 values
        
        with torch.no_grad():
            action_probs, position_size = model(obs_tensor)
            
            print(f"  Raw action_probs shape: {action_probs.shape}")
            print(f"  Raw action_probs: {action_probs}")
            print(f"  Action_probs CPU: {action_probs.cpu().numpy()}")
            
            # Check each step of action selection
            print(f"  Probability sum: {torch.sum(action_probs).item():.6f}")
            print(f"  Max probability: {torch.max(action_probs).item():.6f}")
            print(f"  Min probability: {torch.min(action_probs).item():.6f}")
            
            # Manual argmax check
            action_np = np.argmax(action_probs.cpu().numpy())
            action_torch = torch.argmax(action_probs).item()
            
            print(f"  NumPy argmax: {action_np}")
            print(f"  Torch argmax: {action_torch}")
            print(f"  Position size: {position_size.item():.6f}")
            
            # Verify they match
            if action_np != action_torch:
                print(f"  *** MISMATCH: NumPy={action_np}, Torch={action_torch} ***")
            
            action_counts[action_torch] += 1
            
            # Action mapping
            action_names = ['HOLD', 'BUY', 'SELL']
            print(f"  Selected action: {action_torch} ({action_names[action_torch]})")
    
    print("\n" + "="*80)
    print("SUMMARY RESULTS")
    print("="*80)
    print(f"Total tests: {num_tests}")
    print(f"HOLD selections: {action_counts[0]} ({action_counts[0]/num_tests*100:.1f}%)")
    print(f"BUY selections: {action_counts[1]} ({action_counts[1]/num_tests*100:.1f}%)")
    print(f"SELL selections: {action_counts[2]} ({action_counts[2]/num_tests*100:.1f}%)")
    
    if action_counts[0] == num_tests:
        print("\n*** CRITICAL ISSUE: 100% HOLD BIAS DETECTED ***")
        print("The model ALWAYS selects HOLD regardless of input!")
        
        # Additional debugging
        print("\nDebugging model architecture...")
        debug_model_architecture(model, device)
    
    return action_counts

def debug_model_architecture(model, device):
    """Debug the model's internal architecture"""
    print("\n" + "-" * 40)
    print("MODEL ARCHITECTURE DEBUG")
    print("-" * 40)
    
    # Test a single forward pass step by step
    obs = torch.randn(1, 26).to(device)
    print(f"Input shape: {obs.shape}")
    
    # Check LSTM processing
    obs_reshaped = obs.unsqueeze(1)  # Add sequence dimension
    print(f"LSTM input shape: {obs_reshaped.shape}")
    
    lstm_out, _ = model.lstm(obs_reshaped)
    print(f"LSTM output shape: {lstm_out.shape}")
    print(f"LSTM output sample: {lstm_out[0, -1, :5]}")  # Last timestep, first 5 features
    
    last_out = lstm_out[:, -1, :]
    print(f"Last output shape: {last_out.shape}")
    
    # Check action head processing
    print("\nAction head processing:")
    for i, layer in enumerate(model.action_head):
        if isinstance(layer, nn.Linear):
            out = layer(last_out)
            print(f"  Layer {i} ({layer}): {out.shape}, sample: {out[0][:3]}")
            last_out = out
        elif isinstance(layer, nn.ReLU):
            out = layer(last_out)
            print(f"  Layer {i} (ReLU): {out.shape}, sample: {out[0][:3]}")
            last_out = out
        elif isinstance(layer, nn.Softmax):
            out = layer(last_out)
            print(f"  Layer {i} (Softmax): {out.shape}, sample: {out[0]}")
            print(f"  Softmax sum: {torch.sum(out).item()}")
            last_out = out
    
    # Check for weight issues
    print("\nWeight analysis:")
    for name, param in model.named_parameters():
        if 'action_head' in name:
            print(f"  {name}: shape={param.shape}, mean={param.mean().item():.6f}, "
                  f"std={param.std().item():.6f}, min={param.min().item():.6f}, "
                  f"max={param.max().item():.6f}")

def debug_softmax_specifically():
    """Specifically debug the softmax layer"""
    print("\n" + "-" * 40)
    print("SOFTMAX SPECIFIC DEBUG")
    print("-" * 40)
    
    # Test softmax with various inputs
    test_inputs = [
        torch.tensor([[0.0, 0.0, 0.0]]),  # Equal inputs
        torch.tensor([[1.0, 0.0, 0.0]]),  # Biased to first
        torch.tensor([[0.0, 1.0, 0.0]]),  # Biased to second
        torch.tensor([[0.0, 0.0, 1.0]]),  # Biased to third
        torch.tensor([[-1.0, 0.0, 1.0]]), # Mixed
        torch.tensor([[0.1, 0.1, 0.1]]),  # Small equal
    ]
    
    softmax = nn.Softmax(dim=-1)
    
    for i, inp in enumerate(test_inputs):
        out = softmax(inp)
        argmax_result = torch.argmax(out).item()
        print(f"Test {i+1}: Input={inp[0].numpy()}, Output={out[0].numpy()}, "
              f"Argmax={argmax_result}")

def main():
    """Main debugging function"""
    print("Starting focused action selection debugging...")
    
    # Debug action selection
    action_counts = debug_action_selection()
    
    # Debug softmax specifically
    debug_softmax_specifically()
    
    # If we still have 100% hold bias, there's a deeper issue
    if action_counts[0] == sum(action_counts):
        print("\n" + "="*80)
        print("CRITICAL FINDING: Model architecture or implementation issue!")
        print("The softmax is working correctly, but the model output is biased.")
        print("Recommendation: Check model weight initialization and architecture.")
        print("="*80)

if __name__ == "__main__":
    main()
