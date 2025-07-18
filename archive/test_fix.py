#!/usr/bin/env python3
"""
Test Fix for Action Bias
========================

This script tests the fix for the action selection bias that was preventing trades.
"""

import sys
import os
import torch
import numpy as np
import logging

# Add project root to path  
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import the MAIN training models to test the fix
import importlib.util

# Load the main script to get the fixed SmartTradingBot
spec = importlib.util.spec_from_file_location("main", "run_smart_real_training.py")
main_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main_module)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_action_bias_fix():
    """Test that the action bias has been fixed"""
    print("="*80)
    print("TESTING ACTION BIAS FIX")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Use the FIXED SmartTradingBot from the main training script
    model = main_module.SmartTradingBot().to(device)
    model.eval()
    
    print(f"Device: {device}")
    print("Testing with FIXED SmartTradingBot from main training script...")
    
    # Test multiple random inputs
    num_tests = 50
    action_counts = [0, 0, 0]  # hold, buy, sell
    
    print(f"\nTesting action selection with {num_tests} random inputs:")
    print("-" * 80)
    
    for i in range(num_tests):
        # Create random observation (26 features)
        obs = np.random.randn(26)
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        
        with torch.no_grad():
            action_probs, position_size = model(obs_tensor)
            action = torch.argmax(action_probs).item()
            
            action_counts[action] += 1
            
            # Print every 10th test
            if (i + 1) % 10 == 0:
                print(f"Test {i+1}: Action probs: {action_probs[0].cpu().numpy()}, Selected: {action}")
    
    print("\n" + "="*80)
    print("RESULTS AFTER FIX")
    print("="*80)
    print(f"Total tests: {num_tests}")
    print(f"HOLD selections: {action_counts[0]} ({action_counts[0]/num_tests*100:.1f}%)")
    print(f"BUY selections: {action_counts[1]} ({action_counts[1]/num_tests*100:.1f}%)")  
    print(f"SELL selections: {action_counts[2]} ({action_counts[2]/num_tests*100:.1f}%)")
    
    # Check if the bias is fixed
    max_bias = max(action_counts)
    if max_bias > num_tests * 0.8:  # More than 80% bias
        print(f"\n❌ STILL BIASED: One action selected {max_bias}/{num_tests} times ({max_bias/num_tests*100:.1f}%)")
        return False
    else:
        print(f"\n✅ BIAS FIXED: Actions are more balanced!")
        return True

def test_trading_simulation():
    """Test actual trading simulation to see if trades execute"""
    print("\n" + "="*80)
    print("TESTING TRADING SIMULATION")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create environment and model
    env = main_module.SmartForexEnvironment()
    model = main_module.SmartTradingBot().to(device)
    model.eval()
    
    env.reset()
    trades_executed = 0
    total_steps = 200
    
    print(f"Running {total_steps} step simulation...")
    
    for step in range(total_steps):
        obs = env._get_observation()
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        
        with torch.no_grad():
            action_probs, position_size = model(obs_tensor)
            action = torch.argmax(action_probs).item()
        
        obs, reward, done, _, info = env.step(action, position_size.item())
        
        if info.get('trade_executed', False):
            trades_executed += 1
            print(f"Step {step}: TRADE EXECUTED! Action={action}, Reward={reward:.2f}")
        
        if done:
            break
    
    print(f"\nSimulation Results:")
    print(f"Total steps: {step + 1}")
    print(f"Trades executed: {trades_executed}")
    print(f"Final balance: {env.balance:.2f}")
    print(f"Total trades in history: {len(env.trades)}")
    
    if trades_executed > 0:
        print("✅ SUCCESS: Trades are now being executed!")
        return True
    else:
        print("❌ STILL NO TRADES: Further investigation needed")
        return False

def main():
    """Main test function"""
    print("Testing fix for action selection bias...")
    
    # Test 1: Action bias fix
    bias_fixed = test_action_bias_fix()
    
    # Test 2: Trading simulation  
    trades_working = test_trading_simulation()
    
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    
    if bias_fixed and trades_working:
        print("🎉 ALL TESTS PASSED! The fix appears to be working!")
    elif bias_fixed:
        print("⚠️  Action bias fixed but still no trades - may need additional investigation")
    else:
        print("❌ Tests failed - additional fixes needed")

if __name__ == "__main__":
    main()
