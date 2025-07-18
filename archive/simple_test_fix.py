#!/usr/bin/env python3
"""
Simple test of the fixed environment reset issue
"""

import torch
import sys
import signal

def timeout_handler(signum, frame):
    print("⏰ Timeout reached")
    sys.exit(1)

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(30)  # 30 second timeout

try:
    print("Testing fixed environment reset...")
    
    from run_smart_real_training import SmartTradingBot, SmartForexEnvironment
    
    # Create bot and environment
    bot = SmartTradingBot()
    env = SmartForexEnvironment()
    
    print("✅ Components created")
    
    # Reset environment
    obs, _ = env.reset()
    print(f"✅ Environment reset - current_step: {env.current_step}, last_trade_step: {env.last_trade_step}")
    
    # Test a few steps
    actions_taken = []
    for step in range(5):
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        
        with torch.no_grad():
            action_probs, position_size = bot(obs_tensor)
            action = torch.argmax(action_probs).item()
        
        actions_taken.append(action)
        print(f"Step {step}: Action {action} ({['HOLD', 'BUY', 'SELL'][action]}), Probs: {action_probs.numpy()}")
        
        obs, reward, done, _, info = env.step(action, position_size.item())
        print(f"  Reward: {reward:.2f}, Done: {done}, Trades: {len(env.trades)}")
        
        if done:
            print("  Episode ended")
            break
    
    print(f"\n✅ SUCCESS: Completed {step + 1} steps without issues")
    print(f"Actions taken: {actions_taken}")
    print(f"Unique actions: {len(set(actions_taken))}")
    print(f"Total trades: {len(env.trades)}")
    
    signal.alarm(0)  # Cancel timeout
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
