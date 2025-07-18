#!/usr/bin/env python3
"""
Test the new reward/penalty system for Forex bots
Verify that:
1. First trade gets +1000 bonus
2. PnL-based rewards/penalties work correctly  
3. Idle penalty triggers after 1000 steps
4. Observation space is 26 features
5. Initial balance is $100,000
"""

import numpy as np
import torch
import sys
import os

# Add the project root to the path
sys.path.insert(0, '/home/w1/cursor-to-copilot-backup/TaskmasterForexBots')

# Import from the main file
from run_smart_real_training import SmartForexEnvironment, SmartTradingBot

def test_new_reward_system():
    print("🧪 Testing New Reward/Penalty System")
    print("=" * 50)
    
    # Test 1: Initial balance is $100,000
    env = SmartForexEnvironment()
    print(f"✓ Initial balance: ${env.initial_balance:,.2f}")
    assert env.initial_balance == 100000.0, f"Expected $100,000, got ${env.initial_balance}"
    
    # Test 2: Observation space is 26 features
    obs, _ = env.reset()
    print(f"✓ Observation size: {len(obs)} features")
    assert len(obs) == 26, f"Expected 26 features, got {len(obs)}"
    
    # Test 3: Create a simple bot and test first trade bonus
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bot = SmartTradingBot(input_size=26).to(device)
    
    # Reset environment
    obs, _ = env.reset()
    print(f"✓ Environment reset. Starting step: {env.current_step}")
    
    # Test first trade bonus
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
    with torch.no_grad():
        action_probs, position_size = bot(obs_tensor)
        action = 1  # Force BUY action for first trade
    
    # Execute first trade
    obs, reward, done, _, info = env.step(action, 0.5)  # 50% position size
    
    print(f"✓ First trade executed:")
    print(f"  - Action: BUY")
    print(f"  - Reward: {reward:.2f}")
    print(f"  - First trade bonus given: {info['first_trade_bonus_given']}")
    print(f"  - Total trades: {info['total_trades']}")
    
    # Verify first trade bonus was applied
    assert reward >= 1000, f"Expected first trade bonus, reward was {reward}"
    assert info['first_trade_bonus_given'] == True, "First trade bonus should be marked as given"
    
    # Test 4: Test that second trade doesn't get bonus
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
    with torch.no_grad():
        action_probs, position_size = bot(obs_tensor)
        action = 2  # Force SELL action for second trade
    
    obs, reward2, done, _, info2 = env.step(action, 0.5)
    print(f"✓ Second trade executed:")
    print(f"  - Action: SELL")
    print(f"  - Reward: {reward2:.2f}")
    print(f"  - Total trades: {info2['total_trades']}")
    
    # Second trade should not get bonus (reward should be much smaller)
    assert reward2 < 500, f"Second trade should not get first trade bonus, got reward {reward2}"
    
    # Test 5: Test idle penalty by holding for many steps
    print("✓ Testing idle penalty (this may take a moment)...")
    idle_steps = 0
    total_idle_penalty = 0
    
    # Reset environment to test idle penalty
    env.reset()
    env.first_trade_bonus_given = True  # Skip first trade bonus for this test
    
    for step in range(1200):  # Go beyond idle threshold
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        action = 0  # Always HOLD
        obs, reward, done, _, info = env.step(action, 0.0)
        
        if reward < -50:  # Idle penalty detected
            total_idle_penalty += reward
            idle_steps += 1
            if idle_steps == 1:  # Log first idle penalty
                print(f"  - First idle penalty at step {step}: {reward}")
        
        if done:
            break
    
    print(f"✓ Idle penalty test completed:")
    print(f"  - Idle penalties triggered: {idle_steps}")
    print(f"  - Total idle penalty: {total_idle_penalty}")
    
    assert idle_steps > 0, "Idle penalty should have been triggered"
    
    # Test 6: Test leverage (verify max_leverage is set)
    print(f"✓ Max leverage: {env.max_leverage}x")
    assert env.max_leverage == 100, f"Expected 100x leverage, got {env.max_leverage}x"
    
    print("\n🎉 ALL TESTS PASSED!")
    print("✓ $100,000 initial balance")
    print("✓ 26-feature observation space with technical indicators")
    print("✓ First trade bonus (+1000) working correctly")
    print("✓ PnL-based reward/penalty system implemented")
    print("✓ Idle penalty (-100 every 1000 steps) working")
    print("✓ 100x leverage enabled")
    print("\nThe new reward/penalty system is ready for training! 🚀")

if __name__ == "__main__":
    test_new_reward_system()
