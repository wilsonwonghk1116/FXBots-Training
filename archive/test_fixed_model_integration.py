#!/usr/bin/env python3
"""
Test the fixed SmartTradingBot in the actual training environment.
This script will verify that the fixed model works properly in the real training context.
"""

import torch
import numpy as np
import sys
import os

# Import the fixed model from the main training script
from run_smart_real_training import SmartTradingBot, SmartForexEnvironment

def test_fixed_model_in_environment():
    """Test the fixed model with the actual forex environment"""
    print("Testing Fixed SmartTradingBot in Forex Environment...")
    print("=" * 60)
    
    # Create environment and model
    env = SmartForexEnvironment()
    model = SmartTradingBot(input_size=26)
    model.eval()
    
    print(f"Model input size: {model.input_size}")
    print(f"Model hidden size: {model.hidden_size}")
    print(f"Model temperature: {model.temperature.item():.4f}")
    print()
    
    # Test a few steps in the environment
    obs, _ = env.reset()  # Environment returns (obs, info) tuple
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial observation sample: {obs[:5]}")
    print(f"Initial balance: ${env.balance:,.2f}")
    print()
    
    # Test model with actual environment observations
    actions_taken = []
    rewards_received = []
    
    for step in range(10):
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        
        with torch.no_grad():
            action_probs, position_size = model(obs_tensor)
            action = torch.argmax(action_probs).item()
            
        actions_taken.append(action)
        
        print(f"Step {step + 1}:")
        print(f"  Action Probabilities: HOLD={action_probs[0]:.4f}, BUY={action_probs[1]:.4f}, SELL={action_probs[2]:.4f}")
        print(f"  Position Size: {position_size.item():.4f}")
        print(f"  Chosen Action: {['HOLD', 'BUY', 'SELL'][action]}")
        
        # Take action in environment
        obs, reward, done, _, info = env.step(action, position_size.item())
        rewards_received.append(reward)
        
        print(f"  Reward: {reward:.6f}")
        print(f"  New Balance: ${env.balance:,.2f}")
        print(f"  Position: {env.position}")
        print(f"  Total Trades: {len(env.trades)}")
        print()
        
        if done:
            print("Episode finished!")
            break
    
    # Analyze results
    print("Analysis:")
    print("-" * 30)
    
    unique_actions = set(actions_taken)
    print(f"Actions taken: {actions_taken}")
    print(f"Unique actions: {unique_actions}")
    print(f"Number of unique actions: {len(unique_actions)}")
    
    action_counts = {0: 0, 1: 0, 2: 0}  # HOLD, BUY, SELL
    for action in actions_taken:
        action_counts[action] += 1
    
    print(f"Action distribution:")
    print(f"  HOLD: {action_counts[0]}")
    print(f"  BUY:  {action_counts[1]}")
    print(f"  SELL: {action_counts[2]}")
    
    total_reward = sum(rewards_received)
    print(f"Total reward: {total_reward:.6f}")
    print(f"Average reward: {total_reward/len(rewards_received):.6f}")
    
    # Check if trades were executed
    if len(env.trades) > 0:
        print(f"✅ SUCCESS: {len(env.trades)} trades were executed!")
        for i, trade in enumerate(env.trades):
            print(f"  Trade {i+1}: {trade}")
    else:
        print("⚠️  No trades executed - all HOLD actions")
    
    # Test model responsiveness
    print("\nTesting Model Responsiveness:")
    print("-" * 30)
    
    # Create different market scenarios
    test_scenarios = []
    
    # Bullish scenario
    bullish_obs = obs.copy()
    bullish_obs[:10] = np.linspace(obs[0], obs[0] * 1.1, 10)  # Rising prices
    bullish_obs[10] = 0.3  # Low RSI
    test_scenarios.append(("Bullish", bullish_obs))
    
    # Bearish scenario
    bearish_obs = obs.copy()
    bearish_obs[:10] = np.linspace(obs[0], obs[0] * 0.9, 10)  # Falling prices
    bearish_obs[10] = 0.8  # High RSI
    test_scenarios.append(("Bearish", bearish_obs))
    
    scenario_results = []
    for name, scenario_obs in test_scenarios:
        obs_tensor = torch.tensor(scenario_obs, dtype=torch.float32)
        with torch.no_grad():
            action_probs, position_size = model(obs_tensor)
            action = torch.argmax(action_probs).item()
        
        scenario_results.append((name, action_probs.numpy(), action))
        print(f"{name}: {['HOLD', 'BUY', 'SELL'][action]} (probs: {action_probs.numpy()})")
    
    # Check if model responds differently to different scenarios
    if len(set(result[2] for result in scenario_results)) > 1:
        print("✅ Model responds differently to different market scenarios!")
    else:
        print("⚠️  Model gives same action for different scenarios")
    
    return len(unique_actions) > 1, len(env.trades) > 0

def run_training_step_test():
    """Test a few training steps to ensure the model can learn"""
    print("\n" + "=" * 60)
    print("Testing Model Training Step...")
    print("=" * 60)
    
    model = SmartTradingBot(input_size=26)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create a batch of observations
    batch_size = 8
    obs_batch = torch.randn(batch_size, 26)
    target_actions = torch.randint(0, 3, (batch_size,))
    
    # Forward pass
    action_probs, position_sizes = model(obs_batch)
    
    # Simple loss function
    loss = torch.nn.CrossEntropyLoss()(torch.log(action_probs + 1e-8), target_actions)
    
    print(f"Batch size: {batch_size}")
    print(f"Action probabilities shape: {action_probs.shape}")
    print(f"Position sizes shape: {position_sizes.shape}")
    print(f"Loss: {loss.item():.6f}")
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Check gradients
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
    
    print(f"Gradient norms: min={min(grad_norms):.6f}, max={max(grad_norms):.6f}, avg={np.mean(grad_norms):.6f}")
    
    optimizer.step()
    
    print("✅ Training step completed successfully!")
    
    return True

if __name__ == "__main__":
    print("SmartTradingBot Integration Test")
    print("=" * 60)
    
    # Test in environment
    has_variety, executes_trades = test_fixed_model_in_environment()
    
    # Test training step
    can_train = run_training_step_test()
    
    print("\n" + "=" * 60)
    print("INTEGRATION TEST SUMMARY:")
    print("=" * 60)
    
    if has_variety:
        print("✅ Model produces varied actions")
    else:
        print("⚠️  Model actions lack variety")
    
    if executes_trades:
        print("✅ Model executes trades in environment")
    else:
        print("⚠️  Model doesn't execute trades (all HOLD)")
    
    if can_train:
        print("✅ Model can be trained with gradients")
    else:
        print("❌ Model training failed")
    
    if has_variety and can_train:
        print("\n🎉 SUCCESS: Fixed model is ready for training!")
        print("   The model should learn to make better trading decisions with proper training.")
    else:
        print("\n⚠️  Issues remain - further debugging may be needed.")
