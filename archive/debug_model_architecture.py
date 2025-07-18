#!/usr/bin/env python3
"""
Debug and fix the SmartTradingBot model architecture issues.
This script will analyze and fix the fundamental problems causing identical probabilities.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add the current directory to the path to import the model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class FixedSmartTradingBot(nn.Module):
    """Fixed version of SmartTradingBot with proper architecture"""
    
    def __init__(self, input_size: int = 26, hidden_size: int = 512, output_size: int = 3):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Simplified but effective architecture
        self.input_norm = nn.LayerNorm(input_size)  # LayerNorm instead of BatchNorm
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Simple LSTM for temporal patterns
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Action network - separate from position sizing
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, output_size)
            # NO softmax here - we'll apply it in forward
        )
        
        # Position sizing network
        self.position_head = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        # Proper weight initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Proper weight initialization"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param)
    
    def forward(self, x):
        # Handle both single samples and batches
        if x.dim() == 1:
            x = x.unsqueeze(0)
            single_sample = True
        else:
            single_sample = False
        
        batch_size = x.size(0)
        
        # Normalize input
        x = self.input_norm(x)
        
        # Feature extraction
        features = self.feature_extractor(x)
        
        # Add sequence dimension for LSTM
        x_seq = features.unsqueeze(1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x_seq)
        lstm_features = lstm_out.squeeze(1)
        
        # Action probabilities
        action_logits = self.action_head(lstm_features)
        action_probs = torch.softmax(action_logits, dim=-1)
        
        # Position sizing
        position_size = self.position_head(lstm_features)
        
        # If single sample, squeeze the batch dimension
        if single_sample:
            action_probs = action_probs.squeeze(0)
            position_size = position_size.squeeze(0)
        
        return action_probs, position_size

def test_model_responsiveness():
    """Test if the model produces different outputs for different inputs"""
    print("Testing Fixed SmartTradingBot Model Responsiveness...")
    print("=" * 60)
    
    model = FixedSmartTradingBot(input_size=26)
    model.eval()
    
    # Create diverse test inputs
    test_inputs = []
    
    # Test 1: All zeros
    test_inputs.append(torch.zeros(26))
    
    # Test 2: All ones
    test_inputs.append(torch.ones(26))
    
    # Test 3: Random values
    test_inputs.append(torch.randn(26))
    
    # Test 4: Bullish pattern (increasing prices)
    bullish = torch.zeros(26)
    bullish[:10] = torch.linspace(1.0, 1.1, 10)  # Increasing prices
    bullish[10] = 0.3  # Low RSI (oversold)
    bullish[13] = 1.0  # MACD signal
    test_inputs.append(bullish)
    
    # Test 5: Bearish pattern (decreasing prices)
    bearish = torch.zeros(26)
    bearish[:10] = torch.linspace(1.1, 1.0, 10)  # Decreasing prices
    bearish[10] = 0.8  # High RSI (overbought)
    bearish[13] = -1.0  # MACD signal
    test_inputs.append(bearish)
    
    results = []
    with torch.no_grad():
        for i, input_tensor in enumerate(test_inputs):
            action_probs, position_size = model(input_tensor)
            results.append((action_probs.numpy(), position_size.item()))
            
            print(f"Test {i+1} Input Summary: {input_tensor[:5].numpy()}")
            print(f"  Action Probabilities: HOLD={action_probs[0]:.4f}, BUY={action_probs[1]:.4f}, SELL={action_probs[2]:.4f}")
            print(f"  Position Size: {position_size.item():.4f}")
            print(f"  Chosen Action: {['HOLD', 'BUY', 'SELL'][torch.argmax(action_probs)]}")
            print()
    
    # Check if outputs are different
    print("Checking Model Responsiveness:")
    print("-" * 30)
    
    all_same = True
    first_probs = results[0][0]
    
    for i, (probs, pos_size) in enumerate(results[1:], 1):
        if not np.allclose(probs, first_probs, atol=1e-4):
            all_same = False
            break
    
    if all_same:
        print("❌ PROBLEM: Model outputs identical probabilities for all inputs!")
        print("❌ The model is not responsive to input changes.")
        return False
    else:
        print("✅ SUCCESS: Model produces different outputs for different inputs!")
        print("✅ The model is responsive to input changes.")
        
        # Check for diversity in actions
        actions = [torch.argmax(torch.tensor(probs)).item() for probs, _ in results]
        unique_actions = set(actions)
        print(f"✅ Actions chosen: {unique_actions}")
        print(f"✅ Number of unique actions: {len(unique_actions)}")
        
        return True

def compare_with_original():
    """Compare with the original broken model"""
    print("\n" + "=" * 60)
    print("Comparing with Original Model...")
    print("=" * 60)
    
    # Import the original model
    try:
        from run_smart_real_training import SmartTradingBot as OriginalBot
        
        print("Testing Original Model:")
        original_model = OriginalBot(input_size=26)
        original_model.eval()
        
        test_input = torch.randn(26)
        
        with torch.no_grad():
            try:
                action_probs, position_size = original_model(test_input)
                print(f"  Original Action Probs: {action_probs.numpy()}")
                print(f"  Original Position Size: {position_size.item()}")
            except Exception as e:
                print(f"  ❌ Original model failed: {e}")
        
        print("\nTesting Fixed Model:")
        fixed_model = FixedSmartTradingBot(input_size=26)
        fixed_model.eval()
        
        with torch.no_grad():
            action_probs, position_size = fixed_model(test_input)
            print(f"  Fixed Action Probs: {action_probs.numpy()}")
            print(f"  Fixed Position Size: {position_size.item()}")
            
    except Exception as e:
        print(f"Could not import original model: {e}")

def test_gradient_flow():
    """Test if gradients flow properly through the model"""
    print("\n" + "=" * 60)
    print("Testing Gradient Flow...")
    print("=" * 60)
    
    model = FixedSmartTradingBot(input_size=26)
    model.train()
    
    # Create a simple loss scenario
    input_tensor = torch.randn(26, requires_grad=True)
    target_action = torch.tensor([0.0, 1.0, 0.0])  # Want BUY action
    
    action_probs, position_size = model(input_tensor)
    
    # Simple loss function
    loss = nn.MSELoss()(action_probs, target_action)
    loss.backward()
    
    # Check if gradients exist
    has_gradients = False
    gradient_norms = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_gradients = True
            grad_norm = param.grad.norm().item()
            gradient_norms.append(grad_norm)
            if grad_norm > 1e-6:  # Significant gradient
                print(f"  {name}: gradient norm = {grad_norm:.6f}")
    
    if has_gradients and any(g > 1e-6 for g in gradient_norms):
        print("✅ SUCCESS: Gradients are flowing properly!")
        print(f"✅ Average gradient norm: {np.mean(gradient_norms):.6f}")
        return True
    else:
        print("❌ PROBLEM: No significant gradients detected!")
        return False

if __name__ == "__main__":
    print("SmartTradingBot Architecture Debug and Fix")
    print("=" * 60)
    
    # Test the fixed model
    responsive = test_model_responsiveness()
    
    # Compare with original
    compare_with_original()
    
    # Test gradient flow
    gradients_ok = test_gradient_flow()
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("=" * 60)
    
    if responsive and gradients_ok:
        print("✅ SUCCESS: Fixed model is working properly!")
        print("✅ Ready to replace the original model in the training script.")
        print("\nNext steps:")
        print("1. Replace SmartTradingBot class in run_smart_real_training.py")
        print("2. Run training to verify bots start making diverse trading decisions")
        print("3. Monitor that bots execute actual trades (not just HOLD)")
    else:
        print("❌ ISSUES DETECTED: Further debugging needed.")
        if not responsive:
            print("   - Model not responsive to input changes")
        if not gradients_ok:
            print("   - Gradient flow problems")
