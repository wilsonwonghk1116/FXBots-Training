#!/usr/bin/env python3
"""
Quick Test - Verify the SELL bias fix
"""

import torch
import torch.nn as nn

print("="*50)
print("TESTING SELL BIAS FIX")
print("="*50)

# Copy the FIXED model
class SmartTradingBot(nn.Module):
    def __init__(self, input_size: int = 26, hidden_size: int = 1024, output_size: int = 3):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Simplified for quick test
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.2)
        )
        
        self.lstm1 = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=3,
            batch_first=True,
            dropout=0.3
        )
        
        self.lstm2 = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size // 2,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        self.regime_classifier = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 4),
            nn.Linear(hidden_size // 4, 4),
            nn.Softmax(dim=-1)
        )
        
        self.risk_network = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 4),
            nn.Linear(hidden_size // 4, hidden_size // 8),
            nn.ReLU(),
            nn.Linear(hidden_size // 8, 1),
            nn.Sigmoid()
        )
        
        self.position_control = nn.Sequential(
            nn.Linear(hidden_size // 2 + 4 + 1, hidden_size // 4),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 4),
            nn.Linear(hidden_size // 4, hidden_size // 8),
            nn.ReLU(),
            nn.Linear(hidden_size // 8, 1),
            nn.Sigmoid()
        )
        
        # Action head with softmax
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size // 2 + 4 + 1, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 4),
            nn.Linear(hidden_size // 4, output_size),
            nn.Softmax(dim=-1)
        )
        
        self.confidence_network = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Forward pass
        x_proj = self.input_projection(x)
        x_seq = x_proj.unsqueeze(1)
        
        lstm1_out, _ = self.lstm1(x_seq)
        lstm2_out, _ = self.lstm2(lstm1_out)
        
        attn_out, _ = self.attention(lstm2_out, lstm2_out, lstm2_out)
        features = attn_out.squeeze(1)
        
        regime_probs = self.regime_classifier(features)
        risk_score = self.risk_network(features)
        confidence = self.confidence_network(features)
        
        combined_features = torch.cat([features, regime_probs, risk_score], dim=1)
        position_size = self.position_control(combined_features) * confidence
        
        # FIXED: Use action_head output directly - no confidence weighting!
        action_probs = self.action_head(combined_features)
        
        return action_probs, position_size

# Test the fixed model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SmartTradingBot().to(device)
model.eval()

print("Testing with different random inputs:")
action_counts = [0, 0, 0]
all_probs = []

for i in range(15):
    # Use different random seeds to get varied inputs
    torch.manual_seed(i * 42)
    obs = torch.randn(1, 26).to(device)
    
    with torch.no_grad():
        action_probs, position_size = model(obs)
        action = torch.argmax(action_probs).item()
        
    probs = action_probs[0].cpu().numpy()
    all_probs.append(probs)
    action_counts[action] += 1
    
    action_names = ['HOLD', 'BUY', 'SELL']
    print(f"Test {i+1:2d}: {action_names[action]:4s} - Probs=[{probs[0]:.3f}, {probs[1]:.3f}, {probs[2]:.3f}]")

print(f"\nRESULTS:")
print(f"HOLD: {action_counts[0]}/15 ({action_counts[0]*100//15:2d}%)")
print(f"BUY:  {action_counts[1]}/15 ({action_counts[1]*100//15:2d}%)")
print(f"SELL: {action_counts[2]}/15 ({action_counts[2]*100//15:2d}%)")

# Check if probabilities are varying
import numpy as np
all_probs = np.array(all_probs)
prob_std = np.std(all_probs, axis=0)

print(f"\nPROBABILITY VARIATION:")
print(f"HOLD std: {prob_std[0]:.4f}")
print(f"BUY std:  {prob_std[1]:.4f}")
print(f"SELL std: {prob_std[2]:.4f}")

if np.all(prob_std > 0.001):
    print("\n✓ SUCCESS: Probabilities are varying with different inputs!")
    print("✓ Model is now responding to inputs properly!")
else:
    print("\n✗ ISSUE: Probabilities are still too similar")
    print("✗ Model may still have bias issues")

print("="*50)
