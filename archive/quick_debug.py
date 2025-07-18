#!/usr/bin/env python3
"""
Quick Debug Script - Fast diagnosis of bot decision issues
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np

print("="*60)
print("QUICK BOT DEBUG - FAST DIAGNOSIS")
print("="*60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Define the exact model from the main script
class SmartTradingBot(nn.Module):
    def __init__(self, input_size: int = 26, hidden_size: int = 1024, output_size: int = 3):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.2)
        )
        
        # LSTM layers
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
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size // 2,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Market regime classifier
        self.regime_classifier = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 4),
            nn.Linear(hidden_size // 4, 4),
            nn.Softmax(dim=-1)
        )
        
        # Risk assessment network
        self.risk_network = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 4),
            nn.Linear(hidden_size // 4, hidden_size // 8),
            nn.ReLU(),
            nn.Linear(hidden_size // 8, 1),
            nn.Sigmoid()
        )
        
        # Position sizing network
        self.position_control = nn.Sequential(
            nn.Linear(hidden_size // 2 + 4 + 1, hidden_size // 4),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 4),
            nn.Linear(hidden_size // 4, hidden_size // 8),
            nn.ReLU(),
            nn.Linear(hidden_size // 8, 1),
            nn.Sigmoid()
        )
        
        # Action decision network (WITH SOFTMAX)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size // 2 + 4 + 1, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 4),
            nn.Linear(hidden_size // 4, output_size),
            nn.Softmax(dim=-1)  # THIS IS THE ISSUE!
        )
        
        # Confidence network
        self.confidence_network = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Input projection
        x_proj = self.input_projection(x)
        x_seq = x_proj.unsqueeze(1)
        
        # LSTM processing
        lstm1_out, _ = self.lstm1(x_seq)
        lstm2_out, _ = self.lstm2(lstm1_out)
        
        # Attention
        attn_out, _ = self.attention(lstm2_out, lstm2_out, lstm2_out)
        features = attn_out.squeeze(1)
        
        # Market regime classification
        regime_probs = self.regime_classifier(features)
        
        # Risk assessment
        risk_score = self.risk_network(features)
        
        # Confidence calculation
        confidence = self.confidence_network(features)
        
        # Combine features
        combined_features = torch.cat([features, regime_probs, risk_score], dim=1)
        
        # Position sizing
        position_size = self.position_control(combined_features) * confidence
        
        # Action probabilities (ALREADY HAS SOFTMAX!)
        action_probs = self.action_head(combined_features)
        
        # PROBLEM: Confidence weighting with balanced probabilities
        if confidence.dim() == 1:
            confidence = confidence.unsqueeze(1)
        action_probs = action_probs * confidence + (1 - confidence) * torch.tensor([0.33, 0.33, 0.34], device=x.device)
        # NO additional softmax needed!
        
        return action_probs, position_size

# Test the model
try:
    print("1. Creating model...")
    model = SmartTradingBot().to(device)
    model.eval()
    print("   ✓ Model created successfully")
    
    print("\n2. Testing action selection (10 tests):")
    action_counts = [0, 0, 0]  # hold, buy, sell
    
    for i in range(10):
        obs = torch.randn(1, 26).to(device)
        
        with torch.no_grad():
            action_probs, position_size = model(obs)
            action = torch.argmax(action_probs).item()
            
        probs_str = f"[{action_probs[0][0]:.3f}, {action_probs[0][1]:.3f}, {action_probs[0][2]:.3f}]"
        action_names = ['HOLD', 'BUY', 'SELL']
        print(f"   Test {i+1:2d}: {action_names[action]:4s} - Probs={probs_str}")
        action_counts[action] += 1
    
    print(f"\n3. RESULTS:")
    print(f"   HOLD: {action_counts[0]}/10 ({action_counts[0]*10:3d}%)")
    print(f"   BUY:  {action_counts[1]}/10 ({action_counts[1]*10:3d}%)")
    print(f"   SELL: {action_counts[2]}/10 ({action_counts[2]*10:3d}%)")
    
    print(f"\n4. DIAGNOSIS:")
    if action_counts[0] >= 8:
        print("   *** CRITICAL: HOLD BIAS - Model always chooses HOLD ***")
        print("   CAUSE: Confidence weighting or action_head softmax issue")
    elif action_counts[1] >= 8:
        print("   *** CRITICAL: BUY BIAS - Model always chooses BUY ***")
        print("   CAUSE: Action head or confidence calculation bias")
    elif action_counts[2] >= 8:
        print("   *** CRITICAL: SELL BIAS - Model always chooses SELL ***")
        print("   CAUSE: Action head or confidence calculation bias")
    else:
        print("   ✓ GOOD: Actions are varying - model working correctly!")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("QUICK DEBUG COMPLETE")
print("="*60)
