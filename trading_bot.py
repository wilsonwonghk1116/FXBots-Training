"""
trading_bot.py
TradingBot class for distributed evolutionary RL.
"""
import torch
import torch.nn as nn
import numpy as np
# Import ML models (LSTM, GRU, etc.) from predictors.py
from predictors import *

class TradingBot(nn.Module):
    def __init__(self, strategy_type: str, device: str = 'cpu', input_size=10, hidden_size=64, output_size=3):
        super().__init__()
        self.strategy_type = strategy_type
        self.device = device
        
        # The bot's 'brain' - a simple feed-forward neural network
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=-1)
        ).to(self.device)

    def decide_action(self, state):
        """Uses the neural network to decide an action based on the current market state."""
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        
        # Ensure the state tensor is 2D (batch_size, features)
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        with torch.no_grad():
            action_probs = self.network(state)
        
        # Return the action with the highest probability
        return torch.argmax(action_probs, dim=1).item()

    def mutate(self, mutation_rate=0.05, mutation_strength=0.1):
        """Apply evolutionary mutation to bot's neural network parameters."""
        with torch.no_grad():
            for param in self.parameters():
                if torch.rand(1).item() < mutation_rate:
                    noise = torch.randn_like(param) * mutation_strength
                    param.add_(noise)

    def crossover(self, other_bot):
        """Crossover with another bot to produce a new offspring bot."""
        child = TradingBot(strategy_type=self.strategy_type, device=self.device)
        child.to(self.device)

        with torch.no_grad():
            for self_param, other_param, child_param in zip(self.parameters(), other_bot.parameters(), child.parameters()):
                # Choose a random crossover point for each parameter tensor
                mask = torch.rand_like(self_param) > 0.5
                child_param.data.copy_(torch.where(mask, self_param.data, other_param.data))
        return child

    def update_reward(self, pnl):
        """Stub for RL reward update."""
        # This can be expanded for more complex reinforcement learning later
        pass
        
    # Add stubs for RL agent integration, checkpointing, etc. 
    def forward(self, x):
        return self.network(x) 