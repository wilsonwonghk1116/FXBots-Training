"""
SmartTradingBot: LSTM-based neural network for Forex trading.
"""

from typing import Tuple
import torch
import torch.nn as nn

class SmartTradingBot(nn.Module):
    """Enhanced neural network with LSTM for forex trading"""
    def __init__(self, input_size: int = 26, hidden_size: int = 512, output_size: int = 3):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size // 2,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        self.position_control = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for trading bot.
        Args:
            x: Input tensor of shape (batch, input_size)
        Returns:
            action_probs: Probability distribution over actions
            position_size: Suggested position size
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.unsqueeze(1)
        if x.size(-1) != self.input_size:
            if x.size(-1) < self.input_size:
                device = x.device
                padding = torch.zeros(x.size(0), 1, self.input_size - x.size(-1), device=device)
                x = torch.cat([x, padding], dim=-1)
            else:
                x = x[:, :, :self.input_size]
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        position_size = torch.sigmoid(self.position_control(last_out))
        action_probs = torch.softmax(self.action_head(last_out), dim=1)
        return action_probs, position_size
