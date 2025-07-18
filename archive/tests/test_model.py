import torch
from src.model import SmartTradingBot

def test_model_forward():
    model = SmartTradingBot()
    x = torch.randn(1, 20)
    action_probs, position_size = model(x)
    assert action_probs.shape == (1, 3)
    assert position_size.shape == (1, 1)
