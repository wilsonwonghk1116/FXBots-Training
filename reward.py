"""
reward.py
PnL-based reward/penalty logic for RL.
"""
def compute_reward(pnl: float) -> float:
    """Return RL reward as PnL (profit = positive, loss = negative)."""
    return pnl
# To extend, add more complex reward shaping functions here. 