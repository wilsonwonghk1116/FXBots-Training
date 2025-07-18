#!/usr/bin/env python3
"""
MASSIVELY SCALED TRAINING CONFIGURATION
=======================================
Enhanced for dual-PC Ray cluster with:
- 200 generations
- 1000 episodes per generation  
- 1000 steps per episode
- Advanced PnL-based reward system
- Comprehensive reinforcement learning
"""

# Enhanced Training Configuration
TRAINING_CONFIG = {
    "generations": 200,           # Massive scale: 200 generations
    "episodes_per_generation": 1000,  # 1000 episodes per generation
    "steps_per_episode": 1000,        # 1000 trading steps per episode
    "total_training_steps": 200_000_000,  # 200 million total steps
    
    # PnL Reward System Configuration
    "reward_system": {
        "base_reward_multiplier": 1.0,    # $1 USD = 1 reward point
        "profit_bonus_multiplier": 1.2,   # 20% bonus for profits
        "loss_penalty_multiplier": 1.1,   # 10% extra penalty for losses
        "win_streak_bonus": 0.1,          # 10% bonus per consecutive win
        "max_win_streak_bonus": 2.0,      # Max 200% win streak bonus
        "drawdown_penalty": 2.0,          # 2x penalty for large drawdowns
        "sharpe_bonus": 0.5,              # Bonus for good risk-adjusted returns
    },
    
    # Ray Cluster Configuration
    "ray_config": {
        "workers_per_generation": 20,     # 20 workers per generation
        "cpus_per_worker": 12,            # 12 CPUs per worker (75% of 16)
        "gpus_per_worker": 1,             # 1 GPU per worker
        "episodes_per_worker": 50,        # 50 episodes per worker (1000/20)
    },
    
    # Performance Tracking
    "tracking": {
        "save_frequency": 10,             # Save every 10 generations
        "checkpoint_frequency": 50,       # Checkpoint every 50 generations
        "analysis_frequency": 25,         # Deep analysis every 25 generations
    }
}

class EnhancedPnLRewardSystem:
    """Advanced PnL-based reward system for reinforcement learning"""
    
    def __init__(self, config=TRAINING_CONFIG["reward_system"]):
        self.config = config
        self.trade_history = []
        self.win_streak = 0
        self.total_rewards = 0.0
        self.total_penalties = 0.0
        
    def calculate_trade_reward(self, pnl_usd: float, additional_metrics: dict = None) -> float:
        """
        Calculate reward/penalty for a single trade based on PnL and metrics
        
        Args:
            pnl_usd: Profit/Loss in USD for the trade
            additional_metrics: Dict with sharpe_ratio, drawdown, etc.
        
        Returns:
            float: Reward (positive) or penalty (negative)
        """
        base_reward = pnl_usd * self.config["base_reward_multiplier"]
        
        # Apply profit bonus or loss penalty
        if pnl_usd > 0:
            # Profit: apply bonus
            reward = base_reward * self.config["profit_bonus_multiplier"]
            
            # Win streak bonus
            self.win_streak += 1
            streak_multiplier = min(
                1.0 + (self.win_streak * self.config["win_streak_bonus"]),
                self.config["max_win_streak_bonus"]
            )
            reward *= streak_multiplier
            
            self.total_rewards += reward
            
        else:
            # Loss: apply penalty
            reward = base_reward * self.config["loss_penalty_multiplier"]
            self.win_streak = 0  # Reset win streak
            self.total_penalties += abs(reward)
        
        # Additional metrics bonuses/penalties
        if additional_metrics:
            # Sharpe ratio bonus
            sharpe = additional_metrics.get("sharpe_ratio", 0)
            if sharpe > 1.0:
                reward += (sharpe - 1.0) * self.config["sharpe_bonus"] * abs(pnl_usd)
            
            # Drawdown penalty
            drawdown = additional_metrics.get("max_drawdown", 0)
            if drawdown > 0.1:  # > 10% drawdown
                penalty = (drawdown - 0.1) * self.config["drawdown_penalty"] * abs(pnl_usd)
                reward -= penalty
        
        # Store trade in history
        self.trade_history.append({
            "pnl_usd": pnl_usd,
            "reward": reward,
            "win_streak": self.win_streak,
            "timestamp": time.time()
        })
        
        return reward
    
    def get_episode_summary(self) -> dict:
        """Get summary of rewards/penalties for the episode"""
        return {
            "total_trades": len(self.trade_history),
            "total_rewards": self.total_rewards,
            "total_penalties": self.total_penalties,
            "net_reward": self.total_rewards - self.total_penalties,
            "current_win_streak": self.win_streak,
            "average_reward_per_trade": (self.total_rewards - self.total_penalties) / max(1, len(self.trade_history))
        }
    
    def reset_episode(self):
        """Reset for new episode"""
        self.trade_history = []
        self.win_streak = 0
        self.total_rewards = 0.0
        self.total_penalties = 0.0

class TradingEnvironment:
    """Enhanced trading environment with 1000 steps per episode"""
    
    def __init__(self, steps_per_episode=1000):
        self.steps_per_episode = steps_per_episode
        self.current_step = 0
        self.current_balance = 100000.0  # Starting capital
        self.initial_balance = 100000.0
        self.positions = []
        self.trade_history = []
        
    def generate_market_data(self, steps: int):
        """Generate realistic market data for training"""
        import numpy as np
        
        # Generate price data with realistic forex characteristics
        returns = np.random.normal(0, 0.001, steps)  # 0.1% daily volatility
        returns[0] = 0  # First return is 0
        
        # Add some trends and volatility clustering
        trend_strength = np.random.normal(0, 0.0005, steps)
        volatility = np.abs(np.random.normal(0.001, 0.0002, steps))
        
        prices = np.cumprod(1 + returns + trend_strength) * 1.2000  # Start at 1.2000 (EUR/USD)
        
        return {
            "prices": prices,
            "returns": returns,
            "volatility": volatility,
            "timestamps": [time.time() + i * 60 for i in range(steps)]  # 1-minute intervals
        }
    
    def execute_trade(self, action: str, size: float, market_data: dict, step: int) -> float:
        """
        Execute a trade and return PnL
        
        Args:
            action: 'buy', 'sell', or 'hold'
            size: Trade size (lots)
            market_data: Market data dict
            step: Current step in episode
        
        Returns:
            float: PnL in USD for this trade
        """
        if action == 'hold' or step >= len(market_data["prices"]) - 1:
            return 0.0
        
        current_price = market_data["prices"][step]
        next_price = market_data["prices"][step + 1]
        
        if action == 'buy':
            pnl = (next_price - current_price) * size * 100000  # Standard lot size
        elif action == 'sell':
            pnl = (current_price - next_price) * size * 100000
        else:
            pnl = 0.0
        
        # Add realistic spread and commission
        spread_cost = abs(size) * 2.0  # 2 pip spread
        commission = abs(size) * 7.0   # $7 per lot commission
        
        pnl -= (spread_cost + commission)
        
        # Update balance
        self.current_balance += pnl
        
        # Store trade
        trade_record = {
            "step": step,
            "action": action,
            "size": size,
            "entry_price": current_price,
            "exit_price": next_price,
            "pnl": pnl,
            "balance": self.current_balance,
            "timestamp": market_data["timestamps"][step]
        }
        self.trade_history.append(trade_record)
        
        return pnl
    
    def get_episode_metrics(self) -> dict:
        """Calculate comprehensive episode metrics"""
        if not self.trade_history:
            return {"total_pnl": 0, "win_rate": 0, "sharpe_ratio": 0, "max_drawdown": 0}
        
        pnls = [trade["pnl"] for trade in self.trade_history]
        balances = [trade["balance"] for trade in self.trade_history]
        
        total_pnl = sum(pnls)
        wins = len([p for p in pnls if p > 0])
        win_rate = wins / len(pnls) if pnls else 0
        
        # Calculate Sharpe ratio
        if len(pnls) > 1:
            returns = np.array(pnls) / self.initial_balance
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)  # Annualized
        else:
            sharpe_ratio = 0
        
        # Calculate max drawdown
        peak_balance = self.initial_balance
        max_drawdown = 0
        for balance in balances:
            if balance > peak_balance:
                peak_balance = balance
            drawdown = (peak_balance - balance) / peak_balance
            max_drawdown = max(max_drawdown, drawdown)
        
        return {
            "total_pnl": total_pnl,
            "win_rate": win_rate,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "total_trades": len(self.trade_history),
            "final_balance": self.current_balance,
            "return_pct": (self.current_balance - self.initial_balance) / self.initial_balance * 100
        }
    
    def reset(self):
        """Reset environment for new episode"""
        self.current_step = 0
        self.current_balance = self.initial_balance
        self.positions = []
        self.trade_history = []

def create_enhanced_trading_bot():
    """Create a simple trading bot for demonstration"""
    
    class TradingBot:
        def __init__(self):
            self.strategy_params = {
                "trend_threshold": np.random.uniform(0.0005, 0.002),
                "volatility_threshold": np.random.uniform(0.001, 0.005),
                "position_size": np.random.uniform(0.1, 1.0),
                "risk_tolerance": np.random.uniform(0.02, 0.1)
            }
            
        def decide_action(self, market_data: dict, step: int) -> tuple:
            """Decide trading action based on market data"""
            if step < 10 or step >= len(market_data["prices"]) - 1:
                return "hold", 0.0
            
            # Simple trend-following strategy
            prices = market_data["prices"]
            recent_prices = prices[step-10:step]
            current_price = prices[step]
            
            # Calculate trend
            trend = (current_price - recent_prices[0]) / recent_prices[0]
            volatility = market_data["volatility"][step]
            
            # Trading logic
            if trend > self.strategy_params["trend_threshold"] and volatility < self.strategy_params["volatility_threshold"]:
                return "buy", self.strategy_params["position_size"]
            elif trend < -self.strategy_params["trend_threshold"] and volatility < self.strategy_params["volatility_threshold"]:
                return "sell", self.strategy_params["position_size"]
            else:
                return "hold", 0.0
        
        def update_strategy(self, episode_reward: float):
            """Update strategy based on episode performance (reinforcement learning)"""
            if episode_reward > 0:
                # Successful episode: slightly enhance current parameters
                for param in self.strategy_params:
                    self.strategy_params[param] *= np.random.uniform(0.98, 1.02)
            else:
                # Poor episode: modify parameters more significantly
                for param in self.strategy_params:
                    self.strategy_params[param] *= np.random.uniform(0.9, 1.1)
            
            # Keep parameters within reasonable bounds
            self.strategy_params["trend_threshold"] = np.clip(self.strategy_params["trend_threshold"], 0.0001, 0.01)
            self.strategy_params["volatility_threshold"] = np.clip(self.strategy_params["volatility_threshold"], 0.0005, 0.02)
            self.strategy_params["position_size"] = np.clip(self.strategy_params["position_size"], 0.01, 2.0)
            self.strategy_params["risk_tolerance"] = np.clip(self.strategy_params["risk_tolerance"], 0.01, 0.5)
    
    return TradingBot()

"""
Enhanced training configuration loaded successfully!
"""

import time
import numpy as np
import sys
import os

print("✅ Enhanced training configuration loaded:")
print(f"📊 Total training scale: {TRAINING_CONFIG['generations']} generations × {TRAINING_CONFIG['episodes_per_generation']} episodes × {TRAINING_CONFIG['steps_per_episode']} steps")
print(f"🎯 Total training steps: {TRAINING_CONFIG['total_training_steps']:,}")
print(f"💰 PnL reward system: $1 USD = 1 reward point with bonuses/penalties")
print(f"🖥️  Ray cluster: {TRAINING_CONFIG['ray_config']['workers_per_generation']} workers per generation")
