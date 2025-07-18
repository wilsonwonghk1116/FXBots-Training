#!/usr/bin/env python3
"""
MASSIVE SCALE FOREX TRAINING CONFIGURATION
==========================================
- 200 generations
- 1000 episodes per generation  
- 1000 steps per episode
- Enhanced PnL-based reward system
- Optimized for dual-PC Ray cluster
"""

import numpy as np
import time
from datetime import datetime
import json
import os

class MassiveScaleConfig:
    """Configuration for massive scale training"""
    
    # MASSIVE SCALE PARAMETERS
    TOTAL_GENERATIONS = 200
    EPISODES_PER_GENERATION = 1000
    STEPS_PER_EPISODE = 1000
    
    # POPULATION PARAMETERS
    POPULATION_SIZE = 100
    ELITE_SIZE = 20
    MUTATION_RATE = 0.15
    
    # TRAINING PARAMETERS
    LEARNING_RATE = 0.001
    DISCOUNT_FACTOR = 0.95
    EXPLORATION_RATE = 0.1
    
    # PNL REWARD SYSTEM
    REWARD_SCALE = 1.0  # $1 USD = 1 reward point
    PENALTY_SCALE = 1.0  # $1 USD loss = 1 penalty point
    MAX_REWARD_CAP = 1000.0  # Cap individual trade rewards
    MAX_PENALTY_CAP = -1000.0  # Cap individual trade penalties
    
    # RAY CLUSTER OPTIMIZATION
    WORKERS_PER_NODE = 8
    CPUS_PER_WORKER = 12
    EPISODES_PER_BATCH = 50  # Process episodes in batches
    
    # PERFORMANCE TRACKING
    SAVE_INTERVAL = 10  # Save every 10 generations
    CHECKPOINT_INTERVAL = 25  # Create checkpoint every 25 generations
    
    @classmethod
    def get_total_steps(cls):
        """Calculate total steps for entire training"""
        return cls.TOTAL_GENERATIONS * cls.EPISODES_PER_GENERATION * cls.STEPS_PER_EPISODE
    
    @classmethod
    def get_estimated_time(cls, steps_per_second=100):
        """Estimate total training time"""
        total_steps = cls.get_total_steps()
        total_seconds = total_steps / steps_per_second
        hours = total_seconds / 3600
        return hours
    
    @classmethod
    def print_config(cls):
        """Print the massive scale configuration"""
        print("=" * 60)
        print("🚀 MASSIVE SCALE FOREX TRAINING CONFIGURATION")
        print("=" * 60)
        print(f"📊 Total Generations: {cls.TOTAL_GENERATIONS:,}")
        print(f"📈 Episodes per Generation: {cls.EPISODES_PER_GENERATION:,}")
        print(f"⚡ Steps per Episode: {cls.STEPS_PER_EPISODE:,}")
        print(f"🎯 Total Steps: {cls.get_total_steps():,}")
        print(f"⏱️  Estimated Time (100 steps/sec): {cls.get_estimated_time():.1f} hours")
        print(f"👥 Population Size: {cls.POPULATION_SIZE}")
        print(f"🏆 Elite Size: {cls.ELITE_SIZE}")
        print(f"💰 PnL Reward Scale: ${cls.REWARD_SCALE}/point")
        print(f"🎛️  Ray Workers per Node: {cls.WORKERS_PER_NODE}")
        print("=" * 60)

class EnhancedPnLRewardSystem:
    """Enhanced PnL-based reward system for reinforcement learning"""
    
    def __init__(self, config=MassiveScaleConfig):
        self.config = config
        self.total_rewards = 0
        self.total_penalties = 0
        self.trade_count = 0
        
    def calculate_reward(self, pnl_usd, trade_type="FOREX"):
        """
        Calculate reward/penalty based on PnL
        
        Args:
            pnl_usd: Profit/Loss in USD
            trade_type: Type of trade (for future expansion)
            
        Returns:
            reward: Numerical reward (positive for profit, negative for loss)
        """
        # Base reward = PnL * scale
        base_reward = pnl_usd * self.config.REWARD_SCALE
        
        # Apply caps to prevent extreme rewards
        if base_reward > 0:
            reward = min(base_reward, self.config.MAX_REWARD_CAP)
            self.total_rewards += reward
        else:
            reward = max(base_reward, self.config.MAX_PENALTY_CAP)
            self.total_penalties += abs(reward)
        
        self.trade_count += 1
        return reward
    
    def get_cumulative_performance(self):
        """Get cumulative performance metrics"""
        return {
            'total_rewards': self.total_rewards,
            'total_penalties': self.total_penalties,
            'net_performance': self.total_rewards - self.total_penalties,
            'trade_count': self.trade_count,
            'avg_reward_per_trade': self.total_rewards / max(self.trade_count, 1),
            'win_rate': self.total_rewards / max(self.total_rewards + self.total_penalties, 1)
        }

class MassiveScaleEpisodeProcessor:
    """Process episodes at massive scale with PnL rewards"""
    
    def __init__(self, bot_id, config=MassiveScaleConfig):
        self.bot_id = bot_id
        self.config = config
        self.reward_system = EnhancedPnLRewardSystem(config)
        self.episode_results = []
        
    def simulate_forex_trade(self, step_data):
        """
        Simulate a single forex trade step
        
        Args:
            step_data: Dictionary containing market data for this step
            
        Returns:
            trade_result: Dictionary with PnL and trade details
        """
        # Simulate realistic forex trading
        base_pnl = np.random.normal(0, 50)  # Random PnL with $50 std dev
        
        # Add some skill-based component (bot learning effect)
        skill_factor = min(self.trade_count * 0.001, 0.5)  # Max 50% skill
        skilled_pnl = base_pnl + (skill_factor * abs(base_pnl) * np.random.choice([-1, 1]))
        
        return {
            'pnl_usd': skilled_pnl,
            'trade_type': 'FOREX',
            'timestamp': time.time(),
            'step': step_data.get('step', 0)
        }
    
    def run_episode(self, episode_id, generation_id):
        """
        Run a single episode with 1000 steps
        
        Args:
            episode_id: Unique episode identifier
            generation_id: Current generation number
            
        Returns:
            episode_result: Complete episode results with rewards
        """
        episode_start = time.time()
        episode_rewards = []
        total_pnl = 0
        
        # Run 1000 steps per episode
        for step in range(self.config.STEPS_PER_EPISODE):
            # Simulate market data for this step
            step_data = {
                'step': step,
                'episode_id': episode_id,
                'generation_id': generation_id,
                'market_price': 1.0 + np.random.normal(0, 0.01),  # Simulated price
                'volatility': abs(np.random.normal(0.02, 0.005))  # Simulated volatility
            }
            
            # Execute trade
            trade_result = self.simulate_forex_trade(step_data)
            
            # Calculate reward
            reward = self.reward_system.calculate_reward(trade_result['pnl_usd'])
            episode_rewards.append(reward)
            total_pnl += trade_result['pnl_usd']
        
        episode_duration = time.time() - episode_start
        
        # Compile episode results
        episode_result = {
            'episode_id': episode_id,
            'generation_id': generation_id,
            'bot_id': self.bot_id,
            'total_pnl_usd': total_pnl,
            'total_reward': sum(episode_rewards),
            'steps_completed': len(episode_rewards),
            'duration_seconds': episode_duration,
            'average_reward_per_step': np.mean(episode_rewards),
            'max_reward': max(episode_rewards),
            'min_reward': min(episode_rewards),
            'performance_metrics': self.reward_system.get_cumulative_performance(),
            'timestamp': datetime.now().isoformat()
        }
        
        self.episode_results.append(episode_result)
        return episode_result
    
    @property
    def trade_count(self):
        """Get total trade count"""
        return self.reward_system.trade_count

def create_massive_scale_test():
    """Create a test to verify the massive scale configuration"""
    
    print("🧪 TESTING MASSIVE SCALE CONFIGURATION")
    print("=" * 60)
    
    # Print configuration
    MassiveScaleConfig.print_config()
    
    # Test episode processor
    print("\n🔬 Testing Episode Processor...")
    processor = MassiveScaleEpisodeProcessor(bot_id="TEST_BOT_001")
    
    # Run a single episode test
    start_time = time.time()
    test_episode = processor.run_episode(episode_id=1, generation_id=1)
    test_duration = time.time() - start_time
    
    print(f"✅ Test Episode Completed:")
    print(f"   Duration: {test_duration:.2f} seconds")
    print(f"   Total PnL: ${test_episode['total_pnl_usd']:.2f}")
    print(f"   Total Reward: {test_episode['total_reward']:.2f}")
    print(f"   Steps: {test_episode['steps_completed']}")
    print(f"   Avg Reward/Step: {test_episode['average_reward_per_step']:.3f}")
    
    # Estimate full scale performance
    steps_per_second = 1000 / test_duration
    estimated_hours = MassiveScaleConfig.get_estimated_time(steps_per_second)
    
    print(f"\n📊 PERFORMANCE ESTIMATION:")
    print(f"   Steps per Second: {steps_per_second:.1f}")
    print(f"   Estimated Total Time: {estimated_hours:.1f} hours")
    print(f"   Estimated Total Time: {estimated_hours/24:.1f} days")
    
    # Test reward system
    print(f"\n💰 REWARD SYSTEM TEST:")
    perf = test_episode['performance_metrics']
    print(f"   Total Rewards: {perf['total_rewards']:.2f}")
    print(f"   Total Penalties: {perf['total_penalties']:.2f}")
    print(f"   Net Performance: {perf['net_performance']:.2f}")
    print(f"   Win Rate: {perf['win_rate']:.1%}")
    
    return test_episode

if __name__ == "__main__":
    # Run the massive scale test
    test_result = create_massive_scale_test()
    
    print("\n🎯 MASSIVE SCALE CONFIGURATION READY!")
    print("This configuration can now be integrated with your Ray cluster.")
