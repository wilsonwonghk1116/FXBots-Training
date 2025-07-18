#!/usr/bin/env python3
"""
Quick Start Script for Enhanced Smart Trading Bots
Simplified version to avoid common issues and run immediately
"""

import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import gymnasium as gym
from gymnasium import spaces
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QuickForexEnvironment(gym.Env):
    """Simplified Forex Environment for immediate execution"""
    
    def __init__(self, initial_balance: float = 100000.0):
        super().__init__()
        self.initial_balance = initial_balance
        self.bot_id: Optional[int] = None
        
        # Generate synthetic data
        self.data = self._generate_synthetic_data(5000)
        self.max_steps = 500  # Shorter for faster execution
        
        # Trading state
        self.balance = initial_balance
        self.position = 0
        self.entry_price = None
        self.trades = []
        self.balance_history = []
        self.current_step = 0
        
        # Reward tracking
        self.first_trade_bonus_given = False
        self.last_trade_step = 0
        
        # Trading parameters
        self.trading_cost = 0.0001
        self.max_position_size = 0.1
        
        # Gym spaces (simplified to 10 features)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # 0: hold, 1: buy, 2: sell
        
        self.reset()
    
    def _generate_synthetic_data(self, length: int = 5000) -> np.ndarray:
        """Generate synthetic EUR/USD-like data"""
        np.random.seed(42)
        start_price = 1.1000
        prices = [start_price]
        
        for i in range(length - 1):
            change = np.random.normal(0, 0.001)  # Higher volatility for more action
            trend = 0.00001 * np.sin(i / 50)
            new_price = prices[-1] + change + trend
            new_price = max(0.9000, min(1.3000, new_price))
            prices.append(new_price)
        
        return np.array(prices)
    
    def _get_observation(self) -> np.ndarray:
        """Get simplified observation"""
        # Get last 10 prices
        if self.current_step < 10:
            obs_prices = np.zeros(10)
            available_data = self.data[max(0, self.current_step-9):self.current_step+1]
            obs_prices[-len(available_data):] = available_data
        else:
            obs_prices = self.data[self.current_step-9:self.current_step+1]
        
        # Simple normalization
        if len(obs_prices) > 1:
            norm_prices = (obs_prices - obs_prices.mean()) / (obs_prices.std() + 1e-8)
        else:
            norm_prices = obs_prices
        
        return norm_prices.astype(np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step"""
        if self.current_step >= len(self.data) - 1:
            return self._get_observation(), 0, True, False, {}

        current_price = self.data[self.current_step]
        next_price = self.data[self.current_step + 1]
        price_change = next_price - current_price

        reward = 0.0
        trade_executed = False
        direction = "HOLD"

        # Simplified trading logic
        if action == 1 and self.position <= 0:  # BUY
            direction = "BUY"
            if self.position == -1:  # Close short
                profit = -(current_price - self.entry_price) * 10000
                reward += profit * 5
                self.trades.append({'type': 'close_short', 'profit': profit})
            
            self.position = 1
            self.entry_price = current_price
            trade_executed = True
            self.last_trade_step = self.current_step
            
            # First trade bonus
            if not self.first_trade_bonus_given:
                reward += 500
                self.first_trade_bonus_given = True
                
        elif action == 2 and self.position >= 0:  # SELL
            direction = "SELL"
            if self.position == 1:  # Close long
                profit = (current_price - self.entry_price) * 10000
                reward += profit * 5
                self.trades.append({'type': 'close_long', 'profit': profit})
            
            self.position = -1
            self.entry_price = current_price
            trade_executed = True
            self.last_trade_step = self.current_step
            
            # First trade bonus
            if not self.first_trade_bonus_given:
                reward += 500
                self.first_trade_bonus_given = True
        
        else:  # HOLD
            # Small holding penalty
            reward -= 0.5
            
            # Idle penalty
            if self.current_step - self.last_trade_step > 100:
                reward -= 25

        # Unrealized P&L
        if self.position != 0:
            unrealized_pnl = self.position * (next_price - self.entry_price) * 10000
            reward += unrealized_pnl * 0.02

        self.balance += reward * 0.01
        self.balance_history.append(self.balance)
        self.current_step += 1
        done = self.current_step >= min(len(self.data) - 1, self.max_steps)

        info = {
            'direction': direction,
            'balance': self.balance,
            'position': self.position,
            'trade_executed': trade_executed,
            'total_trades': len(self.trades)
        }

        return self._get_observation(), reward, done, False, info
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment"""
        if seed:
            np.random.seed(seed)
            random.seed(seed)
        
        self.balance = self.initial_balance
        self.position = 0
        self.trades = []
        self.balance_history = [self.initial_balance]
        self.current_step = random.randint(10, len(self.data) - self.max_steps - 1)
        
        # Reset tracking
        self.first_trade_bonus_given = False
        self.last_trade_step = self.current_step
        
        return self._get_observation(), {}

class QuickTradingBot(nn.Module):
    """Simplified but functional trading bot"""
    
    def __init__(self, input_size: int = 10, hidden_size: int = 128, output_size: int = 3):
        super().__init__()
        
        # Simple but effective architecture
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        # Temperature for action selection
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Break symmetry
        with torch.no_grad():
            self.network[-1].bias[0] = 0.1   # HOLD bias
            self.network[-1].bias[1] = -0.05  # BUY bias
            self.network[-1].bias[2] = -0.05  # SELL bias
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.normal_(module.bias, 0.0, 0.01)
    
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
            single_sample = True
        else:
            single_sample = False
        
        # Forward pass
        logits = self.network(x)
        
        # Apply temperature scaling
        scaled_logits = logits / torch.clamp(self.temperature, min=0.1, max=5.0)
        action_probs = torch.softmax(scaled_logits, dim=-1)
        
        if single_sample:
            action_probs = action_probs.squeeze(0)
        
        return action_probs

class QuickTrainer:
    """Simplified trainer for immediate execution"""
    
    def __init__(self, population_size: int = 15):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.population_size = population_size
        self.env = QuickForexEnvironment()
        
        logger.info(f"Quick Trainer initialized: {population_size} bots, device={self.device}")
    
    def create_population(self) -> List[QuickTradingBot]:
        """Create population"""
        return [QuickTradingBot().to(self.device) for _ in range(self.population_size)]
    
    def evaluate_bot(self, bot: QuickTradingBot, episodes: int = 3) -> Dict:
        """Evaluate single bot"""
        bot.eval()
        total_metrics = {
            'total_reward': 0,
            'total_trades': 0,
            'final_balance': 0,
            'actions_taken': {'HOLD': 0, 'BUY': 0, 'SELL': 0}
        }
        
        for episode in range(episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            
            for step in range(500):
                obs_tensor = torch.FloatTensor(obs).to(self.device)
                
                with torch.no_grad():
                    action_probs = bot(obs_tensor)
                    action = torch.argmax(action_probs).item()
                
                obs, reward, done, _, info = self.env.step(action)
                episode_reward += reward
                
                # Track actions
                action_names = ['HOLD', 'BUY', 'SELL']
                total_metrics['actions_taken'][action_names[action]] += 1
                
                if done:
                    break
            
            total_metrics['total_reward'] += episode_reward
            total_metrics['total_trades'] += len(self.env.trades)
            total_metrics['final_balance'] += self.env.balance
        
        # Average over episodes
        for key in ['total_reward', 'final_balance']:
            total_metrics[key] /= episodes
        
        return total_metrics
    
    def evaluate_population(self, population: List[QuickTradingBot]) -> List[Dict]:
        """Evaluate entire population"""
        results = []
        
        for i, bot in enumerate(population):
            metrics = self.evaluate_bot(bot)
            metrics['bot_id'] = i
            results.append(metrics)
        
        return sorted(results, key=lambda x: x['final_balance'], reverse=True)
    
    def mutate(self, bot: QuickTradingBot, mutation_rate: float = 0.1) -> QuickTradingBot:
        """Simple mutation"""
        new_bot = QuickTradingBot().to(self.device)
        new_bot.load_state_dict(bot.state_dict())
        
        with torch.no_grad():
            for param in new_bot.parameters():
                if torch.rand(1).item() < mutation_rate:
                    noise = torch.randn_like(param) * 0.02
                    param.add_(noise)
        
        return new_bot
    
    def crossover(self, parent1: QuickTradingBot, parent2: QuickTradingBot) -> QuickTradingBot:
        """Simple crossover"""
        child = QuickTradingBot().to(self.device)
        
        with torch.no_grad():
            for (name1, param1), (name2, param2), (name_child, param_child) in zip(
                parent1.named_parameters(), parent2.named_parameters(), child.named_parameters()
            ):
                mask = torch.rand_like(param1) > 0.5
                param_child.data = param1 * mask + param2 * (~mask)
        
        return child
    
    def train(self, generations: int = 15) -> Dict:
        """Training loop"""
        logger.info(f"🚀 Starting Quick Training: {generations} generations")
        
        # Create initial population
        population = self.create_population()
        
        # Track best results
        generation_stats = []
        
        for generation in range(generations):
            logger.info(f"\n=== Generation {generation + 1}/{generations} ===")
            
            # Evaluate population
            results = self.evaluate_population(population)
            best_bot_metrics = results[0]
            
            # Log generation stats
            avg_balance = np.mean([r['final_balance'] for r in results])
            avg_trades = np.mean([r['total_trades'] for r in results])
            
            # Count total actions
            total_actions = {'HOLD': 0, 'BUY': 0, 'SELL': 0}
            for r in results:
                for action, count in r['actions_taken'].items():
                    total_actions[action] += count
            
            logger.info(f"Best bot: Balance=${best_bot_metrics['final_balance']:.2f}, "
                       f"Trades={best_bot_metrics['total_trades']}")
            logger.info(f"Population avg: Balance=${avg_balance:.2f}, Trades={avg_trades:.1f}")
            logger.info(f"Actions: HOLD={total_actions['HOLD']}, "
                       f"BUY={total_actions['BUY']}, SELL={total_actions['SELL']}")
            
            generation_stats.append({
                'generation': generation + 1,
                'best_balance': best_bot_metrics['final_balance'],
                'avg_balance': avg_balance,
                'best_trades': best_bot_metrics['total_trades'],
                'actions': total_actions
            })
            
            # Evolution for next generation
            if generation < generations - 1:
                elite_size = max(3, self.population_size // 4)
                elite = [population[r['bot_id']] for r in results[:elite_size]]
                
                new_population = elite.copy()
                
                while len(new_population) < self.population_size:
                    if len(elite) >= 2:
                        parent1 = random.choice(elite)
                        parent2 = random.choice(elite)
                        child = self.crossover(parent1, parent2)
                        child = self.mutate(child)
                        new_population.append(child)
                    else:
                        parent = random.choice(elite)
                        child = self.mutate(parent)
                        new_population.append(child)
                
                population = new_population[:self.population_size]
        
        # Final results
        final_results = self.evaluate_population(population)
        
        return {
            'final_results': final_results[0],
            'generation_stats': generation_stats,
            'summary': {
                'best_final_balance': final_results[0]['final_balance'],
                'improvement': final_results[0]['final_balance'] - 100000,
                'total_generations': generations,
                'timestamp': datetime.now().isoformat()
            }
        }

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick Smart Trading Bot Trainer")
    parser.add_argument('--population_size', type=int, default=15, help='Population size')
    parser.add_argument('--generations', type=int, default=15, help='Number of generations')
    args = parser.parse_args()
    
    logger.info("🎯 Quick Smart Trading Bot System")
    logger.info("="*60)
    
    # Create trainer
    trainer = QuickTrainer(population_size=args.population_size)
    
    # Run training
    results = trainer.train(generations=args.generations)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"quick_training_results_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("🏆 TRAINING COMPLETE")
    logger.info("="*60)
    
    summary = results['summary']
    final_result = results['final_results']
    
    logger.info(f"Best Final Balance: ${summary['best_final_balance']:.2f}")
    logger.info(f"Profit/Loss: ${summary['improvement']:.2f}")
    logger.info(f"Total Trades: {final_result['total_trades']}")
    logger.info(f"Final Actions: {final_result['actions_taken']}")
    logger.info(f"Results saved to: {filename}")
    
    # Show evolution progress
    logger.info(f"\nEvolution Progress:")
    for stat in results['generation_stats'][-5:]:  # Last 5 generations
        logger.info(f"  Gen {stat['generation']}: Best=${stat['best_balance']:.2f}, "
                   f"Trades={stat['best_trades']}")
    
    if final_result['total_trades'] > 0:
        logger.info("\n✅ SUCCESS: Bots learned to execute trades!")
    else:
        logger.info("\n⚠️  No trades executed - may need parameter tuning")
    
    return results

if __name__ == "__main__":
    results = main()
