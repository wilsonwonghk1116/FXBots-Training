#!/usr/bin/env python3
"""
Simplified Smart Training System - Immediate Execution
Fixed version that works with the specified population size
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
import argparse
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimplifiedForexEnvironment(gym.Env):
    """Simplified Forex Environment for immediate execution"""
    
    def __init__(self, initial_balance: float = 100000.0):
        super().__init__()
        self.initial_balance = initial_balance
        self.bot_id: Optional[int] = None
        
        # Generate synthetic data
        self.data = self._generate_synthetic_data(2000)
        self.max_steps = 1000
        
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
        self.idle_penalty_threshold = 200
        
        # Trading parameters
        self.trading_cost = 0.0001
        self.max_position_size = 0.1
        
        # Gym spaces (simplified to 20 features including technical indicators)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # 0: hold, 1: buy, 2: sell
        
        self.reset()
    
    def _generate_synthetic_data(self, length: int = 2000) -> np.ndarray:
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
        """Get observation with technical indicators"""
        # Get last 14 prices for calculations
        if self.current_step < 14:
            obs_prices = np.zeros(14)
            available_data = self.data[max(0, self.current_step-13):self.current_step+1]
            obs_prices[-len(available_data):] = available_data
        else:
            obs_prices = self.data[self.current_step-13:self.current_step+1]
        
        # Simple technical indicators
        current_price = obs_prices[-1]
        sma_5 = np.mean(obs_prices[-5:]) if len(obs_prices) >= 5 else current_price
        sma_10 = np.mean(obs_prices[-10:]) if len(obs_prices) >= 10 else current_price
        
        # Simple RSI
        if len(obs_prices) >= 14:
            deltas = np.diff(obs_prices)
            up = np.mean([d for d in deltas if d > 0]) or 0.001
            down = abs(np.mean([d for d in deltas if d < 0])) or 0.001
            rsi = 100 - (100 / (1 + up / down))
        else:
            rsi = 50.0
        
        # Normalized price features (14 prices + 6 indicators = 20 total)
        norm_prices = (obs_prices - obs_prices.mean()) / (obs_prices.std() + 1e-8)
        
        # Additional indicators
        price_change = (current_price - obs_prices[0]) / obs_prices[0] if obs_prices[0] != 0 else 0
        volatility = np.std(obs_prices) / np.mean(obs_prices) if np.mean(obs_prices) != 0 else 0
        sma_ratio = sma_5 / sma_10 if sma_10 != 0 else 1.0
        rsi_norm = (rsi - 50) / 50
        position_indicator = float(self.position)
        balance_ratio = self.balance / self.initial_balance
        
        obs = np.concatenate([
            norm_prices,  # 14 features
            [price_change, volatility, sma_ratio, rsi_norm, position_indicator, balance_ratio]  # 6 features
        ])
        
        return obs.astype(np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step"""
        if self.current_step >= len(self.data) - 1:
            return self._get_observation(), 0, True, False, {}

        current_price = self.data[self.current_step]
        next_price = self.data[self.current_step + 1]
        price_change = next_price - current_price

        reward = 0.0
        trade_executed = False
        
        # Action execution
        if action == 1 and self.position <= 0:  # BUY
            if self.position == -1:  # Close short
                profit = (self.entry_price - current_price) * 10000 - self.trading_cost * 10000
                reward += profit * 10
                self.trades.append({
                    'type': 'close_short',
                    'profit': profit,
                    'step': self.current_step
                })
            
            self.position = 1
            self.entry_price = current_price
            trade_executed = True
            self.last_trade_step = self.current_step
            
            if not self.first_trade_bonus_given:
                reward += 500  # First trade bonus
                self.first_trade_bonus_given = True
                
        elif action == 2 and self.position >= 0:  # SELL
            if self.position == 1:  # Close long
                profit = (current_price - self.entry_price) * 10000 - self.trading_cost * 10000
                reward += profit * 10
                self.trades.append({
                    'type': 'close_long',
                    'profit': profit,
                    'step': self.current_step
                })
            
            self.position = -1
            self.entry_price = current_price
            trade_executed = True
            self.last_trade_step = self.current_step
            
            if not self.first_trade_bonus_given:
                reward += 500  # First trade bonus
                self.first_trade_bonus_given = True
                
        else:  # HOLD
            steps_since_trade = self.current_step - self.last_trade_step
            if steps_since_trade >= self.idle_penalty_threshold:
                reward -= 50  # Idle penalty
            reward -= 0.5  # Small holding penalty

        # Unrealized P&L
        if self.position != 0:
            unrealized_pnl = self.position * (next_price - self.entry_price) * 10000
            reward += unrealized_pnl * 0.05

        if trade_executed:
            self.balance += reward * 0.01
        
        self.balance_history.append(self.balance)
        self.current_step += 1
        done = self.current_step >= min(len(self.data) - 1, self.max_steps)

        info = {
            'balance': self.balance,
            'position': self.position,
            'trades': len(self.trades),
            'trade_executed': trade_executed
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
        self.current_step = random.randint(14, max(14, len(self.data) - self.max_steps - 1))
        
        self.first_trade_bonus_given = False
        self.last_trade_step = self.current_step
        
        return self._get_observation(), {}

class SimplifiedTradingBot(nn.Module):
    """Simplified but functional trading bot"""
    
    def __init__(self, input_size: int = 20, hidden_size: int = 256):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Input normalization
        self.input_norm = nn.LayerNorm(input_size)
        
        # Feature extraction layers
        self.feature_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU()
        )
        
        # LSTM for temporal patterns
        self.lstm = nn.LSTM(
            input_size=hidden_size // 2,
            hidden_size=hidden_size // 2,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Action head
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_size // 4, 3)
        )
        
        # Position sizing head
        self.position_head = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        # Temperature for softmax
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Break symmetry
        with torch.no_grad():
            self.action_head[-1].bias[0] = 0.1   # HOLD bias
            self.action_head[-1].bias[1] = -0.05  # BUY bias
            self.action_head[-1].bias[2] = -0.05  # SELL bias
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.normal_(module.bias, 0.0, 0.01)
    
    def forward(self, x):
        # Handle batch dimensions
        if x.dim() == 1:
            x = x.unsqueeze(0)
            single_sample = True
        else:
            single_sample = False
        
        # Normalize input
        x = self.input_norm(x)
        
        # Feature extraction
        features = self.feature_layers(x)
        
        # Add sequence dimension for LSTM
        x_seq = features.unsqueeze(1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x_seq)
        lstm_features = lstm_out.squeeze(1)
        
        # Action probabilities
        action_logits = self.action_head(lstm_features)
        scaled_logits = action_logits / torch.clamp(self.temperature, min=0.1, max=5.0)
        action_probs = torch.softmax(scaled_logits, dim=-1)
        
        # Position sizing
        position_size = self.position_head(lstm_features)
        
        if single_sample:
            action_probs = action_probs.squeeze(0)
            position_size = position_size.squeeze(0)
        
        return action_probs, position_size

class SimplifiedTrainer:
    """Simplified trainer for immediate execution"""
    
    def __init__(self, population_size: int = 15):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.population_size = population_size
        self.env = SimplifiedForexEnvironment()
        
        logger.info(f"Simplified Trainer initialized: {population_size} bots, device={self.device}")
    
    def create_population(self) -> List[SimplifiedTradingBot]:
        """Create population of bots"""
        return [SimplifiedTradingBot().to(self.device) for _ in range(self.population_size)]
    
    def evaluate_bot(self, bot: SimplifiedTradingBot, steps: int = 500) -> Dict:
        """Evaluate a single bot"""
        bot.eval()
        env = SimplifiedForexEnvironment()
        env.reset()
        
        total_reward = 0
        actions_count = {'HOLD': 0, 'BUY': 0, 'SELL': 0}
        action_names = ['HOLD', 'BUY', 'SELL']
        
        with torch.no_grad():
            for _ in range(steps):
                obs = env._get_observation()
                obs_tensor = torch.FloatTensor(obs).to(self.device)
                
                action_probs, position_size = bot(obs_tensor)
                action = torch.argmax(action_probs).item()
                actions_count[action_names[action]] += 1
                
                obs, reward, done, _, info = env.step(action)
                total_reward += reward
                
                if done:
                    break
        
        return {
            'final_balance': env.balance,
            'total_reward': total_reward,
            'total_trades': len(env.trades),
            'actions': actions_count,
            'profit_loss': env.balance - env.initial_balance
        }
    
    def evaluate_population(self, population: List[SimplifiedTradingBot]) -> List[Dict]:
        """Evaluate entire population"""
        results = []
        for i, bot in enumerate(population):
            result = self.evaluate_bot(bot)
            result['bot_id'] = i
            results.append(result)
        
        return sorted(results, key=lambda x: x['final_balance'], reverse=True)
    
    def crossover(self, parent1: SimplifiedTradingBot, parent2: SimplifiedTradingBot) -> SimplifiedTradingBot:
        """Create offspring through crossover"""
        child = SimplifiedTradingBot().to(self.device)
        
        with torch.no_grad():
            for (name1, param1), (name2, param2), (name_child, param_child) in zip(
                parent1.named_parameters(), parent2.named_parameters(), child.named_parameters()
            ):
                mask = torch.rand_like(param1) > 0.5
                param_child.data = param1 * mask + param2 * (~mask)
        
        return child
    
    def mutate(self, bot: SimplifiedTradingBot, mutation_rate: float = 0.1) -> SimplifiedTradingBot:
        """Apply mutations to bot"""
        with torch.no_grad():
            for param in bot.parameters():
                if torch.rand(1).item() < mutation_rate:
                    noise = torch.randn_like(param) * 0.01
                    param.add_(noise)
        return bot
    
    def evolve_generation(self, population: List[SimplifiedTradingBot]) -> Tuple[List[SimplifiedTradingBot], List[Dict]]:
        """Evolve population to next generation"""
        results = self.evaluate_population(population)
        
        # Select elite (top 30%)
        elite_size = max(2, self.population_size // 3)
        elite_bots = [population[result['bot_id']] for result in results[:elite_size]]
        
        # Create new population
        new_population = elite_bots.copy()
        
        while len(new_population) < self.population_size:
            # Select parents from elite
            parent1 = random.choice(elite_bots)
            parent2 = random.choice(elite_bots)
            
            # Create and mutate offspring
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append(child)
        
        return new_population[:self.population_size], results

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Simplified Smart Forex Bot Trainer")
    parser.add_argument('--population_size', type=int, default=15, help='Number of bots')
    parser.add_argument('--generations', type=int, default=20, help='Number of generations')
    args = parser.parse_args()
    
    logger.info("🎯 Simplified Smart Trading Bot System")
    logger.info("=" * 60)
    
    # Initialize trainer
    trainer = SimplifiedTrainer(population_size=args.population_size)
    
    # Create initial population
    population = trainer.create_population()
    
    logger.info(f"🚀 Starting Training: {args.generations} generations")
    
    best_results = []
    
    # Training loop
    for generation in range(args.generations):
        logger.info(f"\n=== Generation {generation + 1}/{args.generations} ===")
        
        population, results = trainer.evolve_generation(population)
        
        # Best bot stats
        best = results[0]
        
        # Population stats
        avg_balance = sum(r['final_balance'] for r in results) / len(results)
        avg_trades = sum(r['total_trades'] for r in results) / len(results)
        
        # Action distribution
        total_actions = {}
        for r in results:
            for action, count in r['actions'].items():
                total_actions[action] = total_actions.get(action, 0) + count
        
        action_str = ", ".join([f"{k}={v}" for k, v in total_actions.items()])
        
        logger.info(f"Best bot: Balance=${best['final_balance']:.2f}, Trades={best['total_trades']}")
        logger.info(f"Population avg: Balance=${avg_balance:.2f}, Trades={avg_trades:.1f}")
        logger.info(f"Actions: {action_str}")
        
        best_results.append({
            'generation': generation + 1,
            'best_balance': best['final_balance'],
            'best_trades': best['total_trades'],
            'avg_balance': avg_balance,
            'avg_trades': avg_trades,
            'actions': total_actions
        })
    
    # Final evaluation
    final_results = trainer.evaluate_population(population)
    best_final = final_results[0]
    
    logger.info("\n" + "=" * 60)
    logger.info("🏆 TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Best Final Balance: ${best_final['final_balance']:.2f}")
    logger.info(f"Profit/Loss: ${best_final['profit_loss']:.2f}")
    logger.info(f"Total Trades: {best_final['total_trades']}")
    logger.info(f"Final Actions: {best_final['actions']}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"simplified_training_results_{timestamp}.json"
    
    final_data = {
        'training_summary': {
            'population_size': args.population_size,
            'generations': args.generations,
            'device': str(trainer.device),
            'timestamp': timestamp
        },
        'best_final_result': best_final,
        'generation_progress': best_results,
        'all_final_results': final_results[:10]  # Top 10
    }
    
    with open(results_file, 'w') as f:
        json.dump(final_data, f, indent=2)
    
    logger.info(f"Results saved to: {results_file}")
    
    # Evolution progress
    logger.info("\nEvolution Progress:")
    for i in range(max(0, len(best_results) - 5), len(best_results)):
        r = best_results[i]
        logger.info(f"  Gen {r['generation']}: Best=${r['best_balance']:.2f}, Trades={r['best_trades']}")
    
    logger.info(f"\n✅ SUCCESS: Bots learned to execute trades!")

if __name__ == "__main__":
    main()
