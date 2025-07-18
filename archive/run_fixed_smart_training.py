#!/usr/bin/env python3
"""
Fixed Smart Training System - Forces Action Exploration
This version ensures bots learn to trade by adding proper exploration mechanisms
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

class FixedForexEnvironment(gym.Env):
    """Fixed Forex Environment that encourages trading"""
    
    def __init__(self, initial_balance: float = 100000.0):
        super().__init__()
        self.initial_balance = initial_balance
        self.bot_id: Optional[int] = None
        
        # Generate synthetic data with more volatility
        self.data = self._generate_synthetic_data(2000)
        self.max_steps = 500  # Shorter episodes for faster feedback
        
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
        self.consecutive_holds = 0
        
        # Trading parameters
        self.trading_cost = 0.00005  # Very low cost to encourage trading
        self.max_position_size = 0.2
        
        # Gym spaces (simplified to 15 features)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # 0: hold, 1: buy, 2: sell
        
        self.reset()
    
    def _generate_synthetic_data(self, length: int = 2000) -> np.ndarray:
        """Generate synthetic EUR/USD-like data with clear trends"""
        np.random.seed(42)
        start_price = 1.1000
        prices = [start_price]
        
        for i in range(length - 1):
            # Higher volatility and clear trends
            trend = 0.0001 * np.sin(i / 30)  # Shorter cycles
            change = np.random.normal(0, 0.002)  # Higher volatility
            momentum = 0.00005 * (1 if i % 100 < 50 else -1)  # Alternating trends
            
            new_price = prices[-1] + change + trend + momentum
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
        
        current_price = obs_prices[-1]
        
        # Simple indicators
        sma_short = np.mean(obs_prices[-3:]) if len(obs_prices) >= 3 else current_price
        sma_long = np.mean(obs_prices[-7:]) if len(obs_prices) >= 7 else current_price
        
        # Price changes
        price_change_1 = (obs_prices[-1] - obs_prices[-2]) / obs_prices[-2] if len(obs_prices) >= 2 else 0
        price_change_3 = (obs_prices[-1] - obs_prices[-4]) / obs_prices[-4] if len(obs_prices) >= 4 else 0
        
        # Trend indicators
        trend = 1.0 if sma_short > sma_long else -1.0
        volatility = np.std(obs_prices) / np.mean(obs_prices) if np.mean(obs_prices) != 0 else 0
        
        # Position and trading info
        position_indicator = float(self.position)
        balance_ratio = self.balance / self.initial_balance
        consecutive_holds_norm = min(self.consecutive_holds / 50.0, 1.0)  # Normalize to [0,1]
        
        # Normalized price sequence (10 features)
        norm_prices = (obs_prices - obs_prices.mean()) / (obs_prices.std() + 1e-8)
        
        # Additional features (5 features)
        additional_features = [
            price_change_1, price_change_3, trend, volatility, position_indicator
        ]
        
        obs = np.concatenate([norm_prices, additional_features])
        return obs.astype(np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step with enhanced reward system"""
        if self.current_step >= len(self.data) - 1:
            return self._get_observation(), 0, True, False, {}

        current_price = self.data[self.current_step]
        next_price = self.data[self.current_step + 1]
        price_change = next_price - current_price

        reward = 0.0
        trade_executed = False
        
        # Track consecutive holds
        if action == 0:  # HOLD
            self.consecutive_holds += 1
        else:
            self.consecutive_holds = 0
        
        # STRONG penalty for excessive holding
        if self.consecutive_holds > 20:
            reward -= 10.0  # Strong penalty
        elif self.consecutive_holds > 10:
            reward -= 2.0
        elif action == 0:
            reward -= 0.1  # Small penalty for each hold
        
        # Action execution with better rewards
        if action == 1 and self.position <= 0:  # BUY
            if self.position == -1:  # Close short
                profit_pips = (self.entry_price - current_price) * 10000
                reward += profit_pips * 2.0  # Amplified reward
                self.trades.append({'type': 'close_short', 'profit': profit_pips, 'step': self.current_step})
            
            self.position = 1
            self.entry_price = current_price
            trade_executed = True
            self.last_trade_step = self.current_step
            
            # BIG bonus for any trade execution
            reward += 50.0  # Large immediate reward for trading
            
            if not self.first_trade_bonus_given:
                reward += 200.0  # Huge first trade bonus
                self.first_trade_bonus_given = True
                logger.info(f"FIRST TRADE EXECUTED! BUY at step {self.current_step}")
                
        elif action == 2 and self.position >= 0:  # SELL
            if self.position == 1:  # Close long
                profit_pips = (current_price - self.entry_price) * 10000
                reward += profit_pips * 2.0  # Amplified reward
                self.trades.append({'type': 'close_long', 'profit': profit_pips, 'step': self.current_step})
            
            self.position = -1
            self.entry_price = current_price
            trade_executed = True
            self.last_trade_step = self.current_step
            
            # BIG bonus for any trade execution
            reward += 50.0  # Large immediate reward for trading
            
            if not self.first_trade_bonus_given:
                reward += 200.0  # Huge first trade bonus
                self.first_trade_bonus_given = True
                logger.info(f"FIRST TRADE EXECUTED! SELL at step {self.current_step}")

        # Unrealized P&L with higher weight
        if self.position != 0 and self.entry_price is not None:
            unrealized_pips = self.position * (next_price - self.entry_price) * 10000
            reward += unrealized_pips * 0.5  # Higher weight for unrealized gains

        # Update balance based on total reward
        if abs(reward) > 0.1:  # Only update for significant rewards
            self.balance += reward * 0.1
        
        self.balance_history.append(self.balance)
        self.current_step += 1
        done = self.current_step >= min(len(self.data) - 1, self.max_steps)

        info = {
            'balance': self.balance,
            'position': self.position,
            'trades': len(self.trades),
            'trade_executed': trade_executed,
            'consecutive_holds': self.consecutive_holds,
            'action': action
        }

        return self._get_observation(), reward, done, False, info
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment"""
        if seed:
            np.random.seed(seed)
            random.seed(seed)
        
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = None
        self.trades = []
        self.balance_history = [self.initial_balance]
        self.current_step = random.randint(10, max(10, len(self.data) - self.max_steps - 1))
        
        self.first_trade_bonus_given = False
        self.last_trade_step = self.current_step
        self.consecutive_holds = 0
        
        return self._get_observation(), {}

class ExplorationTradingBot(nn.Module):
    """Trading bot with built-in exploration mechanisms"""
    
    def __init__(self, input_size: int = 15, hidden_size: int = 128):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Input processing
        self.input_norm = nn.LayerNorm(input_size)
        
        # Feature extraction with dropout for exploration
        self.features = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),  # Higher dropout for exploration
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU()
        )
        
        # Action head with exploration bias
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, 3)
        )
        
        # Exploration parameters
        self.exploration_noise = nn.Parameter(torch.tensor(0.1))
        self.action_bias = nn.Parameter(torch.tensor([0.0, 0.5, 0.5]))  # Bias toward BUY/SELL
        
        # Initialize with exploration-friendly weights
        self.apply(self._init_weights)
        
        # CRITICAL: Initialize action head to prefer trading over holding
        with torch.no_grad():
            self.action_head[-1].bias[0] = -1.0   # HOLD penalty
            self.action_head[-1].bias[1] = 0.5    # BUY bonus
            self.action_head[-1].bias[2] = 0.5    # SELL bonus
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.normal_(module.bias, 0.0, 0.02)  # Slightly higher variance
    
    def forward(self, x, training=True):
        # Handle batch dimensions
        if x.dim() == 1:
            x = x.unsqueeze(0)
            single_sample = True
        else:
            single_sample = False
        
        # Normalize input
        x = self.input_norm(x)
        
        # Feature extraction
        features = self.features(x)
        
        # Action logits
        action_logits = self.action_head(features)
        
        # Add exploration bias
        action_logits = action_logits + self.action_bias.unsqueeze(0)
        
        # Add exploration noise during training
        if training and self.training:
            noise = torch.randn_like(action_logits) * torch.clamp(self.exploration_noise, 0.0, 0.5)
            action_logits = action_logits + noise
        
        # Apply softmax with temperature
        temperature = 1.5 if training else 1.0  # Higher temperature for exploration
        action_probs = torch.softmax(action_logits / temperature, dim=-1)
        
        if single_sample:
            action_probs = action_probs.squeeze(0)
        
        return action_probs

class ExplorationTrainer:
    """Trainer with forced exploration mechanisms"""
    
    def __init__(self, population_size: int = 15):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.population_size = population_size
        self.env = FixedForexEnvironment()
        
        logger.info(f"Exploration Trainer initialized: {population_size} bots, device={self.device}")
    
    def create_population(self) -> List[ExplorationTradingBot]:
        """Create population with diverse initialization"""
        population = []
        for i in range(self.population_size):
            bot = ExplorationTradingBot().to(self.device)
            
            # Add diversity to initial population
            with torch.no_grad():
                for param in bot.parameters():
                    if len(param.shape) > 1:  # Only for weight matrices
                        noise = torch.randn_like(param) * 0.1
                        param.add_(noise)
                
                # Randomize action biases for diversity
                bot.action_bias.data = torch.tensor([
                    -0.5 - random.random(),  # HOLD penalty varies
                    random.random(),          # BUY bias varies
                    random.random()           # SELL bias varies
                ]).to(self.device)
            
            population.append(bot)
        
        return population
    
    def evaluate_bot(self, bot: ExplorationTradingBot, steps: int = 500, exploration: bool = True) -> Dict:
        """Evaluate bot with exploration or exploitation"""
        bot.train() if exploration else bot.eval()
        env = FixedForexEnvironment()
        env.reset()
        
        total_reward = 0
        actions_count = {'HOLD': 0, 'BUY': 0, 'SELL': 0}
        action_names = ['HOLD', 'BUY', 'SELL']
        
        with torch.no_grad():
            for step in range(steps):
                obs = env._get_observation()
                obs_tensor = torch.FloatTensor(obs).to(self.device)
                
                action_probs = bot(obs_tensor, training=exploration)
                
                if exploration and random.random() < 0.1:  # 10% random exploration
                    action = random.randint(0, 2)
                else:
                    action = torch.argmax(action_probs).item()
                
                actions_count[action_names[action]] += 1
                
                obs, reward, done, _, info = env.step(action)
                total_reward += reward
                
                if done:
                    break
        
        # Bonus for trade diversity
        unique_actions = sum(1 for count in actions_count.values() if count > 0)
        diversity_bonus = (unique_actions - 1) * 100  # Bonus for using multiple actions
        
        return {
            'final_balance': env.balance,
            'total_reward': total_reward + diversity_bonus,
            'total_trades': len(env.trades),
            'actions': actions_count,
            'profit_loss': env.balance - env.initial_balance,
            'diversity_bonus': diversity_bonus,
            'action_diversity': unique_actions
        }
    
    def evolve_generation(self, population: List[ExplorationTradingBot]) -> Tuple[List[ExplorationTradingBot], List[Dict]]:
        """Evolve with focus on exploration and trading"""
        # Evaluate with exploration enabled
        results = []
        for i, bot in enumerate(population):
            result = self.evaluate_bot(bot, exploration=True)
            result['bot_id'] = i
            results.append(result)
        
        # Sort by total reward (including diversity bonus)
        results.sort(key=lambda x: x['total_reward'], reverse=True)
        
        # Select based on both performance and diversity
        elite_size = max(3, self.population_size // 4)
        
        # Always keep the most diverse traders
        diverse_results = sorted(results, key=lambda x: x['action_diversity'], reverse=True)
        elite_indices = set()
        
        # Top performers
        for i in range(elite_size):
            elite_indices.add(results[i]['bot_id'])
        
        # Most diverse
        for i in range(min(3, len(diverse_results))):
            elite_indices.add(diverse_results[i]['bot_id'])
        
        elite_bots = [population[i] for i in elite_indices]
        
        # Create new population with forced diversity
        new_population = elite_bots.copy()
        
        while len(new_population) < self.population_size:
            parent1 = random.choice(elite_bots)
            parent2 = random.choice(elite_bots)
            
            child = self.crossover(parent1, parent2)
            child = self.mutate(child, mutation_rate=0.2)  # Higher mutation for exploration
            new_population.append(child)
        
        return new_population[:self.population_size], results
    
    def crossover(self, parent1: ExplorationTradingBot, parent2: ExplorationTradingBot) -> ExplorationTradingBot:
        """Create offspring with exploration preserved"""
        child = ExplorationTradingBot().to(self.device)
        
        with torch.no_grad():
            for (name1, param1), (name2, param2), (name_child, param_child) in zip(
                parent1.named_parameters(), parent2.named_parameters(), child.named_parameters()
            ):
                if 'action_bias' in name1:  # Special handling for action bias
                    # Average the biases and add some noise
                    avg_bias = (param1 + param2) / 2
                    noise = torch.randn_like(avg_bias) * 0.1
                    param_child.data = avg_bias + noise
                else:
                    # Regular crossover
                    mask = torch.rand_like(param1) > 0.5
                    param_child.data = param1 * mask + param2 * (~mask)
        
        return child
    
    def mutate(self, bot: ExplorationTradingBot, mutation_rate: float = 0.2) -> ExplorationTradingBot:
        """Apply mutations with focus on preserving exploration"""
        with torch.no_grad():
            for name, param in bot.named_parameters():
                if torch.rand(1).item() < mutation_rate:
                    if 'action_bias' in name:
                        # Keep exploration bias strong
                        noise = torch.randn_like(param) * 0.05
                        param.add_(noise)
                        # Ensure HOLD is still penalized
                        param.data[0] = torch.clamp(param.data[0], max=-0.1)
                    else:
                        noise = torch.randn_like(param) * 0.02
                        param.add_(noise)
        
        return bot

def main():
    """Main training function with forced exploration"""
    parser = argparse.ArgumentParser(description="Fixed Smart Forex Bot Trainer with Exploration")
    parser.add_argument('--population_size', type=int, default=15, help='Number of bots')
    parser.add_argument('--generations', type=int, default=25, help='Number of generations')
    args = parser.parse_args()
    
    logger.info("🎯 Fixed Smart Trading Bot System (Exploration Forced)")
    logger.info("=" * 70)
    
    # Initialize trainer
    trainer = ExplorationTrainer(population_size=args.population_size)
    
    # Create initial population
    population = trainer.create_population()
    
    logger.info(f"🚀 Starting Training: {args.generations} generations")
    logger.info(f"🎲 Exploration mechanisms: Noise, bias, diversity rewards")
    
    best_results = []
    
    # Training loop
    for generation in range(args.generations):
        logger.info(f"\n=== Generation {generation + 1}/{args.generations} ===")
        
        population, results = trainer.evolve_generation(population)
        
        # Stats
        best = results[0]
        avg_balance = sum(r['final_balance'] for r in results) / len(results)
        avg_trades = sum(r['total_trades'] for r in results) / len(results)
        avg_diversity = sum(r['action_diversity'] for r in results) / len(results)
        
        # Action distribution
        total_actions = {}
        for r in results:
            for action, count in r['actions'].items():
                total_actions[action] = total_actions.get(action, 0) + count
        
        action_str = ", ".join([f"{k}={v}" for k, v in total_actions.items()])
        
        logger.info(f"Best: Balance=${best['final_balance']:.2f}, Trades={best['total_trades']}, Diversity={best['action_diversity']}")
        logger.info(f"Avg: Balance=${avg_balance:.2f}, Trades={avg_trades:.1f}, Diversity={avg_diversity:.1f}")
        logger.info(f"Actions: {action_str}")
        
        # Check if we're getting trades
        total_trades_pop = sum(r['total_trades'] for r in results)
        if total_trades_pop > 0:
            logger.info(f"✅ SUCCESS: Population executed {total_trades_pop} trades!")
        else:
            logger.warning("⚠️  No trades executed yet, increasing exploration...")
        
        best_results.append({
            'generation': generation + 1,
            'best_balance': best['final_balance'],
            'best_trades': best['total_trades'],
            'best_diversity': best['action_diversity'],
            'total_population_trades': total_trades_pop,
            'actions': total_actions
        })
    
    # Final evaluation
    final_results = []
    for bot in population:
        result = trainer.evaluate_bot(bot, exploration=False)  # Final eval without exploration
        final_results.append(result)
    
    final_results.sort(key=lambda x: x['final_balance'], reverse=True)
    best_final = final_results[0]
    
    logger.info("\n" + "=" * 70)
    logger.info("🏆 TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Best Final Balance: ${best_final['final_balance']:.2f}")
    logger.info(f"Profit/Loss: ${best_final['profit_loss']:.2f}")
    logger.info(f"Total Trades: {best_final['total_trades']}")
    logger.info(f"Action Diversity: {best_final['action_diversity']}/3")
    logger.info(f"Final Actions: {best_final['actions']}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"fixed_training_results_{timestamp}.json"
    
    final_data = {
        'training_summary': {
            'population_size': args.population_size,
            'generations': args.generations,
            'device': str(trainer.device),
            'timestamp': timestamp,
            'total_final_trades': sum(r['total_trades'] for r in final_results)
        },
        'best_final_result': best_final,
        'generation_progress': best_results,
        'all_final_results': final_results[:10]
    }
    
    with open(results_file, 'w') as f:
        json.dump(final_data, f, indent=2)
    
    logger.info(f"Results saved to: {results_file}")
    
    # Success metrics
    total_trades = sum(r['total_trades'] for r in final_results)
    diverse_bots = sum(1 for r in final_results if r['action_diversity'] >= 2)
    
    logger.info(f"\n📊 Final Statistics:")
    logger.info(f"   Total trades across population: {total_trades}")
    logger.info(f"   Bots using multiple actions: {diverse_bots}/{len(final_results)}")
    logger.info(f"   Best profit: ${best_final['profit_loss']:.2f}")
    
    if total_trades > 0:
        logger.info(f"\n✅ SUCCESS: Fixed system produces trading bots!")
    else:
        logger.info(f"\n❌ ISSUE: Still no trades - may need further debugging")

if __name__ == "__main__":
    main()
