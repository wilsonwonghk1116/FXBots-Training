#!/usr/bin/env python3
"""
Quick Smart Bot Demo - Streamlined demonstration of fixed trading bots
Shows the enhanced SmartTradingBot in action with full technical indicators
"""

import torch
import torch.nn as nn
import numpy as np
import random
import json
import logging
from datetime import datetime
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Import the fixed components
from run_smart_real_training import SmartForexEnvironment, SmartTradingBot

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QuickSmartTrainer:
    """Quick trainer for SmartTradingBot demonstration"""
    
    def __init__(self, population_size: int = 10):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.population_size = population_size
        self.env = SmartForexEnvironment(initial_balance=50000.0)  # $50k starting balance
        
        logger.info(f"Quick Smart Trainer initialized: {population_size} bots, device={self.device}")
    
    def create_population(self) -> List[SmartTradingBot]:
        """Create population of SmartTradingBots"""
        logger.info("Creating population of enhanced SmartTradingBots...")
        return [SmartTradingBot().to(self.device) for _ in range(self.population_size)]
    
    def evaluate_bot(self, bot: SmartTradingBot, episodes: int = 3) -> Dict:
        """Evaluate a single SmartTradingBot"""
        bot.eval()
        total_metrics = {
            'total_reward': 0,
            'total_trades': 0,
            'final_balance': 0,
            'actions_taken': {'HOLD': 0, 'BUY': 0, 'SELL': 0},
            'first_trade_step': None,
            'profitable_trades': 0
        }
        
        all_trades = []
        
        for episode in range(episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            episode_trades_start = len(all_trades)
            
            for step in range(500):  # Medium-length episodes
                obs_tensor = torch.FloatTensor(obs).to(self.device)
                
                with torch.no_grad():
                    action_probs, position_size = bot(obs_tensor)
                    action = torch.argmax(action_probs).item()
                
                obs, reward, done, _, info = self.env.step(action, position_size.item())
                episode_reward += reward
                
                # Track actions
                action_names = ['HOLD', 'BUY', 'SELL']
                total_metrics['actions_taken'][action_names[action]] += 1
                
                # Track first trade
                if info.get('trade_executed', False) and total_metrics['first_trade_step'] is None:
                    total_metrics['first_trade_step'] = step
                
                # Log interesting steps
                if step % 100 == 0 and step > 0:
                    logger.info(f"  Episode {episode+1}, Step {step}: Action={action_names[action]}, "
                              f"Balance=${info.get('balance', 0):.2f}, Position={info.get('position', 0)}, "
                              f"Trades={info.get('total_trades', 0)}")
                
                if done:
                    break
            
            # Collect trades from this episode
            episode_trades = self.env.trades[episode_trades_start:]
            all_trades.extend(episode_trades)
            
            total_metrics['total_reward'] += episode_reward
            total_metrics['final_balance'] += self.env.balance
            total_metrics['total_trades'] += len(self.env.trades)
            
            logger.info(f"  Episode {episode+1} complete: Balance=${self.env.balance:.2f}, "
                       f"Trades={len(self.env.trades)}, Reward={episode_reward:.2f}")
        
        # Average metrics over episodes
        for key in ['total_reward', 'final_balance']:
            total_metrics[key] /= episodes
        
        # Calculate profitable trades
        profits = [t.get('profit', 0) for t in all_trades if 'profit' in t]
        total_metrics['profitable_trades'] = len([p for p in profits if p > 0])
        total_metrics['win_rate'] = (total_metrics['profitable_trades'] / max(len(profits), 1)) * 100
        
        # Calculate action percentages
        total_actions = sum(total_metrics['actions_taken'].values())
        total_metrics['action_percentages'] = {
            k: (v / max(total_actions, 1)) * 100 for k, v in total_metrics['actions_taken'].items()
        }
        
        return total_metrics
    
    def mutate(self, bot: SmartTradingBot, mutation_rate: float = 0.05) -> SmartTradingBot:
        """Apply light mutation to bot"""
        new_bot = SmartTradingBot().to(self.device)
        new_bot.load_state_dict(bot.state_dict())
        
        with torch.no_grad():
            for param in new_bot.parameters():
                if torch.rand(1).item() < mutation_rate:
                    noise = torch.randn_like(param) * 0.01
                    param.add_(noise)
        
        return new_bot
    
    def crossover(self, parent1: SmartTradingBot, parent2: SmartTradingBot) -> SmartTradingBot:
        """Create offspring through crossover"""
        child = SmartTradingBot().to(self.device)
        
        with torch.no_grad():
            for (name1, param1), (name2, param2), (name_child, param_child) in zip(
                parent1.named_parameters(), parent2.named_parameters(), child.named_parameters()
            ):
                mask = torch.rand_like(param1) > 0.5
                param_child.data = param1 * mask + param2 * (~mask)
        
        return child
    
    def train_smart_bots(self, generations: int = 8) -> Dict:
        """Train SmartTradingBots and demonstrate their capabilities"""
        logger.info(f"🚀 Training Enhanced SmartTradingBots: {generations} generations")
        logger.info("Features: Technical indicators, LSTM, attention, advanced reward system")
        
        # Create initial population
        population = self.create_population()
        
        generation_stats = []
        best_bots_history = []
        
        for generation in range(generations):
            logger.info(f"\n=== Generation {generation + 1}/{generations} ===")
            
            # Evaluate population
            results = []
            for i, bot in enumerate(population):
                logger.info(f"Evaluating bot {i+1}/{len(population)}...")
                metrics = self.evaluate_bot(bot)
                metrics['bot_id'] = i
                results.append(metrics)
            
            # Sort by performance (balance * trade activity)
            results.sort(key=lambda x: x['final_balance'] * (1 + x['total_trades'] * 0.1), reverse=True)
            best_bot_metrics = results[0]
            
            # Log generation stats
            avg_balance = np.mean([r['final_balance'] for r in results])
            avg_trades = np.mean([r['total_trades'] for r in results])
            total_actions = {}
            for r in results:
                for action, count in r['actions_taken'].items():
                    total_actions[action] = total_actions.get(action, 0) + count
            
            logger.info(f"\nGeneration {generation + 1} Results:")
            logger.info(f"  Best Bot: Balance=${best_bot_metrics['final_balance']:.2f}, "
                       f"Trades={best_bot_metrics['total_trades']}, "
                       f"Win Rate={best_bot_metrics['win_rate']:.1f}%")
            logger.info(f"  Population Avg: Balance=${avg_balance:.2f}, Trades={avg_trades:.1f}")
            logger.info(f"  Actions: HOLD={total_actions.get('HOLD', 0)}, "
                       f"BUY={total_actions.get('BUY', 0)}, SELL={total_actions.get('SELL', 0)}")
            
            if best_bot_metrics['first_trade_step'] is not None:
                logger.info(f"  First trade at step: {best_bot_metrics['first_trade_step']}")
            
            generation_stats.append({
                'generation': generation + 1,
                'best_balance': best_bot_metrics['final_balance'],
                'avg_balance': avg_balance,
                'best_trades': best_bot_metrics['total_trades'],
                'avg_trades': avg_trades,
                'actions': total_actions,
                'win_rate': best_bot_metrics['win_rate'],
                'first_trade_step': best_bot_metrics['first_trade_step']
            })
            
            best_bots_history.append(best_bot_metrics)
            
            # Evolution for next generation
            if generation < generations - 1:
                elite_size = max(2, self.population_size // 3)
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
        
        # Final champion analysis
        champion = population[results[0]['bot_id']]
        logger.info("\n🏆 Testing champion in detailed simulation...")
        champion_detailed = self.detailed_test(champion)
        
        return {
            'training_summary': {
                'generations': generations,
                'population_size': self.population_size,
                'device': str(self.device),
                'timestamp': datetime.now().isoformat()
            },
            'generation_stats': generation_stats,
            'champion_metrics': results[0],
            'champion_detailed': champion_detailed,
            'evolution_progress': best_bots_history
        }
    
    def detailed_test(self, bot: SmartTradingBot) -> Dict:
        """Detailed test of champion bot with full analysis"""
        bot.eval()
        
        # Extended test run
        obs, _ = self.env.reset()
        actions_log = []
        decision_log = []
        
        for step in range(1000):  # Longer test
            obs_tensor = torch.FloatTensor(obs).to(self.device)
            
            with torch.no_grad():
                action_probs, position_size = bot(obs_tensor)
                action = torch.argmax(action_probs).item()
                action_prob_values = action_probs.cpu().numpy()
            
            obs, reward, done, _, info = self.env.step(action, position_size.item())
            
            action_names = ['HOLD', 'BUY', 'SELL']
            
            # Log detailed decision info
            decision_entry = {
                'step': step,
                'action': action_names[action],
                'action_probs': {
                    'HOLD': float(action_prob_values[0]),
                    'BUY': float(action_prob_values[1]),
                    'SELL': float(action_prob_values[2])
                },
                'position_size': float(position_size.item()),
                'reward': float(reward),
                'balance': float(info['balance']),
                'position': int(info['position']),
                'trade_executed': bool(info.get('trade_executed', False))
            }
            
            decision_log.append(decision_entry)
            
            # Simplified action log
            actions_log.append(action_names[action])
            
            if done:
                break
        
        # Calculate comprehensive stats
        action_counts = {'HOLD': 0, 'BUY': 0, 'SELL': 0}
        for action in actions_log:
            action_counts[action] += 1
        
        trades_executed = [d for d in decision_log if d['trade_executed']]
        
        # Analyze decision confidence
        high_confidence_decisions = []
        for d in decision_log:
            max_prob = max(d['action_probs'].values())
            if max_prob > 0.7:  # High confidence threshold
                high_confidence_decisions.append(d)
        
        return {
            'final_balance': decision_log[-1]['balance'] if decision_log else 0,
            'total_steps': len(decision_log),
            'total_trades': len(trades_executed),
            'action_distribution': action_counts,
            'action_percentages': {
                k: (v / len(actions_log)) * 100 for k, v in action_counts.items()
            },
            'trades_executed_steps': [t['step'] for t in trades_executed],
            'high_confidence_decisions': len(high_confidence_decisions),
            'confidence_rate': (len(high_confidence_decisions) / len(decision_log)) * 100,
            'recent_decisions': decision_log[-10:],  # Last 10 decisions
            'balance_curve': [d['balance'] for d in decision_log][-50:]  # Last 50 balance points
        }

def main():
    """Run quick smart bot demonstration"""
    logger.info("🎯 Enhanced SmartTradingBot Demo")
    logger.info("Features: Technical indicators, LSTM, attention, advanced reward system")
    
    # Create trainer
    trainer = QuickSmartTrainer(population_size=8)  # Small population for speed
    
    # Run training
    results = trainer.train_smart_bots(generations=6)  # Quick training
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"smart_bot_demo_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print comprehensive summary
    logger.info("\n" + "="*70)
    logger.info("🏆 ENHANCED SMARTTRADINGBOT DEMO COMPLETE")
    logger.info("="*70)
    
    champion = results['champion_metrics']
    detailed = results['champion_detailed']
    
    logger.info(f"Champion Performance:")
    logger.info(f"  💰 Final Balance: ${champion['final_balance']:.2f}")
    logger.info(f"  📊 Total Trades: {champion['total_trades']}")
    logger.info(f"  🎯 Win Rate: {champion['win_rate']:.1f}%")
    logger.info(f"  ⚡ First Trade Step: {champion.get('first_trade_step', 'N/A')}")
    logger.info(f"  📈 Action Distribution: {champion['action_percentages']}")
    
    logger.info(f"\nDetailed Analysis:")
    logger.info(f"  🔍 Total Steps: {detailed['total_steps']}")
    logger.info(f"  🤖 High Confidence Decisions: {detailed['high_confidence_decisions']} ({detailed['confidence_rate']:.1f}%)")
    logger.info(f"  📊 Action Breakdown: {detailed['action_percentages']}")
    logger.info(f"  💼 Trades Executed: {detailed['total_trades']}")
    
    logger.info(f"\nEvolution Progress:")
    for i, stat in enumerate(results['generation_stats']):
        if i == 0 or i == len(results['generation_stats']) - 1 or i % 2 == 0:
            logger.info(f"  Gen {stat['generation']}: Balance=${stat['best_balance']:.2f}, "
                       f"Trades={stat['best_trades']}, Win Rate={stat['win_rate']:.1f}%")
    
    logger.info(f"\n💾 Results saved to: {filename}")
    
    # Show recent decision making
    logger.info(f"\nRecent Champion Decisions:")
    for decision in detailed['recent_decisions'][-5:]:
        logger.info(f"  Step {decision['step']}: {decision['action']} "
                   f"(confidence: {max(decision['action_probs'].values()):.2f}) "
                   f"Balance: ${decision['balance']:.2f}")
    
    logger.info("\n✅ Demo shows enhanced SmartTradingBots with:")
    logger.info("   • Technical indicators (RSI, MACD, Bollinger Bands)")
    logger.info("   • LSTM for temporal pattern recognition")
    logger.info("   • Multi-head attention mechanism")
    logger.info("   • Advanced reward system with first trade bonus")
    logger.info("   • Real trade execution and learning")
    logger.info("   • Varying action probabilities and decision confidence")
    
    return results

if __name__ == "__main__":
    results = main()
