"""
VRAMOptimizedTrainer: Genetic algorithm-based trainer for RL bots.
"""

from typing import List, Dict
from src.env import SmartForexEnvironment
from src.model import SmartTradingBot
import torch
import logging
import multiprocessing
from functools import partial
from datetime import datetime
import json
import os
import random
import numpy as np

logger = logging.getLogger(__name__)

def _evaluate_bot(bot: SmartTradingBot, env: SmartForexEnvironment, steps: int = 1000) -> Dict:
    bot.eval()
    metrics = env.simulate_trading_detailed(bot, steps=steps)
    return metrics

class VRAMOptimizedTrainer:
    """VRAM-optimized trainer with parallel evaluation"""
    def __init__(self, population_size: int, target_vram_percent: float = 0.85):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.population_size = population_size
        self.target_vram_percent = target_vram_percent
        torch.backends.cudnn.benchmark = True
        self.generation = 0
        self.current_difficulty = 0
        self.max_difficulty = 4
        self.difficulty_increase_interval = 5
        self.num_workers = max(1, int((os.cpu_count() or 1) * 0.85))
        self.env_pool = [SmartForexEnvironment() for _ in range(self.num_workers)]
        self.env = SmartForexEnvironment()
        logger.info(f"Initialized trainer with population_size={self.population_size}")

    def create_population(self) -> List[SmartTradingBot]:
        logger.info(f"Creating population of {self.population_size} bots on {self.device}...")
        return [SmartTradingBot().to(self.device) for _ in range(self.population_size)]

    def evaluate_population(self, population: List[SmartTradingBot]) -> List[Dict]:
        """Evaluate entire population with detailed metrics using multiprocessing."""
        with multiprocessing.Pool(self.num_workers) as pool:
            envs = [SmartForexEnvironment() for _ in range(len(population))]
            results = pool.starmap(_evaluate_bot, [(bot, env, 1000) for bot, env in zip(population, envs)])
        for i, metrics in enumerate(results):
            metrics['bot_id'] = i
        return sorted(results, key=lambda x: x['final_balance'], reverse=True)

    def genetic_crossover(self, parent1: SmartTradingBot, parent2: SmartTradingBot) -> SmartTradingBot:
        """Create offspring through genetic crossover"""
        child = SmartTradingBot().to(self.device)
        with torch.no_grad():
            for (name1, param1), (name2, param2), (name_child, param_child) in zip(
                parent1.named_parameters(), parent2.named_parameters(), child.named_parameters()
            ):
                mask = torch.rand_like(param1) > 0.5
                param_child.data = param1 * mask + param2 * (~mask)
        return child

    def mutate(self, bot: SmartTradingBot, mutation_rate: float = 0.1) -> SmartTradingBot:
        """Apply mutations to bot"""
        with torch.no_grad():
            for param in bot.parameters():
                if torch.rand(1) < mutation_rate:
                    noise = torch.randn_like(param) * 0.01
                    param.add_(noise)
        return bot

    def evolve_generation(self, population: List[SmartTradingBot], elite_size: int = 100) -> List[SmartTradingBot]:
        if self.generation % self.difficulty_increase_interval == 0:
            self.current_difficulty = min(self.current_difficulty + 1, self.max_difficulty)
            for env in self.env_pool:
                env.set_difficulty(self.current_difficulty)
        results = self.evaluate_population(population)
        elite_bots = [population[result['bot_id']] for result in results[:elite_size]]
        new_population = elite_bots.copy()
        while len(new_population) < self.population_size:
            parent1 = elite_bots[random.randint(0, elite_size//2-1)]
            parent2 = elite_bots[random.randint(0, elite_size//2-1)]
            child = self.genetic_crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append(child)
        return new_population[:self.population_size], results

    def analyze_champion(self, champion_bot: SmartTradingBot, results: List[Dict]) -> Dict:
        champion_metrics = results[0]
        detailed_metrics = self.env.simulate_trading_detailed(champion_bot, steps=5000)
        analysis = {
            'champion_analysis': {
                'bot_id': champion_metrics['bot_id'],
                'final_balance': detailed_metrics['final_balance'],
                'total_return_pct': detailed_metrics['total_return_pct'],
                'win_rate': detailed_metrics['win_rate'],
                'total_trades': detailed_metrics['total_trades'],
                'gross_profit': detailed_metrics['gross_profit'],
                'gross_loss': detailed_metrics['gross_loss'],
                'profit_factor': detailed_metrics['profit_factor'],
                'average_win': detailed_metrics['avg_win'],
                'average_loss': detailed_metrics['avg_loss'],
                'risk_reward_ratio': detailed_metrics['risk_reward_ratio'],
                'max_drawdown': detailed_metrics['max_drawdown'],
                'recovery_factor': detailed_metrics['recovery_factor'],
                'sharpe_ratio': self._calculate_sharpe_ratio(detailed_metrics['balance_history']),
                'calmar_ratio': self._calculate_calmar_ratio(detailed_metrics),
                'trade_history': detailed_metrics['trades'][:50],
                'balance_curve': detailed_metrics['balance_history'][-500:]
            },
            'training_summary': {
                'population_size': self.population_size,
                'target_vram_percent': self.target_vram_percent,
                'device': str(self.device),
                'timestamp': datetime.now().isoformat()
            }
        }
        return analysis

    def _calculate_sharpe_ratio(self, balance_history: List[float]) -> float:
        if len(balance_history) < 2:
            return 0
        returns = np.diff(balance_history) / balance_history[:-1]
        if np.std(returns) == 0:
            return 0
        return np.mean(returns) / np.std(returns) * np.sqrt(252)

    def _calculate_calmar_ratio(self, metrics: Dict) -> float:
        if metrics['max_drawdown'] == 0:
            return 0
        annual_return = metrics['total_return_pct']
        return annual_return / metrics['max_drawdown']

    def save_champion(self, champion_bot: SmartTradingBot, analysis: Dict) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"CHAMPION_BOT_{timestamp}.pth"
        torch.save(champion_bot.state_dict(), filename)
        with open(filename.replace('.pth', '_analysis.json'), 'w') as f:
            json.dump(analysis, f, indent=2)
        logger.info(f"Champion and analysis saved: {filename}")
        return filename

    def adjust_trading_thresholds(self, factor: float) -> None:
        logger.warning(f"No trades executed - adjusting strategy thresholds by {factor*100:.0f}%")
        for env in self.env_pool + [self.env]:
            env.trading_cost = max(0.0, env.trading_cost * (1.0 - factor))
            env.stop_loss_pips = max(1, int(env.stop_loss_pips * (1.0 - factor)))
            env.take_profit_pips = int(env.take_profit_pips * (1.0 + factor))
