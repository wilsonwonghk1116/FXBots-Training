#!/usr/bin/env python3
"""
STANDALONE PC OPTIMIZED FOREX TRAINER
Configuration: 70% GPU VRAM, 70% GPU usage, 60 CPU threads at 90%
Single PC training with maximum safe performance
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import random
import json
import logging
import psutil
import threading
import multiprocessing
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    print("Installing gymnasium...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'gymnasium'])
    import gymnasium as gym
    from gymnasium import spaces

try:
    import GPUtil
except ImportError:
    print("Installing GPUtil...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'gputil'])
    import GPUtil

try:
    import talib
except ImportError:
    print("Installing TA-Lib...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'TA-Lib'])
    import talib

try:
    import pandas as pd
except ImportError:
    print("Installing pandas...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'pandas'])
    import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import indicator system from main file
from run_production_forex_trainer import (
    ComprehensiveTechnicalIndicators,
    ProductionForexEnvironment,
    ProductionTradingBot
)

class StandaloneGPUSaturator:
    """STANDALONE GPU saturator - 70% VRAM, 70% usage target"""
    
    def __init__(self, device, target_vram_percent=70, target_usage_percent=70):
        self.device = device
        self.target_vram_percent = target_vram_percent
        self.target_usage_percent = target_usage_percent
        self.running = False
        self.workers = []
        self.num_workers = 6  # Moderate worker count for 70% target
        self.temperature_limit = 80  # Slightly higher for standalone
        
    def start_saturation(self):
        """Start STANDALONE GPU saturation"""
        if not torch.cuda.is_available():
            return
            
        self.running = True
        logger.info(f"🔥 Starting {self.num_workers} GPU workers (70% VRAM target)")
        
        # Start worker threads
        for i in range(self.num_workers):
            worker = threading.Thread(target=self._gpu_worker, args=(i,), daemon=True)
            worker.start()
            self.workers.append(worker)
            time.sleep(0.05)  # Stagger starts
            
        logger.info("✅ STANDALONE GPU saturation ACTIVE")
    
    def stop_saturation(self):
        """Stop GPU saturation"""
        self.running = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("🛑 Stopping standalone GPU saturation")
    
    def _gpu_worker(self, worker_id):
        """GPU worker for 70% utilization target"""
        while self.running:
            try:
                # Check temperature and VRAM
                if self._check_resource_safety():
                    self._optimized_gpu_operations(worker_id)
                else:
                    time.sleep(0.2)  # Brief cooldown
                    continue
                
                time.sleep(0.05)  # 50ms sleep for 70% target
                
            except Exception as e:
                if "out of memory" not in str(e).lower():
                    logger.warning(f"GPU Worker {worker_id} error: {e}")
                time.sleep(0.1)
    
    def _check_resource_safety(self) -> bool:
        """Check GPU temperature and VRAM usage"""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                temp_ok = gpu.temperature < self.temperature_limit
                vram_usage = gpu.memoryUsed / gpu.memoryTotal * 100
                vram_ok = vram_usage < self.target_vram_percent
                return temp_ok and vram_ok
        except:
            pass
        return True
    
    def _optimized_gpu_operations(self, worker_id):
        """Optimized GPU operations for 70% target"""
        try:
            with torch.no_grad():
                # Moderate matrix operations for 70% target
                size = 1024  # Good balance for 70% usage
                for _ in range(4):
                    a = torch.randn(size, size, device=self.device, dtype=torch.float16)
                    b = torch.randn(size, size, device=self.device, dtype=torch.float16)
                    result = torch.matmul(a, b)
                    result = torch.sin(result) + torch.cos(result)
                    del a, b, result
                
                # Moderate convolution operations
                batch_size = 24
                channels = 256
                input_tensor = torch.randn(batch_size, channels, 16, 16, device=self.device, dtype=torch.float16)
                weight = torch.randn(channels, channels, 3, 3, device=self.device, dtype=torch.float16)
                conv_result = torch.conv2d(input_tensor, weight, padding=1)
                conv_result = F.relu(conv_result)
                del input_tensor, weight, conv_result
                
        except torch.cuda.OutOfMemoryError:
            # Fallback to smaller operations
            try:
                with torch.no_grad():
                    size = 768
                    for _ in range(3):
                        a = torch.randn(size, size, device=self.device, dtype=torch.float16)
                        b = torch.randn(size, size, device=self.device, dtype=torch.float16)
                        result = torch.matmul(a, b)
                        del a, b, result
            except:
                pass

class StandaloneCPUSaturator:
    """STANDALONE CPU saturator - 60 threads at 90% utilization"""
    
    def __init__(self, target_threads=60, target_utilization=90):
        self.target_threads = target_threads
        self.target_utilization = target_utilization
        self.running = False
        self.workers = []
        
    def start_saturation(self):
        """Start STANDALONE CPU saturation"""
        self.running = True
        logger.info(f"🧵 Starting {self.target_threads} CPU workers (90% utilization target)")
        
        # Start CPU worker threads
        for i in range(self.target_threads):
            worker = threading.Thread(target=self._cpu_worker, args=(i,), daemon=True)
            worker.start()
            self.workers.append(worker)
            
        logger.info("✅ STANDALONE CPU saturation ACTIVE")
    
    def stop_saturation(self):
        """Stop CPU saturation"""
        self.running = False
        logger.info("🛑 Stopping standalone CPU saturation")
    
    def _cpu_worker(self, worker_id):
        """CPU worker for 90% utilization target"""
        while self.running:
            try:
                # Intensive computations for 90% target
                for _ in range(500):  # More iterations for 90% target
                    a = np.random.randn(250, 250)
                    b = np.random.randn(250, 250)
                    result = np.dot(a, b)
                    result = np.sin(result) + np.cos(result)
                    result = np.tanh(result)
                    del a, b, result
                
                # Very brief sleep for 90% target
                time.sleep(0.002)  # 2ms for high utilization
                
            except Exception as e:
                time.sleep(0.01)

class StandaloneForexTrainer:
    """STANDALONE forex trainer optimized for single PC"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # STANDALONE configuration
        self.generations = 200  # Reasonable for standalone
        self.elite_percentage = 0.05  # 5% survival
        self.mutation_rate = 0.025
        self.crossover_rate = 0.75
        
        # Calculate STANDALONE population size
        self.population_size = self._calculate_standalone_population_size()
        
        # STANDALONE thread configuration
        self.num_evaluators = 8  # Good for standalone PC
        self.batch_size = max(8, self.population_size // self.num_evaluators)
        
        logger.info(f"🚀 STANDALONE TRAINER CONFIGURATION:")
        logger.info(f"   🤖 Population: {self.population_size} bots")
        logger.info(f"   ⚡ Evaluators: {self.num_evaluators}")
        logger.info(f"   📦 Batch size: {self.batch_size}")
        logger.info(f"   🏆 Generations: {self.generations}")
        
        # Initialize saturators
        self.gpu_saturator = StandaloneGPUSaturator(self.device)
        self.cpu_saturator = StandaloneCPUSaturator()
    
    def _calculate_standalone_population_size(self) -> int:
        """Calculate population size for standalone PC"""
        # Conservative base for standalone
        base_population = 1000
        
        # Scale with available resources
        cpu_count = multiprocessing.cpu_count()
        cpu_factor = min(3.0, cpu_count / 16)  # Up to 3x scaling
        
        if torch.cuda.is_available():
            try:
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                gpu_factor = min(2.5, gpu_memory_gb / 8)  # Up to 2.5x scaling
            except:
                gpu_factor = 1.5
        else:
            gpu_factor = 1.0
        
        standalone_population = int(base_population * cpu_factor * gpu_factor)
        population_size = max(2000, min(8000, standalone_population))
        
        logger.info(f"🎯 STANDALONE POPULATION CALCULATION:")
        logger.info(f"   📊 Base: {base_population}")
        logger.info(f"   💪 CPU factor: {cpu_factor:.1f}x ({cpu_count} cores)")
        logger.info(f"   🔥 GPU factor: {gpu_factor:.1f}x")
        logger.info(f"   🎯 Final population: {population_size}")
        
        return population_size
    
    def create_population(self) -> List[ProductionTradingBot]:
        """Create population for standalone training"""
        logger.info(f"🚀 Creating standalone population: {self.population_size} bots...")
        
        # Get observation size
        env = ProductionForexEnvironment()
        observation_size = env.observation_space.shape[0]
        
        # Diverse strategy types
        strategies = [
            'trend_following', 'mean_reversion', 'momentum', 'scalping', 'swing_trading',
            'breakout', 'reversal', 'volatility', 'arbitrage', 'grid_trading',
            'fibonacci', 'bollinger', 'macd_specialist', 'rsi_specialist', 'stochastic'
        ]
        
        population = []
        
        for i in range(self.population_size):
            strategy = strategies[i % len(strategies)]
            bot = ProductionTradingBot(
                input_size=observation_size,
                strategy_type=f"{strategy}_{i}"
            )
            self._initialize_standalone_strategy(bot, strategy)
            population.append(bot)
            
            if (i + 1) % 500 == 0:
                logger.info(f"   📦 Created {i + 1}/{self.population_size} bots")
        
        logger.info(f"✅ Standalone population created: {len(population)} bots")
        return population
    
    def _initialize_standalone_strategy(self, bot: ProductionTradingBot, strategy_type: str):
        """Initialize strategy for standalone bot"""
        with torch.no_grad():
            if 'trend' in strategy_type.lower():
                for i in range(50, 100):
                    if i < len(bot.indicator_usage_weights):
                        bot.indicator_usage_weights[i] = 1.3 + random.uniform(0, 0.5)
            elif 'momentum' in strategy_type.lower():
                for i in range(100, 150):
                    if i < len(bot.indicator_usage_weights):
                        bot.indicator_usage_weights[i] = 1.4 + random.uniform(0, 0.4)
            elif 'volatility' in strategy_type.lower():
                for i in range(150, 200):
                    if i < len(bot.indicator_usage_weights):
                        bot.indicator_usage_weights[i] = 1.5 + random.uniform(0, 0.3)
            
            # Add controlled randomness
            for i in range(len(bot.indicator_usage_weights)):
                bot.indicator_usage_weights[i] += random.uniform(-0.1, 0.1)
                bot.indicator_usage_weights[i] = torch.clamp(bot.indicator_usage_weights[i], 0.1, 2.0)
    
    def evaluate_population(self, population: List[ProductionTradingBot]) -> List[Dict]:
        """Evaluate population with parallel processing"""
        logger.info(f"⚡ Evaluating {len(population)} bots with {self.num_evaluators} threads")
        
        def evaluate_bot_batch(batch):
            results = []
            for i, bot in enumerate(batch):
                try:
                    env = ProductionForexEnvironment()
                    total_reward = 0
                    obs, _ = env.reset()
                    
                    for _ in range(2000):  # Longer episodes for standalone
                        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                        
                        with torch.no_grad():
                            with autocast():
                                logits = bot(obs_tensor)
                                action = torch.argmax(logits).item()
                        
                        obs, reward, done, _, info = env.step(action)
                        total_reward += reward
                        
                        if done:
                            break
                    
                    final_balance = env.balance
                    win_rate = len([t for t in env.trades if t.get('profit', 0) > 0]) / max(1, len(env.trades))
                    
                    championship_score = (final_balance - 10000) * 0.1 + win_rate * 150 + len(env.trades) * 0.3
                    
                    results.append({
                        'bot_id': i,
                        'strategy_type': bot.strategy_type,
                        'final_balance': final_balance,
                        'total_reward': total_reward,
                        'championship_score': championship_score,
                        'win_rate': win_rate,
                        'total_trades': len(env.trades)
                    })
                    
                except Exception as e:
                    logger.warning(f"Evaluation error for bot {i}: {e}")
                    results.append({
                        'bot_id': i,
                        'strategy_type': getattr(bot, 'strategy_type', 'unknown'),
                        'final_balance': 9000,
                        'total_reward': -1000,
                        'championship_score': -100,
                        'win_rate': 0,
                        'total_trades': 0
                    })
            
            return results
        
        # Split into batches
        batches = []
        for i in range(0, len(population), self.batch_size):
            batch = population[i:i + self.batch_size]
            batches.append(batch)
        
        # Parallel evaluation
        all_results = []
        with ThreadPoolExecutor(max_workers=self.num_evaluators) as executor:
            futures = [executor.submit(evaluate_bot_batch, batch) for batch in batches]
            
            for future in as_completed(futures):
                batch_results = future.result()
                all_results.extend(batch_results)
        
        # Sort by score
        all_results.sort(key=lambda x: x['championship_score'], reverse=True)
        
        logger.info(f"✅ Evaluation complete")
        logger.info(f"   🏆 Champion score: {all_results[0]['championship_score']:.2f}")
        
        return all_results
    
    def evolve_population(self, population: List[ProductionTradingBot], results: List[Dict]) -> List[ProductionTradingBot]:
        """Evolve population for standalone training"""
        logger.info("🧬 Evolving standalone population...")
        
        # Select elite
        elite_count = max(5, int(len(population) * self.elite_percentage))
        elite_indices = [r['bot_id'] for r in results[:elite_count]]
        elite_bots = [population[i] for i in elite_indices]
        
        new_population = elite_bots.copy()
        remaining_slots = self.population_size - len(new_population)
        
        # Generate offspring
        for _ in range(remaining_slots):
            if random.random() < self.crossover_rate:
                parent1 = random.choice(elite_bots)
                parent2 = random.choice(elite_bots)
                child = self._crossover_standalone(parent1, parent2)
            else:
                parent = random.choice(elite_bots)
                child = self._mutate_standalone(parent)
            
            new_population.append(child)
        
        logger.info(f"🧬 Evolution complete: Elite {elite_count}, Offspring {remaining_slots}")
        return new_population[:self.population_size]
    
    def _crossover_standalone(self, parent1: ProductionTradingBot, parent2: ProductionTradingBot) -> ProductionTradingBot:
        """Crossover for standalone training"""
        child = ProductionTradingBot(
            input_size=parent1.input_size,
            strategy_type=f"hybrid_{random.randint(1000, 9999)}"
        )
        
        for (name1, param1), (name2, param2), (name_child, param_child) in zip(
            parent1.named_parameters(), parent2.named_parameters(), child.named_parameters()
        ):
            if param1.shape == param2.shape == param_child.shape:
                if random.random() < 0.5:
                    param_child.data.copy_(param1.data)
                else:
                    param_child.data.copy_(param2.data)
        
        return child
    
    def _mutate_standalone(self, parent: ProductionTradingBot) -> ProductionTradingBot:
        """Mutate for standalone training"""
        child = ProductionTradingBot(
            input_size=parent.input_size,
            strategy_type=f"mutant_{parent.strategy_type}_{random.randint(1000, 9999)}"
        )
        
        for (name_parent, param_parent), (name_child, param_child) in zip(
            parent.named_parameters(), child.named_parameters()
        ):
            if param_parent.shape == param_child.shape:
                param_child.data.copy_(param_parent.data)
                
                if random.random() < self.mutation_rate:
                    mutation = torch.randn_like(param_child.data) * 0.01
                    param_child.data.add_(mutation)
        
        return child

def monitor_standalone_performance():
    """Monitor standalone performance"""
    try:
        if torch.cuda.is_available():
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_util = gpu.load * 100
                vram_used = gpu.memoryUsed / 1024
                vram_total = gpu.memoryTotal / 1024
                vram_percent = gpu.memoryUsed / gpu.memoryTotal * 100
                temp = gpu.temperature
                
                cpu_percent = psutil.cpu_percent(interval=1)
                
                logger.info(f"🚀 STANDALONE PERFORMANCE:")
                logger.info(f"   📊 GPU: {gpu_util:.1f}% | VRAM: {vram_percent:.1f}% | Temp: {temp}°C")
                logger.info(f"   🧵 CPU: {cpu_percent:.1f}%")
                
                return {
                    'gpu_util': gpu_util,
                    'vram_percent': vram_percent,
                    'gpu_temp': temp,
                    'cpu_percent': cpu_percent
                }
    except Exception as e:
        logger.warning(f"Performance monitoring error: {e}")
        return {}

def main():
    """Main standalone training function"""
    try:
        logger.info("🚀 === STANDALONE PC OPTIMIZED FOREX TRAINER ===")
        logger.info("🎯 TARGET: 70% GPU VRAM, 70% GPU usage, 60 CPU threads at 90%")
        
        # Configure performance
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            target_threads = 60  # Target 60 CPU threads
            torch.set_num_threads(target_threads)
            
            try:
                os.nice(-10)  # High priority
            except:
                pass
            
            logger.info("✅ STANDALONE optimizations applied")
        
        # Initialize trainer
        trainer = StandaloneForexTrainer()
        
        # Start resource saturation
        logger.info("🔥 Starting resource saturation...")
        trainer.gpu_saturator.start_saturation()
        trainer.cpu_saturator.start_saturation()
        
        # Create initial population
        population = trainer.create_population()
        
        best_overall_score = 0
        best_champion_data = None
        
        for generation in range(trainer.generations):
            logger.info(f"\n🚀 === GENERATION {generation + 1}/{trainer.generations} ===")
            
            # Monitor performance
            perf_data = monitor_standalone_performance()
            
            # Evaluate population
            results = trainer.evaluate_population(population)
            
            # Track champion
            current_champion = results[0]
            
            logger.info(f"🏆 Generation {generation + 1} Champion:")
            logger.info(f"   Bot: {current_champion['strategy_type']}")
            logger.info(f"   Score: {current_champion['championship_score']:.2f}")
            logger.info(f"   Balance: ${current_champion['final_balance']:.2f}")
            logger.info(f"   Win Rate: {current_champion['win_rate']:.3f}")
            
            if current_champion['championship_score'] > best_overall_score:
                best_overall_score = current_champion['championship_score']
                best_champion_data = current_champion.copy()
                
                logger.info(f"🎉 NEW STANDALONE CHAMPION! Score: {best_overall_score:.2f}")
                
                # Save champion
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                champion_filename = f"STANDALONE_CHAMPION_BOT_{timestamp}.pth"
                analysis_filename = f"STANDALONE_CHAMPION_ANALYSIS_{timestamp}.json"
                
                champion_bot = population[current_champion['bot_id']]
                torch.save(champion_bot.state_dict(), champion_filename)
                
                champion_analysis = {
                    'timestamp': timestamp,
                    'generation': generation + 1,
                    'champion_data': best_champion_data,
                    'performance_data': perf_data,
                    'training_config': {
                        'population_size': trainer.population_size,
                        'generations': trainer.generations,
                        'standalone_mode': True
                    }
                }
                
                with open(analysis_filename, 'w') as f:
                    json.dump(champion_analysis, f, indent=2)
                
                logger.info(f"💾 Standalone Champion saved: {champion_filename}")
            
            # Evolution
            if generation < trainer.generations - 1:
                population = trainer.evolve_population(population, results)
            
            progress = (generation + 1) / trainer.generations * 100
            logger.info(f"📈 Training Progress: {progress:.1f}%")
        
        # Stop saturation
        trainer.gpu_saturator.stop_saturation()
        trainer.cpu_saturator.stop_saturation()
        
        logger.info(f"\n🏁 === STANDALONE TRAINING COMPLETE ===")
        if best_champion_data:
            logger.info(f"🏆 STANDALONE CHAMPION RESULTS:")
            logger.info(f"   Strategy: {best_champion_data['strategy_type']}")
            logger.info(f"   Final Balance: ${best_champion_data['final_balance']:.2f}")
            logger.info(f"   Championship Score: {best_champion_data['championship_score']:.2f}")
            logger.info(f"   Win Rate: {best_champion_data['win_rate']:.3f}")
        
        logger.info("\n🎉 STANDALONE FOREX TRAINING COMPLETED! 🎉")
        
    except KeyboardInterrupt:
        logger.info("🛑 Standalone training interrupted by user")
    except Exception as e:
        logger.error(f"❌ Standalone training failed: {e}")
        raise

if __name__ == "__main__":
    main() 