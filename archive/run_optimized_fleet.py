#!/usr/bin/env python3
"""
Optimized Kelly Monte Carlo FOREX Bot Fleet Runner

This is a high-performance version designed for maximum CPU/GPU utilization.
Features include:
- 50,000 Monte Carlo scenarios per decision for GPU saturation
- Batch processing of multiple time periods simultaneously
- Maximum CPU core utilization
- Advanced memory management and vectorization
- Real-time resource monitoring

Author: TaskMaster AI System
Date: 2025-01-12
"""

import os
import sys
import time
import json
import logging
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import threading
import queue

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

# Import our optimized Kelly Monte Carlo bot system
from kelly_monte_bot import (
    KellyMonteBot, BotFleetManager, DataManager, MonteCarloEngine,
    KellyCalculator, TradingParameters, logger
)

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('optimized_fleet.log'),
        logging.StreamHandler()
    ]
)

class OptimizedFleetRunner:
    """
    High-performance fleet runner designed for maximum hardware utilization
    """
    
    def __init__(self, n_bots: int = 2000, batch_hours: int = 10):
        self.n_bots = n_bots
        self.batch_hours = batch_hours  # Process multiple hours in batch
        self.params = TradingParameters()
        self.performance_stats = {
            'total_decisions': 0,
            'total_trades': 0,
            'gpu_utilization_periods': 0,
            'cpu_utilization_periods': 0
        }
        
        # Resource monitoring
        self.monitoring = True
        self.resource_stats = []
        
        logger.info(f"OptimizedFleetRunner initialized:")
        logger.info(f"- Fleet size: {n_bots} bots")
        logger.info(f"- Batch processing: {batch_hours} hours at once")
        logger.info(f"- Monte Carlo scenarios: {self.params.monte_carlo_scenarios:,}")
        logger.info(f"- CPU cores available: {mp.cpu_count()}")
        logger.info(f"- GPU available: {torch.cuda.is_available()}")
    
    def start_resource_monitoring(self):
        """Start background resource monitoring thread"""
        def monitor_resources():
            import psutil
            import GPUtil
            
            while self.monitoring:
                try:
                    # CPU utilization
                    cpu_percent = psutil.cpu_percent(interval=1)
                    memory_percent = psutil.virtual_memory().percent
                    
                    # GPU utilization (if available)
                    gpu_percent = 0
                    gpu_memory = 0
                    if torch.cuda.is_available():
                        try:
                            gpus = GPUtil.getGPUs()
                            if gpus:
                                gpu_percent = gpus[0].load * 100
                                gpu_memory = gpus[0].memoryUtil * 100
                        except:
                            pass
                    
                    stats = {
                        'timestamp': time.time(),
                        'cpu_percent': cpu_percent,
                        'memory_percent': memory_percent,
                        'gpu_percent': gpu_percent,
                        'gpu_memory': gpu_memory
                    }
                    
                    self.resource_stats.append(stats)
                    
                    # Log high utilization periods
                    if cpu_percent > 75:
                        self.performance_stats['cpu_utilization_periods'] += 1
                    if gpu_percent > 75:
                        self.performance_stats['gpu_utilization_periods'] += 1
                    
                except Exception as e:
                    logger.debug(f"Resource monitoring error: {e}")
                
                time.sleep(1)  # Monitor every second
        
        monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
        monitor_thread.start()
        logger.info("Resource monitoring started")
    
    def run_optimized_simulation(self, 
                                simulation_hours: int = 2000,
                                save_results: bool = True) -> BotFleetManager:
        """
        Run highly optimized fleet simulation with maximum hardware utilization
        
        Args:
            simulation_hours: Number of hours to simulate
            save_results: Whether to save results to file
            
        Returns:
            BotFleetManager with complete results
        """
        logger.info("Starting OPTIMIZED Kelly Monte Carlo fleet simulation")
        logger.info("=" * 60)
        
        # Start resource monitoring
        self.start_resource_monitoring()
        
        # Initialize enhanced trading parameters for maximum performance
        enhanced_params = TradingParameters()
        enhanced_params.monte_carlo_scenarios = 50000  # Maximum GPU utilization
        
        # Create fleet manager with enhanced parameters
        fleet_manager = BotFleetManager(
            n_bots=self.n_bots,
            initial_equity=100000.0,
            params=enhanced_params
        )
        
        # Load and prepare data
        data_manager = DataManager()
        data_manager.load_h1_data("EURUSD")
        
        # Get simulation data
        sim_data = data_manager.price_data.head(simulation_hours)
        logger.info(f"Loaded {len(sim_data)} hours of market data")
        
        # Process data in optimized batches for maximum throughput
        total_batches = (len(sim_data) + self.batch_hours - 1) // self.batch_hours
        logger.info(f"Processing {total_batches} batches of {self.batch_hours} hours each")
        
        simulation_start = time.time()
        trades_executed = 0
        
        # Create batch processing queue for maximum parallel efficiency
        batch_queue = queue.Queue()
        for i in range(0, len(sim_data), self.batch_hours):
            batch_end = min(i + self.batch_hours, len(sim_data))
            batch_data = sim_data.iloc[i:batch_end]
            batch_queue.put((i // self.batch_hours, batch_data))
        
        # Process batches with maximum parallelization
        completed_batches = 0
        with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
            
            # Submit all batch jobs simultaneously for maximum CPU utilization
            batch_futures = []
            while not batch_queue.empty():
                try:
                    batch_num, batch_data = batch_queue.get_nowait()
                    future = executor.submit(
                        self._process_batch_optimized,
                        fleet_manager, batch_data, batch_num
                    )
                    batch_futures.append((batch_num, future))
                except queue.Empty:
                    break
            
            # Collect results as they complete
            for batch_num, future in tqdm(batch_futures, desc="Processing batches"):
                try:
                    batch_trades = future.result()
                    trades_executed += batch_trades
                    completed_batches += 1
                    
                    # Log progress every 10 batches
                    if completed_batches % 10 == 0:
                        elapsed = time.time() - simulation_start
                        rate = completed_batches / elapsed * self.batch_hours
                        logger.info(
                            f"Batch {completed_batches}/{total_batches} complete. "
                            f"Rate: {rate:.1f} hours/sec. Trades: {trades_executed:,}"
                        )
                
                except Exception as e:
                    logger.error(f"Batch {batch_num} failed: {e}")
        
        # Stop monitoring
        self.monitoring = False
        
        simulation_time = time.time() - simulation_start
        logger.info("=" * 60)
        logger.info("OPTIMIZED SIMULATION COMPLETED")
        logger.info(f"Total time: {simulation_time:.2f} seconds")
        logger.info(f"Hours processed: {len(sim_data):,}")
        logger.info(f"Processing rate: {len(sim_data)/simulation_time:.1f} hours/second")
        logger.info(f"Total trades executed: {trades_executed:,}")
        logger.info(f"Total bot decisions: {self.performance_stats['total_decisions']:,}")
        logger.info(f"CPU high utilization periods: {self.performance_stats['cpu_utilization_periods']}")
        logger.info(f"GPU high utilization periods: {self.performance_stats['gpu_utilization_periods']}")
        
        # Calculate resource utilization summary
        if self.resource_stats:
            avg_cpu = np.mean([s['cpu_percent'] for s in self.resource_stats])
            avg_gpu = np.mean([s['gpu_percent'] for s in self.resource_stats])
            avg_memory = np.mean([s['memory_percent'] for s in self.resource_stats])
            
            logger.info(f"Average CPU utilization: {avg_cpu:.1f}%")
            logger.info(f"Average GPU utilization: {avg_gpu:.1f}%")
            logger.info(f"Average Memory utilization: {avg_memory:.1f}%")
        
        # Save results if requested
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"optimized_fleet_results_{timestamp}.json"
            fleet_manager.save_results(results_file)
            
            # Save resource utilization data
            resource_file = f"resource_utilization_{timestamp}.json"
            with open(resource_file, 'w') as f:
                json.dump({
                    'resource_stats': self.resource_stats,
                    'performance_summary': {
                        'avg_cpu': avg_cpu if self.resource_stats else 0,
                        'avg_gpu': avg_gpu if self.resource_stats else 0,
                        'avg_memory': avg_memory if self.resource_stats else 0,
                        'simulation_time': simulation_time,
                        'hours_processed': len(sim_data),
                        'processing_rate': len(sim_data)/simulation_time,
                        'total_trades': trades_executed
                    }
                }, f, indent=2)
            
            logger.info(f"Results saved to {results_file}")
            logger.info(f"Resource data saved to {resource_file}")
        
        return fleet_manager
    
    def _process_batch_optimized(self, 
                               fleet_manager: BotFleetManager,
                               batch_data: pd.DataFrame,
                               batch_num: int) -> int:
        """
        Process a batch of hours with maximum optimization
        
        Args:
            fleet_manager: Fleet manager instance
            batch_data: DataFrame with batch market data
            batch_num: Batch number for logging
            
        Returns:
            Number of trades executed in this batch
        """
        batch_trades = 0
        
        # Process each hour in the batch
        for i, (timestamp, row) in enumerate(batch_data.iterrows()):
            current_price = row['close']
            
            # Get ALL bot decisions in parallel with maximum CPU utilization
            start_time = time.time()
            decisions = fleet_manager.run_parallel_decisions(
                current_price=current_price,
                market_data=row,
                timestamp=timestamp
            )
            decision_time = time.time() - start_time
            
            # Count decisions for performance tracking
            valid_decisions = sum(1 for d in decisions if d is not None)
            self.performance_stats['total_decisions'] += valid_decisions
            
            # Execute trades for bots with decisions (parallelized)
            def execute_bot_trade(bot_decision_pair):
                bot, decision = bot_decision_pair
                if decision is not None and bot.current_position is None:
                    try:
                        bot.execute_trade(decision)
                        return 1
                    except Exception as e:
                        logger.debug(f"Trade execution failed for bot {bot.bot_id}: {e}")
                return 0
            
            # Execute trades in parallel
            bot_decision_pairs = [
                (fleet_manager.bots[j], decisions[j]) 
                for j in range(len(decisions))
            ]
            
            with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
                trade_results = list(executor.map(execute_bot_trade, bot_decision_pairs))
                batch_trades += sum(trade_results)
            
            # Simplified position management (vectorized for performance)
            self._manage_positions_vectorized(fleet_manager, current_price)
        
        return batch_trades
    
    def _manage_positions_vectorized(self, 
                                   fleet_manager: BotFleetManager,
                                   current_price: float):
        """
        Vectorized position management for maximum performance
        """
        open_positions = [
            (i, bot) for i, bot in enumerate(fleet_manager.bots)
            if bot.current_position is not None
        ]
        
        if not open_positions:
            return
        
        # Batch process position closures
        closure_probability = 0.1  # 10% chance to close each hour
        random_values = np.random.random(len(open_positions))
        
        for i, (bot_idx, bot) in enumerate(open_positions):
            if random_values[i] < closure_probability:
                try:
                    # Simulate realistic price movement
                    price_change = np.random.normal(0, 0.001) * current_price
                    exit_price = current_price + price_change
                    
                    # Determine exit reason
                    entry_price = bot.current_position['entry_price']
                    signal = bot.current_position['signal']
                    
                    if signal == 'BUY':
                        pips_moved = (exit_price - entry_price) / self.params.pip_value
                    else:
                        pips_moved = (entry_price - exit_price) / self.params.pip_value
                    
                    if pips_moved >= self.params.take_profit_pips:
                        exit_reason = "TAKE_PROFIT"
                    elif pips_moved <= -self.params.stop_loss_pips:
                        exit_reason = "STOP_LOSS"
                    else:
                        exit_reason = "NATURAL"
                    
                    bot.close_trade(exit_price, exit_reason)
                    self.performance_stats['total_trades'] += 1
                
                except Exception as e:
                    logger.debug(f"Position closure failed for bot {bot.bot_id}: {e}")


def main():
    """Main execution function for optimized fleet"""
    print("OPTIMIZED Kelly Monte Carlo FOREX Bot Fleet")
    print("=" * 60)
    print("MAXIMUM HARDWARE UTILIZATION MODE")
    print("=" * 60)
    
    # Configuration for maximum performance
    N_BOTS = int(os.getenv('N_BOTS', '2000'))
    SIMULATION_HOURS = int(os.getenv('SIM_HOURS', '2000'))
    BATCH_HOURS = int(os.getenv('BATCH_HOURS', '10'))
    
    print(f"Fleet Size: {N_BOTS:,} bots")
    print(f"Simulation: {SIMULATION_HOURS:,} hours")
    print(f"Batch Size: {BATCH_HOURS} hours")
    print(f"Monte Carlo Scenarios: 50,000 per decision")
    print(f"CPU Cores: {mp.cpu_count()}")
    print(f"GPU Available: {torch.cuda.is_available()}")
    print()
    
    try:
        # Create optimized runner
        runner = OptimizedFleetRunner(
            n_bots=N_BOTS,
            batch_hours=BATCH_HOURS
        )
        
        # Run optimized simulation
        fleet_manager = runner.run_optimized_simulation(
            simulation_hours=SIMULATION_HOURS,
            save_results=True
        )
        
        # Calculate final fleet performance
        fleet_performance = fleet_manager.get_fleet_performance()
        
        print("\n" + "=" * 60)
        print("FINAL PERFORMANCE SUMMARY")
        print("=" * 60)
        print(f"Total bots: {len(fleet_manager.bots):,}")
        print(f"Average trades per bot: {fleet_performance.get('avg_trades_per_bot', 0):.1f}")
        print(f"Fleet win rate: {fleet_performance.get('avg_win_rate', 0)*100:.1f}%")
        print(f"Average return per bot: {fleet_performance.get('avg_return_per_bot', 0)*100:.2f}%")
        print(f"Fleet total PnL: ${fleet_performance.get('total_pnl', 0):,.2f}")
        
        logger.info("OPTIMIZED Kelly Monte Carlo bot fleet simulation completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
    except Exception as e:
        logger.error(f"Optimized simulation failed: {e}")
        raise


if __name__ == "__main__":
    main()
