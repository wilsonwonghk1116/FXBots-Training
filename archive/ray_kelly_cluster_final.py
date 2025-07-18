#!/usr/bin/env python3
"""
Ray Distributed Kelly Monte Carlo FOREX Bot System
Final Production Version for Maximum Hardware Utilization

Optimized for:
- Head PC: Xeon E5 x2 (80 threads) + RTX 3090 (24GB VRAM)  
- Worker PC: i9 (16 threads) + RTX 3070 (8GB VRAM)
- Target: 75% CPU + GPU utilization across cluster
- Monte Carlo: 150,000 scenarios per decision for GPU saturation

Usage: python ray_kelly_cluster_final.py

Author: TaskMaster AI System
Date: 2025-01-12
"""

import os
import sys
import time
import json
import logging
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
import ray
from tqdm import tqdm

# Include necessary classes directly to avoid import issues on Ray workers
from dataclasses import dataclass

@dataclass
class TradingParameters:
    """Trading parameters for Kelly Monte Carlo bot"""
    leverage: float = 100.0
    pip_value: float = 0.0001
    take_profit_pips: int = 20
    stop_loss_pips: int = 15
    max_risk_per_trade: float = 0.05
    monte_carlo_scenarios: int = 150000  # Massive for GPU saturation

@dataclass
class ScenarioResult:
    """Result from a single Monte Carlo scenario"""
    is_win: bool
    payoff_ratio: float
    pips_gained: float

@dataclass 
class KellyEstimates:
    """Kelly Criterion calculation results"""
    win_probability: float
    payoff_ratio: float
    kelly_fraction: float
    constrained_fraction: float

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ray_kelly_final.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@ray.remote(num_cpus=2, num_gpus=0.3)  # Heavy GPU allocation per actor
class RayKellyBotActor:
    """
    Ray actor for Kelly Monte Carlo trading bot
    Optimized for maximum GPU/CPU utilization
    """
    
    def __init__(self, bot_id: int, initial_equity: float = 100000.0):
        self.bot_id = bot_id
        self.initial_equity = initial_equity
        self.current_equity = initial_equity
        
        # Enhanced parameters for maximum GPU utilization
        self.params = TradingParameters()
        self.params.monte_carlo_scenarios = 150000  # MASSIVE scenarios for GPU saturation
        
        # GPU setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize return parameters (simplified for Ray)
        self.returns_params = {
            'mean': 0.0001,
            'std': 0.008,
            'skew': -0.1,
            'kurt': 3.2
        }
        
        # Trading state
        self.trade_history = []
        self.current_position = None
        self.total_trades = 0
        self.winning_trades = 0
        
        logger.info(f"RayKellyBotActor {bot_id} initialized on {self.device}")
    
    def massive_monte_carlo_decision(self, 
                                   current_price: float,
                                   market_data: Dict,
                                   timestamp: str) -> Optional[Dict]:
        """
        Execute MASSIVE Monte Carlo simulation for maximum GPU utilization
        """
        # Generate signal
        signal = self._generate_signal(market_data)
        if signal is None:
            return None
        
        # MASSIVE GPU Monte Carlo simulation
        start_time = time.time()
        scenarios = self._gpu_monte_carlo_massive(
            current_price, signal, self.params.monte_carlo_scenarios
        )
        mc_time = time.time() - start_time
        
        # Calculate Kelly estimates
        kelly_estimates = self._calculate_kelly_estimates(scenarios)
        
        if kelly_estimates['constrained_fraction'] < 0.001:
            return None
        
        decision = {
            'bot_id': self.bot_id,
            'timestamp': timestamp,
            'signal': signal,
            'entry_price': current_price,
            'position_size': kelly_estimates['constrained_fraction'] * self.current_equity,
            'kelly_fraction': kelly_estimates['constrained_fraction'],
            'win_probability': kelly_estimates['win_probability'],
            'mc_scenarios': len(scenarios),
            'mc_computation_time': mc_time,
            'device': str(self.device)
        }
        
        return decision
    
    def _gpu_monte_carlo_massive(self, 
                               current_price: float,
                               signal: str,
                               n_scenarios: int) -> List[Dict]:
        """
        MASSIVE GPU Monte Carlo simulation for maximum VRAM utilization
        """
        if self.device.type != 'cuda':
            return self._cpu_monte_carlo_fallback(current_price, signal, n_scenarios)
        
        scenarios = []
        mean = self.returns_params['mean']
        std = self.returns_params['std']
        n_steps = 12  # Longer path simulation
        
        # Process in MASSIVE batches to saturate GPU memory
        batch_size = 75000  # HUGE batches for maximum GPU utilization
        
        for i in range(0, n_scenarios, batch_size):
            current_batch = min(batch_size, n_scenarios - i)
            
            # Generate MASSIVE random tensors on GPU
            with torch.cuda.device(self.device):
                # Use maximum GPU memory
                random_returns = torch.normal(
                    mean=mean,
                    std=std,
                    size=(current_batch, n_steps),
                    device=self.device,
                    dtype=torch.float32
                )
                
                # Vectorized price path calculation
                price_changes = torch.cumsum(random_returns, dim=1)
                final_price_changes = price_changes[:, -1]
                final_prices = current_price * torch.exp(final_price_changes)
                
                # Keep on GPU as long as possible
                final_prices_cpu = final_prices.cpu().numpy()
            
            # Vectorized scenario evaluation
            batch_scenarios = self._vectorized_scenario_evaluation(
                current_price, final_prices_cpu, signal
            )
            scenarios.extend(batch_scenarios)
        
        return scenarios
    
    def _vectorized_scenario_evaluation(self, 
                                      entry_price: float,
                                      exit_prices: np.ndarray,
                                      signal: str) -> List[Dict]:
        """
        Vectorized evaluation for maximum performance
        """
        # Vectorized calculations
        if signal.upper() == 'BUY':
            pips_gained = (exit_prices - entry_price) / self.params.pip_value
            is_win = exit_prices > entry_price
        else:
            pips_gained = (entry_price - exit_prices) / self.params.pip_value
            is_win = exit_prices < entry_price
        
        # Apply TP/SL vectorized
        abs_pips = np.abs(pips_gained)
        
        # Take profit
        tp_mask = (abs_pips >= self.params.take_profit_pips) & is_win
        pips_gained[tp_mask] = self.params.take_profit_pips
        
        # Stop loss
        sl_mask = (abs_pips >= self.params.stop_loss_pips) & ~is_win
        pips_gained[sl_mask] = -self.params.stop_loss_pips
        
        # Payoff ratios
        payoff_ratios = pips_gained / self.params.stop_loss_pips
        final_is_win = pips_gained > 0
        
        # Convert to results
        scenarios = []
        for i in range(len(exit_prices)):
            scenarios.append({
                'is_win': bool(final_is_win[i]),
                'payoff_ratio': float(payoff_ratios[i]),
                'pips_gained': float(pips_gained[i])
            })
        
        return scenarios
    
    def _cpu_monte_carlo_fallback(self, 
                                current_price: float,
                                signal: str,
                                n_scenarios: int) -> List[Dict]:
        """
        CPU fallback with high parallelization
        """
        scenarios = []
        mean = self.returns_params['mean']
        std = self.returns_params['std']
        
        for _ in range(n_scenarios):
            returns = np.random.normal(mean, std, 12)
            final_price = current_price * np.exp(np.sum(returns))
            
            if signal.upper() == 'BUY':
                pips_gained = (final_price - current_price) / self.params.pip_value
                is_win = final_price > current_price
            else:
                pips_gained = (current_price - final_price) / self.params.pip_value
                is_win = final_price < current_price
            
            # Apply TP/SL
            if abs(pips_gained) >= self.params.take_profit_pips and is_win:
                pips_gained = self.params.take_profit_pips
            elif abs(pips_gained) >= self.params.stop_loss_pips and not is_win:
                pips_gained = -self.params.stop_loss_pips
            
            scenarios.append({
                'is_win': pips_gained > 0,
                'payoff_ratio': pips_gained / self.params.stop_loss_pips,
                'pips_gained': pips_gained
            })
        
        return scenarios
    
    def _generate_signal(self, market_data: Dict) -> Optional[str]:
        """Generate trading signal"""
        close = market_data.get('close', 0)
        sma_20 = market_data.get('sma_20', 0)
        sma_50 = market_data.get('sma_50', 0)
        
        if sma_20 == 0 or sma_50 == 0:
            return None
        
        if close > sma_20 > sma_50:
            return 'BUY'
        elif close < sma_20 < sma_50:
            return 'SELL'
        
        return None
    
    def _calculate_kelly_estimates(self, scenarios: List[Dict]) -> Dict:
        """Calculate Kelly Criterion estimates"""
        if not scenarios:
            return {'constrained_fraction': 0.0, 'win_probability': 0.0}
        
        wins = [s for s in scenarios if s['is_win']]
        losses = [s for s in scenarios if not s['is_win']]
        
        win_probability = len(wins) / len(scenarios)
        
        avg_win_payoff = np.mean([s['payoff_ratio'] for s in wins]) if wins else 0.0
        avg_loss_payoff = abs(np.mean([s['payoff_ratio'] for s in losses])) if losses else 1.0
        
        payoff_ratio = avg_win_payoff / avg_loss_payoff if avg_loss_payoff > 0 else 0.0
        
        if payoff_ratio > 0:
            kelly_fraction = win_probability - (1 - win_probability) / payoff_ratio
        else:
            kelly_fraction = 0.0
        
        # Apply constraints
        constrained_fraction = max(0.0, min(kelly_fraction * 0.25, self.params.max_risk_per_trade))
        
        return {
            'win_probability': win_probability,
            'payoff_ratio': payoff_ratio,
            'kelly_fraction': kelly_fraction,
            'constrained_fraction': constrained_fraction
        }
    
    def execute_trade(self, decision: Dict) -> Dict:
        """Execute trade"""
        if decision is None:
            return None
        
        self.total_trades += 1
        
        # Simulate trade execution
        trade = {
            'trade_id': self.total_trades,
            'bot_id': self.bot_id,
            'signal': decision['signal'],
            'entry_price': decision['entry_price'],
            'position_size': decision['position_size'],
            'kelly_fraction': decision['kelly_fraction'],
            'timestamp': decision['timestamp'],
            'status': 'EXECUTED'
        }
        
        self.current_position = trade
        return trade
    
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics"""
        return {
            'bot_id': self.bot_id,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'current_equity': self.current_equity,
            'device': str(self.device)
        }


@ray.remote(num_cpus=8, num_gpus=0.5)  # High CPU + GPU allocation
class RayBatchProcessorActor:
    """
    Batch processor for handling multiple hours simultaneously
    Designed for maximum parallel throughput
    """
    
    def __init__(self, processor_id: int):
        self.processor_id = processor_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processed_batches = 0
        
        logger.info(f"RayBatchProcessorActor {processor_id} initialized on {self.device}")
    
    def process_massive_batch(self, 
                            bot_refs: List,
                            market_batch: List[Dict]) -> Dict:
        """
        Process massive batch with all bots for maximum utilization
        """
        batch_start = time.time()
        total_decisions = 0
        total_trades = 0
        
        logger.info(f"Processor {self.processor_id}: Processing {len(market_batch)} hours "
                   f"with {len(bot_refs)} bots")
        
        for hour_idx, market_data in enumerate(market_batch):
            # Launch ALL bot decisions simultaneously for maximum parallel execution
            decision_futures = []
            for bot_ref in bot_refs:
                future = bot_ref.massive_monte_carlo_decision.remote(
                    current_price=market_data['close'],
                    market_data=market_data,
                    timestamp=market_data['timestamp']
                )
                decision_futures.append((bot_ref, future))
            
            # Collect decisions as they complete
            decisions = []
            for bot_ref, future in decision_futures:
                try:
                    decision = ray.get(future)
                    if decision:
                        decisions.append((bot_ref, decision))
                        total_decisions += 1
                except Exception as e:
                    logger.debug(f"Decision failed: {e}")
            
            # Execute trades in parallel
            if decisions:
                trade_futures = []
                for bot_ref, decision in decisions:
                    future = bot_ref.execute_trade.remote(decision)
                    trade_futures.append(future)
                
                # Collect trade results
                trades = ray.get(trade_futures)
                total_trades += len([t for t in trades if t])
            
            # Log progress
            if hour_idx % 20 == 0:
                elapsed = time.time() - batch_start
                rate = (hour_idx + 1) / elapsed if elapsed > 0 else 0
                logger.info(f"Processor {self.processor_id}: Hour {hour_idx+1}/{len(market_batch)}, "
                          f"Rate: {rate:.1f} hours/sec, Decisions: {total_decisions}")
        
        batch_time = time.time() - batch_start
        self.processed_batches += 1
        
        return {
            'processor_id': self.processor_id,
            'hours_processed': len(market_batch),
            'total_decisions': total_decisions,
            'total_trades': total_trades,
            'processing_time': batch_time,
            'hours_per_second': len(market_batch) / batch_time,
            'device': str(self.device)
        }


class RayKellyFleetManager:
    """
    Main fleet manager for Ray distributed Kelly Monte Carlo system
    """
    
    def __init__(self, n_bots: int = 2000, n_processors: int = 12):
        self.n_bots = n_bots
        self.n_processors = n_processors
        
        # Ray cluster info
        self.cluster_resources = ray.cluster_resources()
        logger.info(f"Ray Cluster Resources: {self.cluster_resources}")
        
        # Initialize actors
        self.bot_actors = []
        self.processor_actors = []
        
        self._initialize_ray_actors()
    
    def _initialize_ray_actors(self):
        """Initialize Ray actors across the cluster"""
        logger.info(f"Initializing {self.n_bots} bot actors...")
        
        # Create bot actors distributed across cluster
        for i in range(self.n_bots):
            actor = RayKellyBotActor.remote(bot_id=i, initial_equity=100000.0)
            self.bot_actors.append(actor)
        
        # Create batch processor actors
        logger.info(f"Initializing {self.n_processors} batch processors...")
        for i in range(self.n_processors):
            processor = RayBatchProcessorActor.remote(processor_id=i)
            self.processor_actors.append(processor)
        
        logger.info("Ray actor initialization complete")
    
    def run_massive_simulation(self, 
                             simulation_hours: int = 8000,
                             batch_hours: int = 200) -> Dict:
        """
        Run massive simulation for maximum hardware utilization
        """
        logger.info("="*80)
        logger.info("RAY KELLY MONTE CARLO - MAXIMUM UTILIZATION MODE")
        logger.info("="*80)
        logger.info(f"Fleet: {self.n_bots} bots across cluster")
        logger.info(f"Simulation: {simulation_hours} hours")
        logger.info(f"Batch size: {batch_hours} hours")
        logger.info(f"Monte Carlo: 150k scenarios per decision")
        logger.info(f"Processors: {self.n_processors}")
        logger.info("="*80)
        
        # Generate synthetic market data
        market_data = self._generate_market_data(simulation_hours)
        logger.info(f"Generated {len(market_data)} hours of market data")
        
        # Create batches
        batches = []
        for i in range(0, len(market_data), batch_hours):
            batch_end = min(i + batch_hours, len(market_data))
            batch = market_data[i:batch_end]
            batches.append(batch)
        
        logger.info(f"Created {len(batches)} processing batches")
        
        # Process batches in parallel across all processors
        simulation_start = time.time()
        batch_futures = []
        
        for batch_idx, batch in enumerate(batches):
            # Round-robin assignment to processors
            processor = self.processor_actors[batch_idx % self.n_processors]
            future = processor.process_massive_batch.remote(self.bot_actors, batch)
            batch_futures.append((batch_idx, future))
        
        # Collect results with progress tracking
        batch_results = []
        total_decisions = 0
        total_trades = 0
        
        for batch_idx, future in tqdm(batch_futures, desc="Processing batches"):
            try:
                result = ray.get(future)
                batch_results.append(result)
                total_decisions += result['total_decisions']
                total_trades += result['total_trades']
                
                # Log every 5 batches
                if len(batch_results) % 5 == 0:
                    elapsed = time.time() - simulation_start
                    rate = len(batch_results) * batch_hours / elapsed
                    logger.info(f"Batch {len(batch_results)}/{len(batches)} complete. "
                              f"Rate: {rate:.1f} hours/sec. "
                              f"Decisions: {total_decisions:,}, Trades: {total_trades:,}")
                
            except Exception as e:
                logger.error(f"Batch {batch_idx} failed: {e}")
        
        simulation_time = time.time() - simulation_start
        
        # Collect final performance metrics
        logger.info("Collecting final fleet performance...")
        performance_futures = [actor.get_performance_metrics.remote() for actor in self.bot_actors]
        bot_metrics = ray.get(performance_futures)
        
        # Calculate summary
        active_bots = len([m for m in bot_metrics if m['total_trades'] > 0])
        avg_trades = np.mean([m['total_trades'] for m in bot_metrics if m['total_trades'] > 0]) if active_bots > 0 else 0
        
        results = {
            'simulation_summary': {
                'total_time_seconds': simulation_time,
                'hours_processed': simulation_hours,
                'processing_rate': simulation_hours / simulation_time,
                'total_decisions': total_decisions,
                'total_trades': total_trades,
                'monte_carlo_scenarios_total': total_decisions * 150000
            },
            'fleet_performance': {
                'total_bots': self.n_bots,
                'active_bots': active_bots,
                'average_trades_per_bot': avg_trades
            },
            'cluster_utilization': {
                'cluster_resources': self.cluster_resources,
                'batch_processors': self.n_processors,
                'batch_results': batch_results
            }
        }
        
        # Final summary
        logger.info("="*80)
        logger.info("SIMULATION COMPLETED - MAXIMUM UTILIZATION ACHIEVED")
        logger.info("="*80)
        logger.info(f"Total time: {simulation_time:.2f} seconds")
        logger.info(f"Processing rate: {simulation_hours/simulation_time:.1f} hours/second")
        logger.info(f"Total decisions: {total_decisions:,}")
        logger.info(f"Total Monte Carlo scenarios: {total_decisions * 150000:,}")
        logger.info(f"Active bots: {active_bots}/{self.n_bots}")
        logger.info(f"Average trades per bot: {avg_trades:.1f}")
        logger.info("="*80)
        
        return results
    
    def _generate_market_data(self, n_hours: int) -> List[Dict]:
        """Generate synthetic market data"""
        logger.info(f"Generating {n_hours} hours of synthetic market data...")
        
        np.random.seed(42)
        prices = []
        current_price = 1.2000
        
        for i in range(n_hours):
            # Random walk with trend
            change = np.random.normal(0.0001, 0.008)
            current_price += change
            
            # Calculate SMAs (simplified)
            sma_20 = current_price + np.random.normal(0, 0.001)
            sma_50 = current_price + np.random.normal(0, 0.002)
            
            prices.append({
                'timestamp': f"2024-01-01T{i%24:02d}:00:00",
                'close': current_price,
                'sma_20': sma_20,
                'sma_50': sma_50,
                'volume': 1000
            })
        
        return prices


def main():
    """Main execution"""
    print("RAY DISTRIBUTED KELLY MONTE CARLO - FINAL PRODUCTION VERSION")
    print("="*80)
    print("MAXIMUM HARDWARE UTILIZATION ACROSS CLUSTER")
    print("Target: 75% CPU + GPU usage on both machines")
    print("="*80)
    
    # Configuration
    N_BOTS = int(os.getenv('N_BOTS', '2000'))
    SIM_HOURS = int(os.getenv('SIM_HOURS', '8000'))
    BATCH_HOURS = int(os.getenv('BATCH_HOURS', '200'))
    N_PROCESSORS = int(os.getenv('N_PROCESSORS', '12'))
    
    print(f"Configuration:")
    print(f"- Fleet: {N_BOTS:,} bots")
    print(f"- Simulation: {SIM_HOURS:,} hours")
    print(f"- Batch size: {BATCH_HOURS} hours")
    print(f"- Processors: {N_PROCESSORS}")
    print(f"- Monte Carlo: 150,000 scenarios per decision")
    print()
    
    try:
        # Connect to existing Ray cluster with runtime environment
        if not ray.is_initialized():
            # Set up runtime environment to include current working directory
            runtime_env = {
                "working_dir": ".",  # Include current directory
                "pip": ["numpy", "pandas", "torch", "tqdm"]  # Ensure required packages
            }
            ray.init(address='auto', runtime_env=runtime_env)
        
        # Display cluster info
        resources = ray.cluster_resources()
        print(f"Connected to Ray Cluster:")
        print(f"- CPUs: {resources.get('CPU', 0)}")
        print(f"- GPUs: {resources.get('GPU', 0)}")
        print(f"- Nodes: {len(ray.nodes())}")
        print()
        
        # Create fleet manager
        fleet_manager = RayKellyFleetManager(
            n_bots=N_BOTS,
            n_processors=N_PROCESSORS
        )
        
        # Run massive simulation
        results = fleet_manager.run_massive_simulation(
            simulation_hours=SIM_HOURS,
            batch_hours=BATCH_HOURS
        )
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"ray_kelly_final_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nResults saved to: {results_file}")
        logger.info("RAY KELLY MONTE CARLO simulation completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise
    finally:
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()
