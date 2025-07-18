#!/usr/bin/env python3
"""
Kelly Monte Carlo Bot Fleet Demo Runner
A smaller demo version to test the Kelly Monte Carlo bot system
"""

import sys
import os
import time
import logging
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from kelly_monte_bot import (
    KellyMonteBot, BotFleetManager, TradingParameters,
    DataManager, MonteCarloEngine, KellyCalculator
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_demo_fleet(n_bots: int = 100, simulation_hours: int = 200):
    """
    Run a smaller demo of the Kelly Monte Carlo bot fleet
    
    Args:
        n_bots: Number of bots in demo fleet
        simulation_hours: Number of hours to simulate
        
    Returns:
        BotFleetManager instance with results
    """
    logger.info(f"Starting Kelly Monte Carlo demo with {n_bots} bots")
    logger.info(f"Demo duration: {simulation_hours} hours")
    
    # Configure trading parameters for demo
    params = TradingParameters(
        max_risk_per_trade=0.02,    # 2% max risk
        stop_loss_pips=30.0,        # 30 pip SL
        take_profit_pips=60.0,      # 60 pip TP (2:1 RR)
        monte_carlo_scenarios=100,  # Reduced for demo speed
        update_frequency=10,        # Update every 10 trades
        rolling_history_size=200    # 200-period history
    )
    
    # Initialize demo fleet
    fleet_manager = BotFleetManager(
        n_bots=n_bots,
        initial_equity=100000.0,  # $100k per bot
        params=params
    )
    
    # Load market data
    data_manager = DataManager()
    market_data = data_manager.load_h1_data("EURUSD")
    
    logger.info(f"Loaded {len(market_data)} hours of market data")
    
    # Run demo simulation
    start_time = time.time()
    trades_executed = 0
    
    # Use subset of data for demo
    demo_data = market_data.tail(simulation_hours)
    logger.info(f"Running demo on {len(demo_data)} hours of data...")
    
    for i, (timestamp, row) in enumerate(demo_data.iterrows()):
        if i % 50 == 0:
            logger.info(f"Demo progress: {i+1}/{len(demo_data)} hours - Trades: {trades_executed}")
        
        current_price = row['close']
        
        # Get trading decisions from all bots
        decisions = fleet_manager.run_parallel_decisions(
            current_price=current_price,
            market_data=row,
            timestamp=timestamp
        )
        
        # Execute trades for bots with decisions
        for j, decision in enumerate(decisions):
            if decision is not None:
                try:
                    bot = fleet_manager.bots[j]
                    if bot.current_position is None:
                        bot.execute_trade(decision)
                        trades_executed += 1
                except Exception as e:
                    logger.error(f"Error executing trade for bot {j}: {e}")
        
        # Close trades randomly (simplified)
        for bot in fleet_manager.bots:
            if (bot.current_position is not None and 
                np.random.random() < 0.15):  # 15% chance to close
                try:
                    # Simulate market movement
                    price_change = np.random.normal(0, 0.0008) * current_price
                    exit_price = current_price + price_change
                    
                    # Determine exit reason
                    entry_price = bot.current_position['entry_price']
                    signal = bot.current_position['signal']
                    
                    if signal == 'BUY':
                        pips_moved = (exit_price - entry_price) / params.pip_value
                    else:
                        pips_moved = (entry_price - exit_price) / params.pip_value
                    
                    if pips_moved >= params.take_profit_pips:
                        exit_reason = "TAKE_PROFIT"
                    elif pips_moved <= -params.stop_loss_pips:
                        exit_reason = "STOP_LOSS"
                    else:
                        exit_reason = "NATURAL"
                    
                    bot.close_trade(exit_price, exit_reason)
                
                except Exception as e:
                    logger.error(f"Error closing trade for bot {bot.bot_id}: {e}")
    
    simulation_time = time.time() - start_time
    logger.info(f"Demo completed in {simulation_time:.2f} seconds")
    logger.info(f"Total trades executed: {trades_executed:,}")
    
    return fleet_manager

def analyze_demo_results(fleet_manager: BotFleetManager):
    """Analyze and display demo results"""
    logger.info("Analyzing demo results...")
    
    # Get fleet performance
    fleet_metrics = fleet_manager.get_fleet_performance()
    
    # Collect bot metrics
    bot_metrics = []
    for bot in fleet_manager.bots:
        metrics = bot.get_performance_metrics()
        if metrics:
            bot_metrics.append(metrics)
    
    # Print summary
    print("\n" + "="*60)
    print("KELLY MONTE CARLO BOT DEMO RESULTS")
    print("="*60)
    
    if fleet_metrics:
        print(f"Active Bots: {fleet_metrics.get('n_active_bots', 0)}")
        print(f"Total Trades: {fleet_metrics.get('total_trades', 0):,}")
        print(f"Fleet Win Rate: {fleet_metrics.get('fleet_win_rate', 0)*100:.1f}%")
        print(f"Total P&L: ${fleet_metrics.get('total_pnl', 0):,.2f}")
        print(f"Average Return: {fleet_metrics.get('average_return_pct', 0):.2f}%")
        print(f"Average Sharpe: {fleet_metrics.get('average_sharpe_ratio', 0):.2f}")
    
    if bot_metrics:
        returns = [m['total_return_pct'] for m in bot_metrics]
        profitable_bots = sum(1 for r in returns if r > 0)
        
        print(f"\nProfitable Bots: {profitable_bots}/{len(bot_metrics)} " +
              f"({profitable_bots/len(bot_metrics)*100:.1f}%)")
        print(f"Best Return: {max(returns):.2f}%")
        print(f"Worst Return: {min(returns):.2f}%")
        print(f"Avg Return: {np.mean(returns):.2f}%")
        print(f"Return Std: {np.std(returns):.2f}%")
    
    print("="*60)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"kelly_demo_results_{timestamp}.json"
    fleet_manager.save_results(results_file)
    logger.info(f"Demo results saved to {results_file}")
    
    return fleet_metrics, bot_metrics

def main():
    """Main demo execution"""
    print("Kelly Monte Carlo FOREX Bot Demo")
    print("=" * 40)
    
    # Demo configuration
    N_BOTS = 100          # Smaller fleet for demo
    DEMO_HOURS = 200      # Shorter simulation
    
    print(f"Demo Fleet Size: {N_BOTS} bots")
    print(f"Demo Duration: {DEMO_HOURS} hours")
    
    try:
        # Run demo fleet
        fleet_manager = run_demo_fleet(
            n_bots=N_BOTS,
            simulation_hours=DEMO_HOURS
        )
        
        # Analyze results
        fleet_metrics, bot_metrics = analyze_demo_results(fleet_manager)
        
        logger.info("Kelly Monte Carlo demo completed successfully!")
        
        return fleet_manager, fleet_metrics, bot_metrics
        
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise

if __name__ == "__main__":
    main()
