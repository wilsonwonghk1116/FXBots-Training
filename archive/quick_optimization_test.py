#!/usr/bin/env python3
"""
Quick test of optimized Kelly Monte Carlo system
Tests with fewer bots to verify high resource utilization
"""

import os
import sys
import time
import numpy as np
import multiprocessing as mp
import torch
from kelly_monte_bot import (
    KellyMonteBot, BotFleetManager, TradingParameters, logger
)

def quick_optimization_test():
    """Quick test of the optimized system"""
    print("QUICK OPTIMIZATION TEST")
    print("=" * 50)
    
    # Test parameters for quick verification
    N_BOTS = 100  # Smaller fleet for quick test
    TEST_HOURS = 10
    
    # Enhanced parameters for maximum utilization
    params = TradingParameters()
    params.monte_carlo_scenarios = 50000  # Still use high scenario count
    
    print(f"Test Configuration:")
    print(f"- Bots: {N_BOTS}")
    print(f"- Monte Carlo scenarios: {params.monte_carlo_scenarios:,}")
    print(f"- CPU cores: {mp.cpu_count()}")
    print(f"- GPU available: {torch.cuda.is_available()}")
    print(f"- Test hours: {TEST_HOURS}")
    print()
    
    # Initialize fleet
    logger.info("Initializing test fleet...")
    fleet_manager = BotFleetManager(
        n_bots=N_BOTS,
        initial_equity=100000.0,
        params=params
    )
    
    # Test Monte Carlo performance
    logger.info("Testing Monte Carlo performance...")
    start_time = time.time()
    
    # Create a sample trading decision for all bots
    import pandas as pd
    current_price = 1.2000
    market_data = pd.Series({
        'close': current_price,
        'sma_20': 1.1980,
        'sma_50': 1.1950,
        'volume': 1000
    })
    
    # This should trigger high CPU/GPU utilization
    decisions = fleet_manager.run_parallel_decisions(
        current_price=current_price,
        market_data=market_data,
        timestamp=pd.Timestamp.now()
    )
    
    decision_time = time.time() - start_time
    valid_decisions = sum(1 for d in decisions if d is not None)
    
    print(f"\nPerformance Results:")
    print(f"- Decision time: {decision_time:.2f} seconds")
    print(f"- Valid decisions: {valid_decisions}/{N_BOTS}")
    print(f"- Decisions per second: {len(decisions)/decision_time:.1f}")
    print(f"- Total Monte Carlo scenarios: {valid_decisions * params.monte_carlo_scenarios:,}")
    print(f"- Scenarios per second: {valid_decisions * params.monte_carlo_scenarios / decision_time:,.0f}")
    
    if torch.cuda.is_available():
        print(f"- GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"- GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    print("\nThis should have generated high CPU/GPU utilization!")
    print("Check your system monitor for resource usage.")
    
    return fleet_manager

if __name__ == "__main__":
    quick_optimization_test()
