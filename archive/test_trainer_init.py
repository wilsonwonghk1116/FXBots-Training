#!/usr/bin/env python3
"""
Test the VRAMOptimizedTrainer initialization to find any issues
"""

import sys
import torch
import traceback

try:
    print("Testing VRAMOptimizedTrainer initialization...")
    
    # Import components
    from run_smart_real_training import VRAMOptimizedTrainer, SmartTradingBot, SmartForexEnvironment
    print("✅ Imports successful")
    
    # Test model creation
    model = SmartTradingBot()
    print("✅ Model creation successful")
    
    # Test environment creation
    env = SmartForexEnvironment()
    print("✅ Environment creation successful")
    
    # Test trainer initialization with smaller population
    print("Initializing trainer with small population...")
    trainer = VRAMOptimizedTrainer(population_size=10, target_vram_percent=0.5)
    print("✅ Trainer initialization successful")
    
    # Test population creation
    print("Creating small population...")
    population = trainer.create_population()
    print(f"✅ Population created: {len(population)} bots")
    
    print("\n🎉 All components working! The issue might be with the large population size or other training parameters.")
    
except Exception as e:
    print(f"❌ Error occurred: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
