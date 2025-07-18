#!/usr/bin/env python3
"""
Direct Training Execution Script
===============================
This script directly launches the forex bot training, assuming Ray cluster is already running.
Use this if you've already verified cluster connectivity.

Usage:
    python direct_training.py [test|full]

Author: AI Assistant
Date: July 13, 2025
"""

import sys
import os
import time
from datetime import datetime

# Add project directory to path
sys.path.append('/home/w1/cursor-to-copilot-backup/TaskmasterForexBots')

def log(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

def run_test_training():
    """Run quick test training (5 generations)"""
    log("🧪 STARTING TEST TRAINING")
    log("Duration: ~5-10 minutes")
    log("Parameters: 5 generations, 10 bots, 50 episodes, 100 steps")
    
    try:
        # Set optimal environment variables
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        os.environ['OMP_NUM_THREADS'] = '40'
        os.environ['MKL_NUM_THREADS'] = '40'
        
        # Connect to Ray cluster
        import ray
        ray.init(address='ray://192.168.1.10:10001')
        
        log("✅ Connected to Ray cluster")
        log(f"📊 Cluster resources: {ray.cluster_resources()}")
        
        # Import and run training
        from comprehensive_trading_system import ComprehensiveTradingSystem
        
        log("🎯 Initializing trading system...")
        trading_system = ComprehensiveTradingSystem()
        
        log("🚀 Starting training...")
        results = trading_system.run_training(
            generations=5,           # Quick test
            population_size=10,      # Small population
            episodes_per_gen=50,     # Fewer episodes
            steps_per_episode=100    # Shorter episodes
        )
        
        log("✅ TEST TRAINING COMPLETED!")
        log(f"📈 Results: {results}")
        
        return True
        
    except Exception as e:
        log(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        try:
            ray.shutdown()
        except:
            pass

def run_full_training():
    """Run full scale training (200 generations)"""
    log("🚀 STARTING FULL SCALE TRAINING")
    log("Duration: ~3-4 hours")
    log("Parameters: 200 generations, 20 bots, 1000 episodes, 1000 steps")
    
    try:
        # Set optimal environment variables
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        os.environ['OMP_NUM_THREADS'] = '40'
        os.environ['MKL_NUM_THREADS'] = '40'
        
        # Connect to Ray cluster
        import ray
        ray.init(address='ray://192.168.1.10:10001')
        
        log("✅ Connected to Ray cluster")
        log(f"📊 Cluster resources: {ray.cluster_resources()}")
        
        # Import and run training
        from comprehensive_trading_system import ComprehensiveTradingSystem
        
        log("🎯 Initializing trading system...")
        trading_system = ComprehensiveTradingSystem()
        
        log("🚀 Starting full scale training...")
        log("🌐 Monitor progress at: http://192.168.1.10:8265")
        
        results = trading_system.run_training(
            generations=200,         # Full training
            population_size=20,      # Full population
            episodes_per_gen=1000,   # Full episodes
            steps_per_episode=1000   # Full steps
        )
        
        log("🎉 FULL SCALE TRAINING COMPLETED!")
        log(f"📈 Final results: {results}")
        
        return True
        
    except Exception as e:
        log(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        try:
            ray.shutdown()
        except:
            pass

def main():
    """Main execution"""
    print("🤖 DIRECT FOREX BOT TRAINING")
    print("=" * 40)
    
    # Check command line arguments
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        print("Select training mode:")
        print("  [1] Test (5 generations, ~5-10 minutes)")
        print("  [2] Full (200 generations, ~3-4 hours)")
        choice = input("Choice (1/2): ").strip()
        mode = "test" if choice == "1" else "full" if choice == "2" else "test"
    
    if mode == "test":
        log("🧪 Test mode selected")
        success = run_test_training()
    elif mode == "full":
        log("🚀 Full mode selected")
        confirm = input("⚠️  This will take 3-4 hours. Continue? (y/N): ").strip().lower()
        if confirm == 'y':
            success = run_full_training()
        else:
            log("❌ Training cancelled")
            return
    else:
        print("❌ Invalid mode. Use 'test' or 'full'")
        return
    
    if success:
        print("\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
    else:
        print("\n❌ Training failed")

if __name__ == "__main__":
    main()
