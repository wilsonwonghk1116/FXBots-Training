#!/usr/bin/env python3
"""
STANDALONE PC TRAINING LAUNCHER
Simple script to start forex bot training on single PC
Target: 100% GPU VRAM, 75% GPU usage, 80 CPU threads at 75%
"""

import os
import sys
import time
import torch
import subprocess
import logging
import multiprocessing
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_system_requirements():
    """Check if system meets requirements for standalone training"""
    logger.info("🔍 Checking system requirements...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        logger.error("❌ Python 3.8+ required")
        return False
    logger.info("✅ Python version OK")
    
    # Check GPU availability
    if not torch.cuda.is_available():
        logger.warning("⚠️  No CUDA GPU detected - will use CPU only")
    else:
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"✅ GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    
    # Check CPU cores
    cpu_count = multiprocessing.cpu_count()
    logger.info(f"✅ CPU cores: {cpu_count}")
    
    # Check required packages
    required_packages = ['torch', 'talib', 'pandas', 'numpy', 'psutil', 'gputil']
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package}")
        except ImportError:
            logger.error(f"❌ {package} - MISSING")
            missing.append(package)
    
    if missing:
        logger.error(f"❌ Missing packages: {missing}")
        logger.info("📦 Installing missing packages automatically...")
        
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing)
            logger.info("✅ Missing packages installed successfully")
            
            # Verify installation
            for package in missing:
                try:
                    __import__(package)
                    logger.info(f"✅ {package} now available")
                except ImportError:
                    logger.warning(f"⚠️  {package} installation may have failed, but continuing...")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to install packages automatically: {e}")
            logger.info("📋 Please install manually: pip install " + " ".join(missing))
            
            # Allow continuing without gputil if it's the only missing package
            if missing == ['gputil']:
                logger.warning("⚠️  Continuing without gputil (GPU monitoring will be limited)")
                return True
            
            return False
    
    return True

def optimize_system_for_standalone():
    """Apply system optimizations for standalone training"""
    logger.info("⚡ Applying standalone optimizations...")
    
    if torch.cuda.is_available():
        # GPU optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Set thread count for 75% CPU utilization (all 80 threads)
        target_threads = 80
        torch.set_num_threads(target_threads)
        logger.info(f"🧵 Set {target_threads} CPU threads (75% utilization)")
        
        # High priority process
        try:
            os.nice(-10)
            logger.info("⚡ Set high process priority")
        except:
            logger.warning("⚠️  Could not set high priority (run as admin for better performance)")
        
        logger.info("✅ Standalone optimizations applied")
    else:
        logger.warning("⚠️  Limited optimizations without GPU")

def check_training_files():
    """Check if required training files exist"""
    logger.info("📁 Checking training files...")
    
    required_files = [
        'run_production_forex_trainer.py',
        'data/EURUSD_H1.csv'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
            logger.error(f"❌ Missing: {file_path}")
        else:
            logger.info(f"✅ Found: {file_path}")
    
    if missing_files:
        logger.error("❌ Required files missing!")
        return False
    
    return True

def start_standalone_training():
    """Start the standalone training process"""
    logger.info("🚀 Starting STANDALONE forex bot training...")
    logger.info("🎯 Configuration: 100% GPU VRAM, 75% GPU usage, 80 CPU threads at 75%")
    
    # Import and run the production trainer with standalone settings
    try:
        # Import the training system
        from run_production_forex_trainer import (
            ProductionForexTrainer,
            monitor_performance
        )
        
        # Initialize trainer
        trainer = ProductionForexTrainer()
        
        # Override settings for STANDALONE
        logger.info("🔧 Configuring for STANDALONE mode...")
        
        # Conservative population for standalone
        original_population = trainer.population_size
        trainer.population_size = min(5000, max(2000, original_population // 2))
        
        # Conservative generations
        trainer.generations = 200
        
        # Set CPU and GPU targets for standalone (if supported by trainer)
        if hasattr(trainer, 'cpu_threads'):
            trainer.cpu_threads = 80
        if hasattr(trainer, 'cpu_utilization'):
            trainer.cpu_utilization = 75
        if hasattr(trainer, 'gpu_vram_target'):
            trainer.gpu_vram_target = 100
        if hasattr(trainer, 'gpu_usage_target'):
            trainer.gpu_usage_target = 75
        
        # Update standalone configuration log
        logger.info(f"📊 STANDALONE CONFIGURATION:")
        logger.info(f"   🤖 Population: {trainer.population_size} bots")
        logger.info(f"   🏆 Generations: {trainer.generations}")
        logger.info(f"   🔥 GPU Target: 100% VRAM, 75% usage")
        logger.info(f"   🧵 CPU Target: useing 80 workers out of total 80 CPU threads at 85% CPU thread processing resource")
        
        # Create population
        population = trainer.create_population()
        
        best_overall_score = 0
        best_champion_data = None
        
        # Training loop
        for generation in range(trainer.generations):
            logger.info(f"\n🚀 === STANDALONE GENERATION {generation + 1}/{trainer.generations} ===")
            
            # Monitor performance
            try:
                perf_data = monitor_performance()
                logger.info(f"📊 Performance: GPU {perf_data.get('gpu_util', 0):.1f}%, CPU {perf_data.get('cpu_percent', 0):.1f}%")
            except:
                perf_data = {}
            
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
                
                import json
                champion_analysis = {
                    'timestamp': timestamp,
                    'generation': generation + 1,
                    'champion_data': best_champion_data,
                    'performance_data': perf_data,
                    'training_config': {
                        'mode': 'standalone',
                        'population_size': trainer.population_size,
                        'generations': trainer.generations,
                        'gpu_target': '100%',
                        'cpu_target': '75%'
                    }
                }
                
                with open(analysis_filename, 'w') as f:
                    json.dump(champion_analysis, f, indent=2)
                
                logger.info(f"💾 Champion saved: {champion_filename}")
            
            # Evolution
            if generation < trainer.generations - 1:
                population = trainer.evolve_population(population, results)
            
            progress = (generation + 1) / trainer.generations * 100
            logger.info(f"📈 Training Progress: {progress:.1f}%")
        
        logger.info(f"\n🏁 === STANDALONE TRAINING COMPLETE ===")
        if best_champion_data:
            logger.info(f"🏆 FINAL CHAMPION RESULTS:")
            logger.info(f"   Strategy: {best_champion_data['strategy_type']}")
            logger.info(f"   Final Balance: ${best_champion_data['final_balance']:.2f}")
            logger.info(f"   Championship Score: {best_champion_data['championship_score']:.2f}")
            logger.info(f"   Win Rate: {best_champion_data['win_rate']:.3f}")
        
        logger.info("\n🎉 STANDALONE FOREX TRAINING COMPLETED! 🎉")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def main():
    """Main standalone training launcher"""
    logger.info("🚀 === STANDALONE PC FOREX TRAINER LAUNCHER ===")
    logger.info("🎯 TARGET: 100% GPU VRAM, 75% GPU usage, 80 CPU threads at 75%")
    
    # Check system requirements
    if not check_system_requirements():
        logger.error("❌ System requirements not met")
        return
    
    # Check training files
    if not check_training_files():
        logger.error("❌ Required training files missing")
        return
    
    # Apply optimizations
    optimize_system_for_standalone()
    
    # Confirm start
    logger.info("\n📋 READY TO START STANDALONE TRAINING")
    logger.info("⚠️  This will use significant system resources")
    
    response = input("Start training? (y/n): ").lower().strip()
    if response != 'y':
        logger.info("❌ Training cancelled by user")
        return
    
    # Start training
    try:
        success = start_standalone_training()
        if success:
            logger.info("🎉 Training completed successfully!")
        else:
            logger.error("❌ Training failed!")
    except KeyboardInterrupt:
        logger.info("🛑 Training interrupted by user")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main() 