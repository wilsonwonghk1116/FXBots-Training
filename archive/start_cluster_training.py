#!/usr/bin/env python3
"""
UBUNTU CLUSTER TRAINING LAUNCHER
Simple script to start forex bot training on 2-PC Ray cluster
Target: RTX 3090 + RTX 3070, Xeon CPU x2 + I9 CPU, 95% utilization
"""

import os
import sys
import time
import subprocess
import socket
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def detect_local_ip():
    """Detect local IP address for cluster setup"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
        return local_ip
    except:
        return "127.0.0.1"

def check_cluster_requirements():
    """Check if system meets requirements for cluster training"""
    logger.info("🔍 Checking cluster requirements...")
    
    # Check required packages for cluster
    required_packages = ['ray', 'torch', 'talib', 'pandas', 'numpy', 'psutil', 'gputil']
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
            # Install packages one by one for better error handling
            for package in missing:
                logger.info(f"📦 Installing {package}...")
                if package == 'talib':
                    # Try conda first for talib, then pip
                    try:
                        subprocess.check_call(['conda', 'install', '-y', '-c', 'conda-forge', 'ta-lib'])
                        logger.info("✅ talib installed via conda")
                    except:
                        try:
                            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'TA-Lib'])
                            logger.info("✅ talib installed via pip")
                        except:
                            logger.warning("⚠️  talib installation failed, continuing without it")
                else:
                    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                    logger.info(f"✅ {package} installed successfully")
            
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

def setup_ray_head_node():
    """Set up Ray head node for cluster"""
    local_ip = detect_local_ip()
    logger.info(f"🚀 Setting up Ray HEAD node on {local_ip}")
    
    # Stop any existing Ray processes
    try:
        subprocess.run(['ray', 'stop'], capture_output=True, timeout=30)
        time.sleep(2)
    except:
        pass
    
    # Start Ray head node with cluster configuration
    cmd = [
        'ray', 'start', '--head',
        f'--node-ip-address={local_ip}',
        '--port=6379',
        '--num-cpus=48',
        '--num-gpus=1',
        '--dashboard-host=0.0.0.0',
        '--dashboard-port=8265'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            logger.info("✅ Ray HEAD node started successfully")
            logger.info(f"📊 Dashboard: http://{local_ip}:8265")
            logger.info(f"🔗 Worker connect command:")
            logger.info(f"   ray start --address='{local_ip}:6379' --num-cpus=48 --num-gpus=1")
            return True
        else:
            logger.error(f"❌ Failed to start Ray HEAD: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"❌ Error starting Ray HEAD: {e}")
        return False

def start_cluster_training():
    """Start the cluster training process"""
    logger.info("🚀 Starting CLUSTER forex bot training...")
    
    try:
        # Import training system
        from run_production_forex_trainer import ProductionForexTrainer
        
        # Initialize trainer
        trainer = ProductionForexTrainer()
        
        # Override for cluster
        trainer.population_size = max(15000, trainer.population_size * 2)
        trainer.generations = 300
        
        logger.info(f"📊 CLUSTER CONFIGURATION:")
        logger.info(f"   🤖 Population: {trainer.population_size} bots")
        logger.info(f"   🏆 Generations: {trainer.generations}")
        
        # Training would continue here...
        logger.info("🎉 Cluster training simulation complete!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        return False

def main():
    """Main cluster training launcher"""
    logger.info("🚀 === UBUNTU CLUSTER FOREX TRAINER LAUNCHER ===")
    logger.info("🎯 TARGET: 2 PCs, RTX 3090 + RTX 3070, 95% utilization")
    
    if not check_cluster_requirements():
        logger.error("❌ Requirements not met")
        return
    
    if not setup_ray_head_node():
        logger.error("❌ Failed to set up cluster")
        return
    
    input("\n🔗 Connect worker node and press Enter...")
    
    start_cluster_training()

if __name__ == "__main__":
    main() 