#!/usr/bin/env python3
"""
INSTALL MISSING PACKAGES
Simple script to install any missing packages for forex bot training
"""

import sys
import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def install_packages():
    """Install commonly missing packages"""
    
    packages_to_install = [
        'gputil',
        'ray',
        'torch', 
        'talib',
        'pandas',
        'numpy',
        'psutil',
        'gymnasium'
    ]
    
    logger.info("🔧 Installing commonly missing packages...")
    
    for package in packages_to_install:
        try:
            # Check if already installed
            __import__(package)
            logger.info(f"✅ {package} already installed")
        except ImportError:
            # Install the package
            logger.info(f"📦 Installing {package}...")
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                logger.info(f"✅ {package} installed successfully")
            except Exception as e:
                logger.error(f"❌ Failed to install {package}: {e}")
    
    logger.info("🎉 Package installation complete!")
    logger.info("📋 You can now run:")
    logger.info("   python start_standalone_training.py")
    logger.info("   python start_cluster_training.py")

if __name__ == "__main__":
    install_packages() 