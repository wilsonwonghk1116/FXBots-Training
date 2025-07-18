#!/usr/bin/env python3
"""
FIX CONDA RAY ENVIRONMENT
Script to properly install all required packages in ray_env conda environment
"""

import sys
import subprocess
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_conda_environment():
    """Check if we're in a conda environment"""
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'base')
    logger.info(f"🐍 Current conda environment: {conda_env}")
    
    if 'ray' in conda_env.lower():
        logger.info("✅ Detected Ray environment")
        return True
    else:
        logger.warning("⚠️  Not in ray_env - consider activating it first")
        return True  # Continue anyway

def install_conda_packages():
    """Install packages via conda (preferred for ray_env)"""
    
    conda_packages = [
        'pytorch',
        'pandas', 
        'numpy',
        'psutil'
    ]
    
    logger.info("📦 Installing packages via conda...")
    
    for package in conda_packages:
        try:
            logger.info(f"📦 Installing {package} via conda...")
            subprocess.check_call(['conda', 'install', '-y', package])
            logger.info(f"✅ {package} installed via conda")
        except Exception as e:
            logger.warning(f"⚠️  Failed to install {package} via conda: {e}")

def install_talib():
    """Install TA-Lib with proper method"""
    logger.info("📊 Installing TA-Lib...")
    
    try:
        # Method 1: Try conda-forge
        logger.info("📦 Trying conda-forge for TA-Lib...")
        subprocess.check_call(['conda', 'install', '-y', '-c', 'conda-forge', 'ta-lib'])
        logger.info("✅ TA-Lib installed via conda-forge")
        return True
    except:
        logger.warning("⚠️  conda-forge failed, trying pip...")
    
    try:
        # Method 2: Try pip
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'TA-Lib'])
        logger.info("✅ TA-Lib installed via pip")
        return True
    except:
        logger.warning("⚠️  pip failed, trying alternative...")
    
    try:
        # Method 3: Try specific talib version
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', 'TA-Lib==0.4.28'])
        logger.info("✅ TA-Lib 0.4.28 installed")
        return True
    except:
        logger.error("❌ All TA-Lib installation methods failed")
        return False

def install_pip_packages():
    """Install remaining packages via pip"""
    
    pip_packages = [
        'gputil',
        'gymnasium'
    ]
    
    logger.info("📦 Installing additional packages via pip...")
    
    for package in pip_packages:
        try:
            logger.info(f"📦 Installing {package} via pip...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            logger.info(f"✅ {package} installed via pip")
        except Exception as e:
            logger.warning(f"⚠️  Failed to install {package}: {e}")

def verify_installations():
    """Verify all packages are working"""
    
    packages_to_test = {
        'torch': 'PyTorch',
        'pandas': 'Pandas', 
        'numpy': 'NumPy',
        'talib': 'TA-Lib',
        'psutil': 'psutil',
        'gputil': 'GPUtil',
        'ray': 'Ray'
    }
    
    logger.info("🔍 Verifying package installations...")
    
    success_count = 0
    for package, name in packages_to_test.items():
        try:
            __import__(package)
            logger.info(f"✅ {name} working")
            success_count += 1
        except ImportError:
            logger.error(f"❌ {name} not available")
    
    logger.info(f"📊 {success_count}/{len(packages_to_test)} packages working")
    
    if success_count >= 6:  # At least 6 out of 7 working
        logger.info("🎉 Environment setup successful!")
        return True
    else:
        logger.warning("⚠️  Some packages missing - training may have limited functionality")
        return False

def main():
    """Main environment setup function"""
    logger.info("🔧 === FIXING CONDA RAY ENVIRONMENT ===")
    
    # Check environment
    check_conda_environment()
    
    # Install packages
    logger.info("\n📦 === INSTALLING PACKAGES ===")
    install_conda_packages()
    install_talib()
    install_pip_packages()
    
    # Verify installation
    logger.info("\n🔍 === VERIFYING INSTALLATION ===")
    success = verify_installations()
    
    if success:
        logger.info("\n🎉 === ENVIRONMENT READY ===")
        logger.info("📋 You can now run:")
        logger.info("   python start_standalone_training.py")
        logger.info("   python start_cluster_training.py")
    else:
        logger.info("\n⚠️  === PARTIAL SUCCESS ===")
        logger.info("📋 Try running training scripts - they may work with limited functionality")
    
    # Show final command suggestions
    logger.info("\n💡 === ALTERNATIVE MANUAL COMMANDS ===")
    logger.info("If automated installation failed, try these commands manually:")
    logger.info("conda install -y pytorch pandas numpy psutil")
    logger.info("conda install -y -c conda-forge ta-lib")
    logger.info("pip install gputil gymnasium ray")

if __name__ == "__main__":
    main() 