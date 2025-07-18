#!/usr/bin/env python3
"""
Fix Ray Workers Module Path Issue
This script ensures Ray workers can find our project modules
"""

import sys
import os

# Add the project directory to Python path so Ray workers can find modules
PROJECT_DIR = "/home/w2/cursor-to-copilot-backup/TaskmasterForexBots"
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
    print(f"✅ Added {PROJECT_DIR} to Python path")

# Also set the working directory for Ray workers
os.chdir(PROJECT_DIR)
print(f"✅ Changed working directory to {PROJECT_DIR}")

# Test imports to make sure everything works
try:
    import synthetic_env
    print("✅ synthetic_env import OK")
    
    import bot_population
    print("✅ bot_population import OK")
    
    import trading_bot
    print("✅ trading_bot import OK")
    
    print("🎉 All Ray worker imports should now work!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1) 