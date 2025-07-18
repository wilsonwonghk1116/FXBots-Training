import os

# CONFIGURATION
# General training parameters
GENERATIONS = 200
CHECKPOINT_INTERVAL = 10

# Population settings
# Total population will be sum of these
POP_SIZE_3090 = 1000 
POP_SIZE_3070 = 250

# Resource allocation for each node
GPU_3090 = 1
GPU_3070 = 1
CPU_3090 = 76
CPU_3070 = 14

# NEW: CPU Safety Governor
# This prevents us from using 100% of the assigned CPUs for simulation actors,
# reserving a buffer for the Raylet and OS to prevent heartbeat failures.
# 0.8 means we will use 80% of the allocated CPU cores for actors.
ACTOR_CPU_SCALE_FACTOR = 0.8

# Data paths - Fixed for multi-PC support
def get_project_root():
    """Get the correct project root based on which PC we're on"""
    # Try to determine which PC we're on and return appropriate path
    possible_paths = [
        "/home/w1/cursor-to-copilot-backup/TaskmasterForexBots",  # Head PC
        "/home/w2/cursor-to-copilot-backup/TaskmasterForexBots"   # Worker PC
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # Fallback to current file's directory
    return os.path.dirname(__file__)

PROJECT_ROOT = get_project_root()
EURUSD_H1_PATH = os.path.join(PROJECT_ROOT, 'data', 'EURUSD_H1.csv')

# File paths
CHECKPOINT_DIR = 'checkpoints/'
MODEL_DIR = 'models/' 