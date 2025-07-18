#!/bin/bash
# Ray Worker Environment Setup Script
cd /home/w2/cursor-to-copilot-backup/TaskmasterForexBots

# Activate conda environment
source /home/w2/miniconda3/etc/profile.d/conda.sh
conda activate BotsTraining_env

# Set environment variables
export PYTHONPATH="/home/w1/cursor-to-copilot-backup/TaskmasterForexBots:/home/w2/cursor-to-copilot-backup/TaskmasterForexBots:$PYTHONPATH"
export RAY_CLUSTER=1

# Test imports
echo "Testing Python imports..."
python3 -c "
import sys
print('Python executable:', sys.executable)
print('Python path:', sys.path[:3])

try:
    import synthetic_env
    print('✅ synthetic_env imported successfully')
except ImportError as e:
    print('❌ synthetic_env import failed:', e)

try:
    import ray
    print('✅ ray imported successfully')
    print('Ray version:', ray.__version__)
except ImportError as e:
    print('❌ ray import failed:', e)

try:
    import torch
    print('✅ torch imported successfully')
    print('Torch version:', torch.__version__)
    print('CUDA available:', torch.cuda.is_available())
except ImportError as e:
    print('❌ torch import failed:', e)
"

echo "Environment setup complete. Ready to start Ray worker."
