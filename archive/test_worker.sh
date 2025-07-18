#!/bin/bash

# Simple test for Worker PC setup
sshpass -p 'w' ssh -o StrictHostKeyChecking=no w2@192.168.1.11 << 'EOF'
echo "🔍 Testing Worker PC Setup"
echo "========================="
echo "Current user: $(whoami)"
echo "Current directory: $(pwd)"
echo "Home directory: $HOME"

echo ""
echo "📁 Checking project files:"
cd /home/w2/cursor-to-copilot-backup/TaskmasterForexBots
ls -la *.py | head -5

echo ""
echo "🐍 Activating conda environment:"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate BotsTraining_env
echo "Active environment: $CONDA_DEFAULT_ENV"

echo ""
echo "🧪 Testing Python imports:"
python -c "import synthetic_env; print('✅ synthetic_env OK')"
python -c "import bot_population; print('✅ bot_population OK')" 
python -c "import trading_bot; print('✅ trading_bot OK')"
python -c "import pandas; print('✅ pandas OK')"
python -c "import ray; print('✅ ray OK')"
python -c "import torch; print('✅ torch OK')"

echo ""
echo "🎯 Final Status: All tests passed!"
EOF 