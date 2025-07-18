#!/bin/bash

echo "🚀 Syncing Project Files to Worker PC"
echo "====================================="

# Configuration
WORKER_IP="192.168.1.11"
WORKER_USER="w2"  # FIXED: Worker PC username is w2, not w1!
PROJECT_DIR="/home/w2/cursor-to-copilot-backup/TaskmasterForexBots"
REMOTE_DIR="${WORKER_USER}@${WORKER_IP}:${PROJECT_DIR}"

echo "📡 Worker PC: ${WORKER_IP}"
echo "📁 Project Directory: ${PROJECT_DIR}"
echo "👤 User: ${WORKER_USER}"

# Check if we can reach the worker PC
echo ""
echo "🔍 Testing connection to Worker PC..."
if ! ping -c 1 -W 2 ${WORKER_IP} > /dev/null 2>&1; then
    echo "❌ ERROR: Cannot reach Worker PC at ${WORKER_IP}"
    echo "   Please check network connection and IP address"
    exit 1
fi
echo "✅ Worker PC is reachable"

# Create remote directory if it doesn't exist
echo ""
echo "📁 Creating project directory on Worker PC..."
ssh ${WORKER_USER}@${WORKER_IP} "mkdir -p ${PROJECT_DIR}" 2>/dev/null || {
    echo "❌ ERROR: Cannot SSH to Worker PC. Please check:"
    echo "   1. SSH key authentication is set up"
    echo "   2. Username is correct (${WORKER_USER})"
    echo "   3. Worker PC allows SSH connections"
    exit 1
}
echo "✅ Project directory ready on Worker PC"

# Required Python files to sync
PYTHON_FILES=(
    "synthetic_env.py"
    "bot_population.py"
    "trading_bot.py"
    "config.py"
    "indicators.py"
    "predictors.py"
    "reward.py"
    "utils.py"
    "champion_analysis.py"
    "checkpoint_utils.py"
    "run_stable_85_percent_trainer.py"
)

# Required directories
DIRECTORIES=(
    "data/"
    "models/"
    "checkpoints/"
)

echo ""
echo "📤 Syncing Python files..."
for file in "${PYTHON_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "   📄 Copying $file..."
        scp "$file" "${REMOTE_DIR}/" || {
            echo "   ❌ Failed to copy $file"
            exit 1
        }
    else
        echo "   ⚠️  Warning: $file not found locally"
    fi
done

echo ""
echo "📤 Syncing directories..."
for dir in "${DIRECTORIES[@]}"; do
    if [ -d "$dir" ]; then
        echo "   📁 Copying $dir..."
        scp -r "$dir" "${REMOTE_DIR}/" || {
            echo "   ❌ Failed to copy $dir"
            exit 1
        }
    else
        echo "   ⚠️  Warning: $dir not found locally"
        ssh ${WORKER_USER}@${WORKER_IP} "mkdir -p ${PROJECT_DIR}/$dir"
    fi
done

echo ""
echo "🔄 Syncing conda environment requirements..."
if [ -f "requirements.txt" ]; then
    scp "requirements.txt" "${REMOTE_DIR}/" || echo "   ⚠️  Failed to copy requirements.txt"
fi

echo ""
echo "🐍 Installing Python dependencies on Worker PC..."
ssh ${WORKER_USER}@${WORKER_IP} "cd ${PROJECT_DIR} && \
    source ~/miniconda3/etc/profile.d/conda.sh && \
    conda activate BotsTraining_env && \
    pip install -r requirements.txt 2>/dev/null || echo 'Requirements install attempted'"

echo ""
echo "🔍 Verifying files on Worker PC..."
ssh ${WORKER_USER}@${WORKER_IP} "cd ${PROJECT_DIR} && \
    echo 'Python files:' && \
    ls -la *.py | head -10 && \
    echo 'Data directory:' && \
    ls -la data/ 2>/dev/null || echo 'No data directory' && \
    echo 'Environment check:' && \
    source ~/miniconda3/etc/profile.d/conda.sh && \
    conda activate BotsTraining_env && \
    python -c 'import synthetic_env; print(\"✅ synthetic_env import successful\")' 2>/dev/null || echo '❌ synthetic_env import failed'"

echo ""
echo "🎉 File synchronization complete!"
echo ""
echo "📋 Next steps:"
echo "   1. Verify all files are present on Worker PC"
echo "   2. Check that the conda environment is properly activated on Worker PC"  
echo "   3. Run the training command: RAY_CLUSTER=1 python run_stable_85_percent_trainer.py"
echo ""
echo "🔧 If issues persist, manually SSH to Worker PC and run:"
echo "   ssh ${WORKER_USER}@${WORKER_IP}"
echo "   cd ${PROJECT_DIR}"
echo "   conda activate BotsTraining_env"
echo "   python -c 'import synthetic_env; print(\"Import successful\")'" 