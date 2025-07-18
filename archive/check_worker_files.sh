#!/bin/bash

echo "🔍 Checking Worker PC Files Status"
echo "=================================="

WORKER_IP="192.168.1.11"
WORKER_USER="w2"
PROJECT_DIR="/home/w2/cursor-to-copilot-backup/TaskmasterForexBots"

echo "📡 Worker PC: ${WORKER_IP}"
echo "👤 User: ${WORKER_USER}"
echo "📁 Project Dir: ${PROJECT_DIR}"
echo ""

# List of critical files that MUST exist on worker PC
CRITICAL_FILES=(
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
    "data/EURUSD_H1.csv"
)

echo "🔍 Checking critical files on Worker PC..."
echo ""

ALL_FILES_EXIST=true

for file in "${CRITICAL_FILES[@]}"; do
    echo -n "Checking ${file}... "
    if ssh ${WORKER_USER}@${WORKER_IP} "test -f ${PROJECT_DIR}/${file}"; then
        echo "✅ EXISTS"
    else
        echo "❌ MISSING"
        ALL_FILES_EXIST=false
    fi
done

echo ""
echo "🧪 Testing Python imports on Worker PC..."
echo ""

# Test critical imports that are failing
IMPORTS_TO_TEST=(
    "synthetic_env"
    "bot_population" 
    "trading_bot"
    "config"
    "indicators"
    "predictors"
    "reward"
    "utils"
    "champion_analysis"
    "checkpoint_utils"
)

IMPORT_SUCCESS=true

for module in "${IMPORTS_TO_TEST[@]}"; do
    echo -n "Testing import ${module}... "
    if ssh ${WORKER_USER}@${WORKER_IP} "cd ${PROJECT_DIR} && python -c 'import ${module}' 2>/dev/null"; then
        echo "✅ SUCCESS"
    else
        echo "❌ FAILED"
        IMPORT_SUCCESS=false
    fi
done

echo ""
echo "📊 SUMMARY REPORT"
echo "================="

if [ "$ALL_FILES_EXIST" = true ] && [ "$IMPORT_SUCCESS" = true ]; then
    echo "🎉 ALL CHECKS PASSED!"
    echo "   Worker PC is properly configured for Ray training"
    echo ""
    echo "🚀 You can now run:"
    echo "   RAY_CLUSTER=1 python run_stable_85_percent_trainer.py"
else
    echo "❌ CONFIGURATION PROBLEMS DETECTED!"
    echo ""
    if [ "$ALL_FILES_EXIST" = false ]; then
        echo "🔧 SOLUTION: Run file sync:"
        echo "   ./sync_files_to_worker.sh"
    fi
    if [ "$IMPORT_SUCCESS" = false ]; then
        echo "🔧 SOLUTION: Check Worker PC Python environment:"
        echo "   ssh ${WORKER_USER}@${WORKER_IP}"
        echo "   conda activate BotsTraining_env"
        echo "   pip list | grep -E '(torch|ray|sklearn|pandas|numpy)'"
    fi
fi

echo ""
echo "🔍 Additional Worker PC Info:"
ssh ${WORKER_USER}@${WORKER_IP} "echo 'Current directory:'; pwd; echo 'Python version:'; python --version; echo 'Conda env:'; conda info --envs | grep '*'" 