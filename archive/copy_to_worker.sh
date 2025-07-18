#!/bin/bash

echo "🚀 Copying Files to Worker PC 2"
echo "==============================="

WORKER_IP="192.168.1.11"
WORKER_USER="w2"
WORKER_PASS="w"
PROJECT_DIR="/home/w2/cursor-to-copilot-backup/TaskmasterForexBots"

echo "📡 Worker PC: ${WORKER_IP}"
echo "👤 User: ${WORKER_USER}"
echo "🔑 Password: ${WORKER_PASS}"
echo "📁 Target Dir: ${PROJECT_DIR}"
echo ""

# Make sure we're in the right directory
cd /home/w1/cursor-to-copilot-backup/TaskmasterForexBots

# Check if tar file exists
if [ ! -f "worker_files.tar.gz" ]; then
    echo "📦 Creating tar file..."
    tar czf worker_files.tar.gz *.py data/
    echo "✅ Tar file created: $(ls -lh worker_files.tar.gz | awk '{print $5}')"
fi

# Install sshpass if not present
if ! command -v sshpass &> /dev/null; then
    echo "📦 Installing sshpass..."
    sudo apt-get update && sudo apt-get install -y sshpass
fi

echo ""
echo "🔄 Step 1: Creating directory on Worker PC..."
sshpass -p "${WORKER_PASS}" ssh -o StrictHostKeyChecking=no ${WORKER_USER}@${WORKER_IP} "mkdir -p ${PROJECT_DIR}"

echo "🔄 Step 2: Copying tar file to Worker PC..."
sshpass -p "${WORKER_PASS}" scp -o StrictHostKeyChecking=no worker_files.tar.gz ${WORKER_USER}@${WORKER_IP}:${PROJECT_DIR}/

echo "🔄 Step 3: Extracting files on Worker PC..."
sshpass -p "${WORKER_PASS}" ssh -o StrictHostKeyChecking=no ${WORKER_USER}@${WORKER_IP} "cd ${PROJECT_DIR} && tar xzf worker_files.tar.gz && rm worker_files.tar.gz"

echo "🔄 Step 4: Testing imports on Worker PC..."
sshpass -p "${WORKER_PASS}" ssh -o StrictHostKeyChecking=no ${WORKER_USER}@${WORKER_IP} "cd ${PROJECT_DIR} && source ~/miniconda3/etc/profile.d/conda.sh && conda activate BotsTraining_env && python -c 'import synthetic_env; print(\"✅ synthetic_env OK\")'"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 SUCCESS! Worker PC 2 is ready!"
    echo "📁 Files copied successfully"
    echo "✅ Python imports working"
    echo ""
    echo "🚀 You can now run:"
    echo "   RAY_CLUSTER=1 python run_stable_85_percent_trainer.py"
else
    echo ""
    echo "❌ Import test failed. Check Worker PC environment."
fi 