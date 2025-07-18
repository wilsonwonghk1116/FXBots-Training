#!/bin/bash
# ==============================================================================
# ONE-CLICK DUAL PC TRAINING LAUNCHER
# ------------------------------------------------------------------------------
# This script automates the entire process of:
# 1. Cleaning up any old Ray processes on both PCs.
# 2. Starting the Ray Head node on PC1 (this machine).
# 3. Connecting the Ray Worker node from PC2.
# 4. Verifying the cluster is stable.
# 5. Launching the distributed training application.
# ==============================================================================

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Go to Project Root ---
# This ensures the script can be run from any directory
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT=$( dirname -- "$SCRIPT_DIR" )
cd -- "$PROJECT_ROOT"
echo "Changed working directory to project root: $(pwd)"
echo ""

# --- Configuration ---
HEAD_PC_IP="192.168.1.10"
WORKER_PC_IP="192.168.1.11"
WORKER_PC_USER="w2"
WORKER_PC_PASS="w"
CONDA_ENV_NAME="Training_env"
TRAINING_SCRIPT_PATH="TaskmasterForexBots/fixed_integrated_training_75_percent.py"
WORKER_CONDA_PATH="/home/w2/miniconda3/etc/profile.d/conda.sh"
HEAD_CONDA_PATH="/home/w1/miniconda3/etc/profile.d/conda.sh"

# --- Activate Conda Environment for this script ---
echo "Initializing Conda for this script..."
source "${HEAD_CONDA_PATH}"
conda activate "${CONDA_ENV_NAME}"
echo "Conda environment '${CONDA_ENV_NAME}' activated."

# --- STEP 1: Clean Head PC (PC1) ---
echo ""
echo "--- STEP 1: Forcing Ray stop on Head PC (${HEAD_PC_IP}) ---"
ray stop --force
echo "--- Head PC clean. ---"
echo ""

# --- STEP 2: Clean Worker PC (PC2) ---
echo "--- STEP 2: Forcing Ray stop on Worker PC (${WORKER_PC_IP}) via SSH ---"
sshpass -p "${WORKER_PC_PASS}" ssh ${WORKER_PC_USER}@${WORKER_PC_IP} "source ${WORKER_CONDA_PATH} && conda activate ${CONDA_ENV_NAME} && ray stop --force"
echo "--- Worker PC clean. ---"
echo ""

# --- STEP 3: Start Head Node (PC1) ---
echo "--- STEP 3: Starting Ray Head Node on PC1 (${HEAD_PC_IP}) ---"
ray start --head --node-ip-address="${HEAD_PC_IP}" --dashboard-host='0.0.0.0'
echo "--- Head Node started. ---"
echo ""

# --- STEP 4: Connect Worker Node (PC2) ---
echo "--- STEP 4: Connecting Worker Node PC2 (${WORKER_PC_IP}) to Head Node ---"
sshpass -p "${WORKER_PC_PASS}" ssh ${WORKER_PC_USER}@${WORKER_PC_IP} "source ${WORKER_CONDA_PATH} && conda activate ${CONDA_ENV_NAME} && ray start --address='${HEAD_PC_IP}:6379' --node-ip-address='${WORKER_PC_IP}'"
echo "--- Worker Node connection command sent. ---"
echo ""

# --- STEP 5: Stabilize and Verify ---
echo "--- STEP 5: Waiting 5 seconds for cluster to stabilize... ---"
sleep 5
echo "--- Verifying final cluster status... ---"
ray status | cat
echo "--- Status check complete. ---"
echo ""

# --- STEP 6: Launch Training Application ---
echo "========================================================"
echo "--- STEP 6: Cluster ready. Launching training script... ---"
echo "========================================================"
python ${TRAINING_SCRIPT_PATH} 