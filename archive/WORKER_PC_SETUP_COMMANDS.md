# 🔧 Worker PC Setup Commands

## Problem Diagnosis
**Issue**: `ModuleNotFoundError: No module named 'synthetic_env'`
**Cause**: Worker PC (192.168.1.11) doesn't have the required Python files
**Solution**: Copy all project files from Head PC to Worker PC

---

## 🚀 QUICK AUTOMATED SOLUTION

Run this script from Head PC (192.168.1.10):
```bash
cd ~/cursor-to-copilot-backup/TaskmasterForexBots
./sync_files_to_worker.sh
```

---

## 🔧 MANUAL SOLUTION (if automated script fails)

### Step 1: Copy Required Python Files

From Head PC, run these commands:
```bash
cd ~/cursor-to-copilot-backup/TaskmasterForexBots

# Copy all Python modules
scp *.py w2@192.168.1.11:~/cursor-to-copilot-backup/TaskmasterForexBots/

# Copy data directory
scp -r data/ w2@192.168.1.11:~/cursor-to-copilot-backup/TaskmasterForexBots/

# Copy models and checkpoints directories
scp -r models/ w2@192.168.1.11:~/cursor-to-copilot-backup/TaskmasterForexBots/ 2>/dev/null || echo "models dir not found"
scp -r checkpoints/ w2@192.168.1.11:~/cursor-to-copilot-backup/TaskmasterForexBots/ 2>/dev/null || echo "checkpoints dir not found"
```

### Step 2: Setup Conda Environment on Worker PC

SSH to Worker PC and setup environment:
```bash
# SSH to Worker PC
ssh w2@192.168.1.11

# Navigate to project directory
cd ~/cursor-to-copilot-backup/TaskmasterForexBots

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate BotsTraining_env

# Install any missing dependencies
pip install torch torchvision ray pandas numpy

# Test imports
python -c "import synthetic_env; print('✅ synthetic_env import successful')"
python -c "import bot_population; print('✅ bot_population import successful')"
python -c "import trading_bot; print('✅ trading_bot import successful')"
```

### Step 3: Verify File Structure

On Worker PC, check file structure:
```bash
ls -la *.py
ls -la data/
python -c "import sys; print(sys.path)"
```

---

## 🧪 TESTING COMMANDS

After copying files, test Ray cluster functionality:

### On Head PC (192.168.1.10):
```bash
cd ~/cursor-to-copilot-backup/TaskmasterForexBots
python test_new_ray_cluster.py
```

### Test Training (Both PCs):
```bash
RAY_CLUSTER=1 python run_stable_85_percent_trainer.py
```

---

## 🚨 TROUBLESHOOTING

### If SSH fails:
```bash
# Test SSH connection
ssh w2@192.168.1.11 "echo 'SSH connection working'"

# If fails, setup SSH keys:
ssh-keygen -t rsa -b 4096 -C "ray-cluster"
ssh-copy-id w1@192.168.1.11
```

### If conda environment doesn't exist on Worker PC:
```bash
# Create environment on Worker PC
conda create -n BotsTraining_env python=3.12 -y
conda activate BotsTraining_env
pip install torch torchvision ray pandas numpy scipy matplotlib seaborn ta
```

### If files still missing:
```bash
# Manual verification on Worker PC
cd ~/cursor-to-copilot-backup/TaskmasterForexBots
ls -la *.py | grep -E "(synthetic_env|bot_population|trading_bot|config)"
```

---

## 📋 REQUIRED FILES LIST

Essential Python files that MUST be on Worker PC:
- ✅ `synthetic_env.py`
- ✅ `bot_population.py` 
- ✅ `trading_bot.py`
- ✅ `config.py`
- ✅ `indicators.py`
- ✅ `predictors.py`
- ✅ `reward.py`
- ✅ `utils.py`
- ✅ `champion_analysis.py`
- ✅ `checkpoint_utils.py`
- ✅ `data/EURUSD_H1.csv`

---

## 🎯 SUCCESS CRITERIA

After completing setup:
1. ✅ SSH connection works between Head and Worker PC
2. ✅ All Python files present on both machines
3. ✅ Conda environment activated on both machines
4. ✅ `python -c "import synthetic_env"` works on Worker PC
5. ✅ Ray cluster shows 2 active nodes
6. ✅ Training starts without import errors 