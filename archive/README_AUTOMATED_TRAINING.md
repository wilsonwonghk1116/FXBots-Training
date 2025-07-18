# 🤖 Automated Ray Cluster Training System

## Overview

This automated system handles the complete process of setting up a dual-PC Ray cluster and launching massive scale forex trading bot training. It eliminates manual steps and ensures consistent setup across multiple training sessions.

## Features

- 🐍 **Automatic conda environment activation**
- 🖥️ **Ray head node setup on PC1**
- 🔗 **Automated PC2 worker connection via SSH**
- ✅ **Cluster verification and resource checking**
- 🚀 **Seamless training launch**
- 🧹 **Automatic cleanup on completion/interruption**
- ⚡ **Python version compatibility checking**
- 📊 **Real-time progress monitoring**

## Quick Start

### 1. Configure SSH Password

```bash
python setup_cluster_config.py
```

Enter your SSH password for PC2 when prompted.

### 2. Run Automated Training

```bash
python automated_cluster_training.py
```

Select your training mode:
- **Full Scale**: 200 generations × 1000 episodes × 1000 steps (~3 hours)
- **Test Scale**: 2 generations × 10 episodes × 100 steps (~5 minutes)

## System Requirements

### PC1 (Head Node) - 192.168.1.10
- 80 CPU cores
- 1 GPU (RTX 3090)
- 16GB object store memory
- Training_env conda environment

### PC2 (Worker Node) - 192.168.1.11
- 16 CPU cores
- 1 GPU (RTX 3070)
- 8GB object store memory
- Training_env conda environment
- SSH access enabled

## Configuration

All settings are in `cluster_config.py`:

```python
# Network Configuration
PC1_IP = "192.168.1.10"
PC2_IP = "192.168.1.11"
PC2_USER = "w1"
PC2_SSH_PASSWORD = "your_password"  # Set via setup script

# Ray Configuration
RAY_PORT = 10001
RAY_DASHBOARD_PORT = 8265

# Hardware Configuration
PC1_CPUS = 80
PC1_GPUS = 1
PC2_CPUS = 16
PC2_GPUS = 1

# Training Configuration
TEST_GENERATIONS = 2
FULL_GENERATIONS = 200
```

## Process Flow

1. **Environment Check**: Verifies conda environments on both PCs
2. **Python Compatibility**: Ensures matching Python versions
3. **Ray Head Start**: Launches Ray head node on PC1
4. **Worker Connection**: SSH connects PC2 as Ray worker
5. **Cluster Verification**: Confirms resource availability
6. **Training Launch**: Starts massive scale distributed training
7. **Monitoring**: Real-time progress tracking
8. **Cleanup**: Automatic Ray cluster shutdown

## Monitoring

- **Ray Dashboard**: http://192.168.1.10:8265
- **Real-time logs**: Terminal output with timestamps
- **Progress tracking**: Generation completion status
- **Resource usage**: CPU/GPU utilization monitoring

## Troubleshooting

### Python Version Mismatch
```
❌ Version mismatch: The cluster was started with Ray: 2.47.1 Python: 3.12.11
   This process was started with Ray: 2.47.1 Python: 3.13.5
```

**Solution**: Ensure both PCs use the same Python version in Training_env:
```bash
# On both PCs
conda activate Training_env
python --version
```

### SSH Connection Failed
```
❌ Failed to connect PC2: Permission denied
```

**Solutions**:
1. Update password in `cluster_config.py`
2. Test SSH manually: `ssh w1@192.168.1.11`
3. Install sshpass: `sudo apt install sshpass`

### Ray Connection Issues
```
❌ Failed to connect to Ray cluster
```

**Solutions**:
1. Check Ray dashboard: http://192.168.1.10:8265
2. Verify network connectivity between PCs
3. Restart with: `ray stop --force` on both PCs

### Insufficient Resources
```
❌ CLUSTER_INSUFFICIENT: Got 80/96 CPUs, 1/2 GPUs
```

**Solution**: Check that PC2 worker connected successfully and both GPUs are detected.

## Files

- `automated_cluster_training.py` - Main automation script
- `cluster_config.py` - Configuration file
- `setup_cluster_config.py` - SSH password setup utility
- `massive_scale_distributed_training.py` - Core training system

## Training Results

Results are automatically saved to:
- `massive_scale_results_TIMESTAMP.json` - Final results
- `massive_training_checkpoint_gen_X.json` - Intermediate checkpoints

## Safety Features

- **Signal handling**: Graceful shutdown on Ctrl+C
- **Automatic cleanup**: Ray processes stopped on exit
- **Resource verification**: Ensures adequate cluster resources
- **Progress checkpoints**: Regular training state saves
- **Error recovery**: Detailed error messages and solutions

## Advanced Usage

### Custom Training Configuration

Modify `cluster_config.py` to adjust:
- Number of generations/episodes/steps
- CPU/GPU allocation per PC
- Memory limits
- Network ports

### Manual Cluster Control

```bash
# Start head node only
ray start --head --port=10001 --dashboard-host=0.0.0.0

# Connect worker manually
ray start --address=192.168.1.10:10001

# Check cluster status
ray status

# Stop cluster
ray stop --force
```

---

## 🚀 Ready to Train!

Your automated cluster training system is ready. Simply run:

```bash
python automated_cluster_training.py
```

And let the system handle everything from cluster setup to training completion!
