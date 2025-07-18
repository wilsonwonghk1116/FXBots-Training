# Real Forex Bot Training Scripts
==================================

## Quick Start Guide

### Option 1: Full Automated Setup (Recommended)
```bash
cd /home/w1/cursor-to-copilot-backup/TaskmasterForexBots
conda activate Training_env
python launch_real_training.py
```

### Option 2: Direct Training (if cluster already running)
```bash
cd /home/w1/cursor-to-copilot-backup/TaskmasterForexBots
conda activate Training_env
python direct_training.py test    # For 5-minute test
python direct_training.py full    # For 3-hour full training
```

### Option 3: Original Automated Script
```bash
cd /home/w1/cursor-to-copilot-backup/TaskmasterForexBots
conda activate Training_env
python automated_cluster_training.py
```

## Training Modes

### Test Mode (5-10 minutes)
- 5 generations
- 10 bots per generation
- 50 episodes per bot
- 100 steps per episode
- Total: ~25,000 training steps

### Full Mode (3-4 hours)
- 200 generations
- 20 bots per generation
- 1000 episodes per bot
- 1000 steps per episode
- Total: ~4,000,000 training steps

## System Configuration

### Cluster Setup
- **PC1 (Head Node)**: 192.168.1.10
  - 80 CPUs, 1 GPU, 18GB VRAM
  - Ray head node with dashboard
- **PC2 (Worker Node)**: 192.168.1.11
  - 16 CPUs, 1 GPU, 6GB VRAM
  - Connected via SSH with password "w"

### Total Resources
- **96 CPUs** (75% utilization = 72 active)
- **2 GPUs** (75% VRAM = ~18GB available)
- **Distributed training** across both machines

## Monitor Training

### Ray Dashboard
- URL: http://192.168.1.10:8265
- Shows cluster status, resource usage, task progress

### Terminal Output
- Real-time training progress
- Performance metrics
- Error messages and status updates

## Troubleshooting

### If training fails to start:
1. Check Ray cluster status: `ray status`
2. Verify PC2 connectivity: `ping 192.168.1.11`
3. Test SSH: `sshpass -p 'w' ssh w1@192.168.1.11 'echo success'`

### If cluster setup fails:
1. Stop all Ray processes: `ray stop --force`
2. Restart with: `python launch_real_training.py`

### If out of memory:
- Training automatically uses 75% resource limits
- Reduce batch sizes if needed
- Monitor GPU memory in dashboard

## Expected Results

### Test Mode Results
- Quick validation of system functionality
- Basic performance metrics
- Confirmation cluster is working

### Full Mode Results
- Trained forex trading bot models
- Performance analysis reports
- Champion bot selection
- Saved model files (.pth format)

## Files Created During Training

- `CHAMPION_BOT_[timestamp].pth` - Best performing model
- `CHAMPION_ANALYSIS_[timestamp].json` - Performance analysis
- `training_results_[timestamp].json` - Full training metrics
- Dashboard logs in Ray temp directories

## Support

If you encounter any issues:
1. Check Ray dashboard for cluster status
2. Review terminal output for error messages
3. Verify network connectivity between PCs
4. Ensure conda environments are properly activated

## Performance Tips

1. **Monitor Resource Usage**: Use Ray dashboard to ensure 75% utilization
2. **Network Stability**: Ensure stable connection between PC1 and PC2
3. **Storage Space**: Ensure adequate disk space for model checkpoints
4. **Temperature**: Monitor GPU temperatures during long training sessions

## Command Summary

```bash
# Complete setup and training
python launch_real_training.py

# Quick test (if cluster ready)
python direct_training.py test

# Full training (if cluster ready)
python direct_training.py full

# Original comprehensive script
python automated_cluster_training.py

# Manual cluster check
ray status

# Stop all Ray processes
ray stop --force
```
