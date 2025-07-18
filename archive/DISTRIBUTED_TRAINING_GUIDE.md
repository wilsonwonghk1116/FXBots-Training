# Distributed Forex Bot Training Guide

## 🎯 Quick Start (After Worker PC Reconnects)

### Step 1: Setup Ray Cluster on Head PC (192.168.1.10)
```bash
cd /home/w1/cursor-to-copilot-backup/TaskmasterForexBots
conda activate BotsTraining_env
python setup_ray_cluster_with_env.py
```

This will:
- ✅ Start Ray head node with proper environment
- ✅ Create worker connection commands
- ✅ Verify cluster status
- ✅ Generate `worker_connection_commands.txt`

### Step 2: Connect Worker PC (192.168.1.11)
Copy and run the commands from `worker_connection_commands.txt` on Worker PC:

```bash
# Commands for Worker PC:
conda activate BotsTraining_env

# Set environment variables:
export PYTHONPATH="/home/w1/cursor-to-copilot-backup/TaskmasterForexBots:/home/w2/cursor-to-copilot-backup/TaskmasterForexBots:$PYTHONPATH"
export RAY_CLUSTER=1

# Connect to Ray cluster:
ray start --address='192.168.1.10:6379' --temp-dir=/tmp/ray

# Verify connection:
ray status
```

### Step 3: Test Worker Environment (Optional but Recommended)
On Worker PC, run the environment test:
```bash
cd /home/w2/cursor-to-copilot-backup/TaskmasterForexBots
python test_worker_environment.py
```

### Step 4: Launch Distributed Training
On Head PC:
```bash

```

## 🔧 What We Fixed

### Key Problems Solved:
1. **Module Import Errors**: Ray workers now have proper PYTHONPATH setup
2. **Environment Configuration**: Workers use correct conda environment 
3. **Project Path Issues**: Both head and worker paths are included
4. **Ray Import Issues**: Fixed type checking and conditional imports

### Enhanced Scripts:
- `run_stable_85_percent_trainer.py` - Fixed Ray worker environment setup
- `setup_ray_cluster_with_env.py` - Comprehensive cluster setup
- `test_worker_environment.py` - Environment verification
- `launch_distributed_training.py` - Simple training launcher

## 📊 Expected Training Performance

### Head PC (192.168.1.10) - RTX 3090:
- **GPU Usage**: 85% VRAM, target utilization
- **CPU Usage**: 50 threads at 90% 
- **Population**: ~3000-5000 bots
- **Generations**: 300

### Worker PC (192.168.1.11) - RTX 3070:
- **GPU Usage**: 85% VRAM, target utilization  
- **CPU Usage**: 16 threads at 85%
- **Population**: ~2000-3000 bots
- **Generations**: 300

### Combined Performance:
- **Total Bots**: 5000-8000 per generation
- **Total CPUs**: 96 (66 + 30)
- **Total GPUs**: 2 (RTX 3090 + RTX 3070)
- **Expected Training Time**: 4-6 hours for 300 generations

## 🚨 Troubleshooting

### If Worker PC Disconnects:
1. On Worker PC: `ray stop`
2. Wait 30 seconds
3. Re-run connection commands from Step 2

### If Training Fails:
1. Check Ray cluster status: `ray status`
2. Verify environment: `python test_worker_environment.py`
3. Check logs in `/tmp/ray/session_*/logs/`

### If Import Errors Persist:
1. Ensure files are copied: `scp -r w1@192.168.1.10:/home/w1/cursor-to-copilot-backup/TaskmasterForexBots /home/w2/cursor-to-copilot-backup/`
2. Verify conda environment: `conda activate BotsTraining_env`
3. Test imports manually: `python -c "import synthetic_env; print('✅ Success')"`

## 🎮 GPU Monitoring

Monitor GPU usage during training:
```bash
# On Head PC:
watch -n 1 nvidia-smi

# On Worker PC:
ssh w2@192.168.1.11 'watch -n 1 nvidia-smi'
```

## 📁 Output Files

Training will generate:
- `final_champion_3090.pt` - Head PC champion
- `final_champion_3070.pt` - Worker PC champion  
- Training logs with performance metrics
- Ray dashboard at `http://192.168.1.10:8265`

## 🔄 Restarting After Reboot

If either PC restarts:
1. **Head PC**: Run Step 1 again
2. **Worker PC**: Run Step 2 again  
3. **Both**: Verify with `ray status`
4. **Launch**: Run Step 4

## 🎯 Success Indicators

✅ **Ray Cluster Ready**:
```
Total CPUs: 96.0
Total GPUs: 2.0
Active nodes: 2
```

✅ **Training Started**:
```
🔥 === GENERATION 1/300 (3090) ===
🔥 === GENERATION 1/300 (3070) ===
```

✅ **Training Complete**:
```
🎊 === TRAINING COMPLETE on 3090 ===
🎊 === TRAINING COMPLETE on 3070 ===
```

The system is now fully prepared for distributed training! 🚀 