# 🚀 MAXIMUM GPU/CPU UTILIZATION COMMANDS

## FOR 75% RESOURCE SATURATION ON BOTH PCs WITH OVERCLOCKED VRAM

### Option 1: Fixed Enhanced Launcher
```bash
cd /home/w1/cursor-to-copilot-backup/TaskmasterForexBots
python3 enhanced_training_launcher.py --gpu-utilization-target 0.75 --cpu-utilization-target 0.75
```

### Option 2: Direct Maximum Resource Training (RECOMMENDED)
```bash
cd /home/w1/cursor-to-copilot-backup/TaskmasterForexBots
./max_resource_training.sh
```

### Option 3: Pure Ray Training (Maximum Saturation)
```bash
cd /home/w1/cursor-to-copilot-backup/TaskmasterForexBots

# Start GUIs first
python3 generate_demo_fleet_data.py &
python3 kelly_bot_dashboard.py &
python3 ray_cluster_monitor_75_percent.py &

# Launch maximum resource training
python3 ray_kelly_ultimate_75_percent.py
```

### Option 4: Extreme Performance (Push to 85%)
```bash
cd /home/w1/cursor-to-copilot-backup/TaskmasterForexBots
python3 enhanced_training_launcher.py --gpu-utilization-target 0.85 --cpu-utilization-target 0.85
```

### Option 5: Ultra-Conservative Test (50%)
```bash
cd /home/w1/cursor-to-copilot-backup/TaskmasterForexBots
python3 enhanced_training_launcher.py --gpu-utilization-target 0.50 --cpu-utilization-target 0.50 --n-bots 500
```

## 🎯 What These Commands Do:

1. **CPU Saturation**: Uses ALL CPU cores at 75% utilization
2. **GPU Saturation**: Pushes GPU compute to 75% utilization  
3. **VRAM Saturation**: Uses 75% of available VRAM (safe for overclocked VRAM)
4. **Monte Carlo Scenarios**: 300,000 scenarios per bot for maximum GPU load
5. **2000 Bot Fleet**: Full fleet for maximum resource usage
6. **Parallel Processing**: ALL CPU cores + GPU cores working simultaneously

## 📊 Real-time Monitoring:

While training runs:
```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor CPU usage  
htop

# Check Ray dashboard
# http://192.168.1.10:8265
```

## 🔥 Maximum Performance Verification:

You should see:
- ✅ CPU usage: ~75% across all cores
- ✅ GPU usage: ~75% on both GPUs
- ✅ VRAM usage: ~75% (safe for overclocked VRAM)
- ✅ Bot dashboard showing 2000 active bots
- ✅ Resource monitor showing sustained high utilization

## 🚨 Emergency Stop:
```bash
pkill -f "ray_kelly"
pkill -f "kelly_bot"
ray stop --force
```
