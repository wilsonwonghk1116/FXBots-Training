# Kelly Monte Carlo Ray Cluster Training Commands

## 🚀 Complete Training Setup with GUI Monitoring

### Prerequisites
```bash
# Install required packages
pip install ray[default] torch pandas numpy matplotlib psutil GPUtil PyQt6

# Ensure all scripts are executable
chmod +x start_full_training.sh
chmod +x setup_ray_cluster_75_percent.sh
```

### Option 1: Automated Full Setup (Recommended)
```bash
# Single command to start everything with GUI
./start_full_training.sh
```

### Option 2: Manual Step-by-Step Setup

#### Step 1: Setup Ray Cluster
```bash
# On HEAD NODE (PC1):
ray start --head --port=8265 --dashboard-host=0.0.0.0 --dashboard-port=8266 \
  --num-cpus=$(nproc) --num-gpus=$(nvidia-smi -L | wc -l) \
  --memory=$(free -b | grep '^Mem:' | awk '{printf "%.0f", $2 * 0.75}') \
  --include-dashboard=true

# On WORKER NODE (PC2):
ray start --address='HEAD_NODE_IP:10001' \
  --num-cpus=$(nproc) --num-gpus=$(nvidia-smi -L | wc -l) \
  --memory=$(free -b | grep '^Mem:' | awk '{printf "%.0f", $2 * 0.75}')
```

#### Step 2: Start GUI Monitoring (on Head Node)
```bash
# Terminal 1: Resource Monitor
python3 ray_cluster_monitor_75_percent.py &

# Terminal 2: Bot Performance Dashboard
python3 generate_demo_fleet_data.py  # Generate initial data
python3 kelly_bot_dashboard.py &
```

#### Step 3: Launch Main Training
```bash
# Option A: Using enhanced launcher
python3 enhanced_training_launcher.py \
  --n-bots 2000 \
  --duration-hours 24 \
  --gpu-utilization-target 0.75 \
  --cpu-utilization-target 0.75

# Option B: Direct training script
python3 ray_kelly_ultimate_75_percent.py
```

### Option 3: Quick Test Run
```bash
# Fast 1-hour test with smaller fleet
python3 enhanced_training_launcher.py \
  --n-bots 100 \
  --duration-hours 1 \
  --gpu-utilization-target 0.50 \
  --cpu-utilization-target 0.50
```

### Option 4: Production 24-Hour Training
```bash
# Full production run with logging
nohup python3 enhanced_training_launcher.py \
  --n-bots 2000 \
  --duration-hours 24 \
  --gpu-utilization-target 0.75 \
  --cpu-utilization-target 0.75 \
  --session-dir "production_$(date +%Y%m%d_%H%M%S)" \
  > training_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

## 📊 GUI Monitoring Access

1. **Ray Dashboard**: http://HEAD_NODE_IP:8266
   - Real-time cluster status
   - Task execution monitoring
   - Resource utilization graphs

2. **Resource Monitor**: Automatic popup window
   - CPU/GPU/Memory usage
   - 75% utilization target tracking
   - Performance metrics

3. **Bot Performance Dashboard**: Automatic popup window
   - Top 20 bot rankings
   - Real-time P&L updates
   - Trade history visualization

## 🔧 Configuration Options

### Environment Variables
```bash
export RAY_HEAD_IP="192.168.1.100"      # Change to your head node IP
export RAY_DASHBOARD_PORT="8266"         # Ray dashboard port
export TRAINING_DURATION="24"            # Training hours
export BOT_FLEET_SIZE="2000"            # Number of bots
export GPU_TARGET="0.75"                # GPU utilization target
export CPU_TARGET="0.75"                # CPU utilization target
```

### Custom Training Parameters
```bash
python3 enhanced_training_launcher.py \
  --n-bots 1500 \
  --duration-hours 12 \
  --gpu-utilization-target 0.80 \
  --cpu-utilization-target 0.70 \
  --auto-save-interval 1800 \
  --head-node-ip "192.168.1.100"
```

## 📈 Real-time Monitoring Commands

### Check Cluster Status
```bash
ray status                              # Ray cluster status
nvidia-smi                             # GPU status
htop                                   # CPU/Memory status
```

### Monitor Training Progress
```bash
# View live training logs
tail -f training_*.log

# Check bot performance
python3 -c "
import json
with open('fleet_results.json') as f:
    data = json.load(f)
    print(f'Active bots: {len(data[\"bots\"])}')
    top_bot = max(data['bots'], key=lambda x: x['total_capital'])
    print(f'Top performer: {top_bot[\"name\"]} - ${top_bot[\"total_capital\"]:,.2f}')
"
```

### Emergency Stop
```bash
# Graceful stop
pkill -f "ray_kelly_ultimate"
pkill -f "kelly_bot_dashboard"
pkill -f "ray_cluster_monitor"

# Force stop Ray cluster
ray stop --force
```

## 🎯 Optimal Performance Commands

### Maximum GPU Saturation
```bash
python3 enhanced_training_launcher.py \
  --n-bots 2000 \
  --duration-hours 24 \
  --gpu-utilization-target 0.95 \
  --cpu-utilization-target 0.85
```

### Balanced Performance
```bash
python3 enhanced_training_launcher.py \
  --n-bots 1500 \
  --duration-hours 18 \
  --gpu-utilization-target 0.75 \
  --cpu-utilization-target 0.75
```

### Conservative Testing
```bash
python3 enhanced_training_launcher.py \
  --n-bots 500 \
  --duration-hours 2 \
  --gpu-utilization-target 0.50 \
  --cpu-utilization-target 0.50
```

## 📋 Success Verification

After starting, you should see:
1. ✅ Ray dashboard accessible at http://HEAD_NODE_IP:8266
2. ✅ Resource monitor GUI window showing utilization graphs
3. ✅ Bot dashboard GUI window showing top 20 performers
4. ✅ Training logs showing bot initialization and Monte Carlo scenarios
5. ✅ GPU utilization reaching target percentage
6. ✅ fleet_results.json being updated every 2 seconds

## 🚨 Troubleshooting

### Ray Connection Issues
```bash
# Reset Ray cluster
ray stop --force
sleep 5
./start_full_training.sh
```

### GPU Not Detected
```bash
# Check NVIDIA drivers
nvidia-smi
# Reinstall PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### GUI Issues
```bash
# Check PyQt6 installation
pip install PyQt6
# Run GUI components separately
python3 kelly_bot_dashboard.py
python3 ray_cluster_monitor_75_percent.py
```
