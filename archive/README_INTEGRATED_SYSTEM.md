# 🚀 Integrated Training System with Real-Time GUI

## Overview
This system combines Ray distributed training with a real-time PyQt6 dashboard that shows the top 20 performing bots ranked by total capital. Perfect for monitoring your 2000-bot fleet performance in real-time!

## Features
- 🎯 **Real-time Performance Monitoring**: Live updates of top 20 bots ranked by total capital
- 📊 **Comprehensive Metrics**: Win rates, Sharpe ratios, PnL, drawdowns, and more
- 🚀 **Integrated Training**: Starts Ray training automatically with GUI monitoring
- 💻 **Resource Utilization**: Targets 75% CPU/GPU/VRAM utilization across your 2-PC cluster
- 🔄 **Auto-sorting**: Continuously re-ranks bots by total capital performance
- 📈 **Live Charts**: Real-time performance visualization

## Quick Start

### Option 1: One-Click Launch (Recommended)
```bash
# Launch everything with automatic dependency checking
./launch_training_gui.sh
```

### Option 2: Direct Python Launch
```bash
# Run the integrated system directly
python3 integrated_training_with_gui.py
```

### Option 3: Manual Ray + GUI
```bash
# Terminal 1: Start Ray training
python3 ray_kelly_ultimate_75_percent.py

# Terminal 2: Start GUI monitor (in parallel)
python3 integrated_training_with_gui.py
```

## How It Works

### 1. Training Process
- Initializes 2000 Kelly Monte Carlo bots
- Distributes across your Ray cluster (both PCs)
- Runs 300,000 Monte Carlo scenarios per bot
- Achieves 75% resource utilization target

### 2. Real-Time Monitoring
- Updates every 2-3 seconds
- Shows top 20 bots by total capital
- Auto-sorts and re-ranks continuously
- Displays comprehensive performance metrics

### 3. Data Flow
```
Ray Training → fleet_results.json → GUI Dashboard → Live Display
     ↓              ↓                    ↓            ↓
   2000 bots    Real-time saves    Auto-refresh   Top 20 ranked
```

## GUI Interface

### Main Dashboard Components

1. **Control Panel**
   - 🚀 START TRAINING: Begins Ray distributed training
   - ⏹️ STOP TRAINING: Safely stops all processes
   - 🔄 REFRESH DATA: Manual data update

2. **Performance Summary**
   - Total Active Bots
   - Fleet Total Capital
   - Average Performance %
   - Best Performer identification

3. **Top 20 Table** (Main Focus)
   - Bot ID
   - Total Capital ($) ← **Primary Sort Column**
   - Total PnL ($)
   - Win Rate (%)
   - Number of Trades
   - Sharpe Ratio
   - Max Drawdown (%)

4. **Training Log**
   - Real-time status updates
   - Resource utilization metrics
   - Progress indicators

### Color Coding
- 🟢 **Green**: High performers (>10% return)
- 🟡 **Yellow**: Profitable bots (0-10% return)
- 🔴 **Red**: Loss-making bots (<0% return)

## System Requirements

### Hardware
- **PC1**: Xeon + RTX 3090 (24GB VRAM)
- **PC2**: i9 + RTX 3070 (8GB VRAM)
- **Network**: Both PCs connected for Ray cluster

### Software Dependencies
```bash
# Core requirements (auto-installed by launcher)
pip install PyQt6              # GUI framework
pip install ray[default]       # Distributed computing
pip install torch              # GPU acceleration
pip install pandas numpy       # Data processing
pip install psutil GPUtil      # Resource monitoring
```

## Performance Targets

### Resource Utilization
- ✅ CPU: 75% across all cores (both PCs)
- ✅ GPU: 75% utilization (RTX 3090 + RTX 3070)
- ✅ VRAM: 75% usage (24GB + 8GB = 32GB total)
- ✅ Memory: Efficient RAM usage with monitoring

### Trading Performance
- 🤖 **Fleet Size**: 2000 bots simultaneously
- 📊 **Scenarios**: 300,000 Monte Carlo simulations per bot
- ⚡ **Speed**: Real-time processing with 2-3 second updates
- 🎯 **Focus**: Top 20 performers by total capital

## Troubleshooting

### Common Issues

1. **"PyQt6 not found"**
   ```bash
   pip install PyQt6
   ```

2. **Ray cluster connection issues**
   ```bash
   # On head node (PC1)
   ray start --head --port=8265
   
   # On worker node (PC2)
   ray start --address='PC1_IP:10001'
   ```

3. **GPU not detected**
   ```bash
   # Check CUDA
   nvidia-smi
   python3 -c "import torch; print(torch.cuda.is_available())"
   ```

4. **No real-time updates**
   - Check if `fleet_results.json` is being created
   - Ensure Ray training is actually running
   - Verify file permissions

### Performance Tips

1. **Maximize GPU Usage**
   - Ensure both RTX 3090 and RTX 3070 are detected
   - Monitor VRAM usage in GUI log
   - Increase Monte Carlo scenarios if GPUs underutilized

2. **Optimize Network**
   - Use gigabit ethernet between PCs
   - Monitor Ray cluster status in terminal

3. **Real-time Responsiveness**
   - Close unnecessary applications
   - Set high process priority for GUI
   - Monitor system resources in GUI

## File Structure
```
TaskmasterForexBots/
├── integrated_training_with_gui.py    # Main integrated system
├── launch_training_gui.sh             # One-click launcher
├── ray_kelly_ultimate_75_percent.py   # Ray training backend
├── kelly_monte_bot.py                 # Core bot implementation
├── fleet_results.json                 # Real-time data file
└── README_INTEGRATED_SYSTEM.md        # This guide
```

## Advanced Usage

### Custom Configuration
```python
# Modify bot count
fleet_manager = BotFleetManager.remote(n_bots=3000)  # Increase to 3000

# Adjust update frequency
self.update_timer.start(1000)  # 1-second updates (faster)

# Change display count
self.performance_table = AnimatedTable(50, 7)  # Show top 50 instead of 20
```

### Monitoring Multiple Metrics
The system can be extended to sort by different criteria:
- Total PnL (profit/loss)
- Sharpe ratio (risk-adjusted returns)
- Win rate percentage
- Number of trades
- Maximum drawdown

## Support
- 📧 Check logs in GUI for real-time diagnostics
- 🔍 Monitor resource usage for optimization
- 🚀 Ensure Ray cluster is properly configured
- 💻 Verify GPU acceleration is working

---

**🎯 Ready to launch your high-performance trading fleet with real-time monitoring!**

Use `./launch_training_gui.sh` to get started immediately.
