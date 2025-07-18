# Ray Distributed Kelly Monte Carlo FOREX Bot System

## High-Performance Multi-GPU Cluster Setup

This system is optimized for maximum hardware utilization across your Ray cluster:

- **Head PC**: Xeon E5 x2 (80 threads) + RTX 3090 (24GB VRAM)
- **Worker PC**: i9 (16 threads) + RTX 3070 (8GB VRAM)
- **Target**: 75% CPU + GPU utilization across all nodes
- **Fleet**: 2000 Kelly Monte Carlo trading bots
- **Scenarios**: 100,000 Monte Carlo scenarios per decision

## Quick Start

### 1. Setup Environment (Both PCs)

```bash
# Install Python dependencies
./start_ray_cluster.sh setup

# Verify GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
```

### 2. Start Ray Cluster

**On Head PC (Xeon + RTX 3090):**
```bash
# Start head node
./start_ray_cluster.sh head

# Note the IP address shown for worker connection
```

**On Worker PC (i9 + RTX 3070):**
```bash
# Connect to head node (replace with actual head PC IP)
./start_ray_cluster.sh worker 192.168.1.100
```

### 3. Verify Cluster

```bash
# Check cluster status
./start_ray_cluster.sh status

# Should show both nodes connected
```

### 4. Run Distributed Simulation

```bash
# Start performance monitoring (optional, in separate terminal)
python ray_cluster_monitor.py

# Run the distributed Kelly Monte Carlo fleet
python ray_distributed_kelly_bot.py
```

## System Architecture

### Resource Allocation

The system is designed to maximize utilization of your cluster resources:

- **CPU Cores**: 96 total (80 + 16) - targeting 75% usage
- **GPU Memory**: 32GB total (24GB + 8GB) - targeting 75% usage  
- **Monte Carlo Engine**: 100k scenarios per decision for GPU saturation
- **Parallel Processing**: Ray actors distributed across both machines

### Component Overview

```
Ray Cluster
├── Head PC (Xeon E5 x2 + RTX 3090)
│   ├── Ray Head Node + Dashboard
│   ├── 1000 RayKellyBot actors
│   ├── 4 RayBatchProcessor actors
│   └── GPU Monte Carlo Engine (100k scenarios)
│
└── Worker PC (i9 + RTX 3070)
    ├── Ray Worker Node
    ├── 1000 RayKellyBot actors
    ├── 4 RayBatchProcessor actors
    └── GPU Monte Carlo Engine (100k scenarios)
```

## Configuration Options

### Environment Variables

```bash
# Fleet configuration
export N_BOTS=2000              # Total number of bots
export SIM_HOURS=5000           # Hours to simulate
export BATCH_HOURS=100          # Hours per batch
export N_PROCESSORS=8           # Batch processors

# Run with custom config
N_BOTS=4000 SIM_HOURS=10000 python ray_distributed_kelly_bot.py
```

### Trading Parameters

Edit `kelly_monte_bot.py` to adjust:

```python
@dataclass
class TradingParameters:
    monte_carlo_scenarios: int = 100000  # GPU saturation level
    stop_loss_pips: float = 30.0
    take_profit_pips: float = 60.0
    max_risk_per_trade: float = 0.02
```

## Performance Monitoring

### Real-time Monitor

```bash
# Start real-time performance monitoring
python ray_cluster_monitor.py
```

Monitor output shows:
- CPU utilization across all 96 threads
- GPU utilization for both RTX 3090 and RTX 3070
- VRAM usage (24GB + 8GB)
- Ray cluster resource allocation
- Performance recommendations

### Ray Dashboard

Access the Ray dashboard at: `http://<head_pc_ip>:8265`

Features:
- Real-time cluster metrics
- Actor distribution across nodes
- GPU memory usage
- Task execution timeline

## Expected Performance

### Target Metrics
- **CPU Usage**: 75% across 96 threads (72 active cores)
- **GPU Usage**: 75% on both RTX 3090 and RTX 3070
- **VRAM Usage**: ~24GB (75% of 32GB total)
- **Processing Rate**: 500+ hours/second simulation time
- **Monte Carlo Rate**: 200M+ scenarios/second

### Optimization Features

1. **GPU Saturation**: 100,000 Monte Carlo scenarios per decision
2. **Vectorized Processing**: Batch operations on GPU tensors
3. **Memory Management**: Optimized CUDA memory allocation
4. **Load Balancing**: Ray's automatic work distribution
5. **Parallel I/O**: Concurrent data processing across nodes

## Troubleshooting

### Common Issues

**Ray connection fails:**
```bash
# Check firewall (port 10001)
sudo ufw allow 10001

# Verify network connectivity
ping <head_pc_ip>
```

**Low GPU utilization:**
```bash
# Increase Monte Carlo scenarios
export MC_SCENARIOS=200000
python ray_distributed_kelly_bot.py
```

**Memory issues:**
```bash
# Reduce batch size
export BATCH_HOURS=50
export N_BOTS=1000
```

### Performance Tuning

**For higher CPU usage:**
- Increase `N_PROCESSORS` (batch processors)
- Reduce `BATCH_HOURS` for more parallel batches
- Increase `N_BOTS` for more concurrent work

**For higher GPU usage:**
- Increase `monte_carlo_scenarios` in TradingParameters
- Use larger batch sizes for GPU operations
- Enable mixed precision training

### Monitoring Commands

```bash
# Check Ray status
ray status

# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor CPU usage
htop

# Check cluster resources
python -c "import ray; ray.init('auto'); print(ray.cluster_resources())"
```

## File Structure

```
TaskmasterForexBots/
├── kelly_monte_bot.py              # Core bot implementation
├── ray_distributed_kelly_bot.py    # Ray distributed version
├── ray_cluster_monitor.py          # Performance monitoring
├── start_ray_cluster.sh           # Cluster management script
├── data/
│   └── EURUSD_H1.csv              # Historical data
├── logs/
│   ├── ray_distributed_kelly.log  # Simulation logs
│   └── ray_cluster_monitor.log    # Performance logs
└── results/
    ├── ray_distributed_results_*.json
    └── ray_cluster_performance_*.json
```

## Expected Output

### Simulation Start
```
RAY DISTRIBUTED KELLY MONTE CARLO FOREX BOT SYSTEM
================================================================================
MAXIMUM CLUSTER UTILIZATION MODE
Target: 75% CPU + GPU usage across all nodes
================================================================================
Configuration:
- Fleet size: 2,000 bots
- Simulation: 5,000 hours  
- Batch size: 100 hours
- Batch processors: 8

Ray Cluster Connected:
- Total CPUs: 96.0
- Total GPUs: 2.0
- Active nodes: 2
```

### Performance Summary
```
==========================================
CLUSTER PERFORMANCE SUMMARY
==========================================
Ray Cluster:
  CPU Usage: 76.2% (73.2/96.0 cores)
  GPU Usage: 78.5% (1.6/2.0 GPUs)
  Active Nodes: 2/2
Local System:
  CPU: 75.8% (80 cores)
  Memory: 45.2% (128.5GB available)
  Load Avg: 60.45
GPU 0 (NVIDIA GeForce RTX 3090):
  Usage: 78.3%
  VRAM: 82.1% (19,712MB/24,576MB)
  Temp: 72°C
==========================================
```

### Final Results
```
RAY DISTRIBUTED SIMULATION COMPLETED
================================================================================
Total time: 1,847.3 seconds
Hours processed: 5,000
Processing rate: 2.7 hours/second
Total trades: 47,832
Total decisions: 1,247,891
Active bots: 1,847/2,000
Average return: 12.4%
Fleet total PnL: $24,759,384.50
================================================================================
```

## Support

For issues or optimization help:

1. Check the performance monitor output
2. Review Ray dashboard metrics
3. Examine log files in `logs/` directory
4. Verify cluster connectivity with `ray status`

This system is designed to push your hardware to 75% utilization while maintaining stability and producing meaningful trading results.
