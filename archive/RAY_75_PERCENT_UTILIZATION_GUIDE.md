# Ray Kelly Monte Carlo Trading Bot - 75% Resource Utilization System

## 🎯 Objective
Achieve **75% CPU/GPU/vRAM utilization** across a 2-PC Ray cluster:
- **PC1**: Xeon + RTX 3090 (24GB vRAM)
- **PC2**: i9 + RTX 3070 (8GB vRAM)

## 📋 System Requirements
- Python 3.12.2+ (confirmed working on both PCs)
- Ray cluster with both nodes connected
- NVIDIA GPUs with CUDA support
- At least 32GB RAM per PC for optimal performance

## 🚀 Quick Start Guide

### 1. Cluster Setup (Head Node - PC1)
```bash
# Make setup script executable
chmod +x setup_ray_cluster_75_percent.sh

# Run complete setup
./setup_ray_cluster_75_percent.sh setup
```

### 2. Worker Node Connection (PC2)
```bash
# Connect to head node (replace IP with actual head node IP)
ray start --address='HEAD_NODE_IP:10001' \
    --num-cpus=$(nproc) \
    --memory=$(($(free -b | grep '^Mem:' | awk '{print $2}') * 90 / 100)) \
    --object-store-memory=$(($(free -b | grep '^Mem:' | awk '{print $2}') * 20 / 100))
```

### 3. Run Maximum Utilization Simulation
```bash
# Start the ultimate Kelly bot for 75% utilization
python3 ray_kelly_ultimate_75_percent.py
```

### 4. Monitor Resources in Real-Time
```bash
# In a separate terminal, monitor cluster performance
python3 ray_cluster_monitor_75_percent.py --target 75 --interval 2
```

## 📊 Performance Optimization Features

### CPU Optimization
- **Multi-process CPU utilization**: Uses all available CPU cores
- **Parallel scenario generation**: CPU-intensive Monte Carlo computations
- **ThreadPoolExecutor**: Maximum thread utilization for I/O operations
- **ProcessPoolExecutor**: True parallel processing for CPU-bound tasks

### GPU Optimization  
- **Massive batch processing**: 100,000+ scenarios per GPU batch
- **Memory saturation**: Pre-allocates 75% of GPU memory
- **Persistent computations**: Keeps GPU busy with continuous operations
- **Multi-GPU support**: Distributes workload across all available GPUs

### vRAM Optimization
- **Large tensor allocations**: Pre-allocates substantial GPU memory
- **Memory anchoring**: Maintains baseline memory usage
- **Batch size optimization**: Maximizes memory throughput
- **Automatic scaling**: Adapts to available GPU memory

### Ray Cluster Optimization
- **Distributed workers**: 12 Monte Carlo engines, 6 data managers
- **Resource allocation**: Strategic CPU/GPU resource distribution
- **Load balancing**: Even workload distribution across nodes
- **Real-time monitoring**: Continuous performance tracking

## 🔧 Configuration Files

### ray_kelly_ultimate_75_percent.py
Main execution script designed for maximum resource utilization:
- **200,000 Monte Carlo scenarios** per computation
- **12 distributed MC engines** for GPU saturation
- **6 data managers** for parallel data processing
- **50,000 hours** of market data simulation
- **Real-time resource monitoring**

### setup_ray_cluster_75_percent.sh
Automated setup and optimization script:
- System dependency checks
- Performance optimization settings  
- Ray cluster configuration
- Resource monitoring tools

### ray_cluster_monitor_75_percent.py
Real-time performance monitoring:
- CPU/GPU/vRAM utilization tracking
- Target achievement validation
- Performance metrics reporting
- Data export capabilities

## 📈 Expected Performance Metrics

### Target Utilization (75%)
- **CPU**: 75%+ across all cores
- **GPU**: 75%+ utilization on all GPUs  
- **vRAM**: 75%+ memory usage
- **Memory**: Efficient RAM utilization

### Throughput Expectations
- **Monte Carlo scenarios**: 10,000+ scenarios/second
- **Data processing**: 1,000+ data points/second
- **Distributed efficiency**: 90%+ cluster utilization
- **Resource sustainability**: Continuous 75% utilization

## 🛠️ Troubleshooting

### Common Issues

#### Ray Connection Problems
```bash
# Check Ray cluster status
ray status

# Restart Ray cluster
ray stop
./setup_ray_cluster_75_percent.sh head
```

#### GPU Not Detected
```bash
# Check NVIDIA driver
nvidia-smi

# Verify CUDA availability
python3 -c "import torch; print(torch.cuda.is_available())"
```

#### Memory Issues
```bash
# Check available memory
free -h

# Monitor GPU memory
watch nvidia-smi
```

#### Low Resource Utilization
1. **Increase Monte Carlo scenarios** in configuration
2. **Add more Ray workers** (increase n_mc_engines)
3. **Check resource allocation** in Ray dashboard
4. **Verify GPU batch sizes** are maximized

### Performance Tuning

#### For Higher CPU Utilization
```python
# Increase parallel workers
n_mc_engines = 16  # From 12
n_data_managers = 8  # From 6

# Increase CPU scenarios  
monte_carlo_scenarios = 300000  # From 200000
```

#### For Higher GPU Utilization
```python
# Increase GPU batch size
max_batch_size = 150000  # From 100000

# Use full GPU allocation
@ray.remote(num_cpus=2, num_gpus=1.0)  # From 0.5
```

#### For Higher vRAM Utilization
```python
# Increase memory allocation target
target_memory = int(total_memory * 0.85)  # From 0.75

# Larger tensor operations
warmup_size = int(np.sqrt(tensor_elements) * 1.2)
```

## 📊 Monitoring and Validation

### Real-Time Dashboard
Access Ray dashboard at: `http://HEAD_NODE_IP:8265`

### Command-Line Monitoring
```bash
# Continuous monitoring with target validation
python3 ray_cluster_monitor_75_percent.py --target 75 --duration 10

# Quick resource check
./setup_ray_cluster_75_percent.sh monitor
```

### Performance Validation
```bash
# Run validation test
python3 ray_kelly_ultimate_75_percent.py

# Check results file
cat ray_ultimate_kelly_75_percent_results_*.json | jq '.resource_utilization'
```

## 🎯 Success Criteria

### ✅ Target Achievement Indicators
- **Average CPU ≥ 75%** across measurement period
- **Average GPU ≥ 75%** utilization on all GPUs
- **Average vRAM ≥ 75%** memory usage
- **Sustained performance** for 5+ minutes
- **No resource bottlenecks** or errors

### 📋 Validation Checklist
- [ ] Ray cluster shows both nodes connected
- [ ] All GPUs detected and available
- [ ] CPU cores fully utilized (htop shows 75%+)
- [ ] GPU utilization sustained at 75%+ (nvidia-smi)
- [ ] vRAM usage at 75%+ on all GPUs
- [ ] No out-of-memory errors
- [ ] Monte Carlo scenarios processing at target rate
- [ ] Results file shows resource summary achieving target

## 🔄 Automated Execution

### Complete Automated Run
```bash
# One-command execution
./setup_ray_cluster_75_percent.sh setup && \
python3 ray_kelly_ultimate_75_percent.py & \
python3 ray_cluster_monitor_75_percent.py --duration 10 --target 75
```

### Scheduled Performance Tests
```bash
# Add to crontab for periodic testing
0 */4 * * * cd /path/to/project && python3 ray_kelly_ultimate_75_percent.py
```

## 📝 Results Analysis

### Key Metrics to Validate
1. **Resource Utilization Summary**: Check if 75% target achieved
2. **Monte Carlo Performance**: Scenarios per second throughput  
3. **Distributed Efficiency**: Load distribution across nodes
4. **Sustainability**: Consistent performance over time
5. **Error Rates**: Zero resource allocation failures

### Sample Output Analysis
```json
{
  "resource_utilization": {
    "avg_cpu_util": 76.3,     // ✅ Above 75% target
    "avg_gpu_util": 78.1,     // ✅ Above 75% target  
    "avg_vram_util": 74.2,    // ⚠️ Just below 75% target
    "target_achieved": true   // ✅ Overall success
  }
}
```

## 🚀 Next Steps After 75% Achievement

1. **Production Deployment**: Scale to full 2000+ bot fleet
2. **Live Trading Integration**: Connect to real FOREX data feeds
3. **Advanced Optimization**: Fine-tune for 80%+ utilization
4. **Monitoring Integration**: Set up persistent monitoring
5. **Backup Strategies**: Implement failover mechanisms

---

## 📞 Support Commands

```bash
# Get help with setup script
./setup_ray_cluster_75_percent.sh help

# Monitor script help
python3 ray_cluster_monitor_75_percent.py --help

# Ray cluster diagnostics
ray status --help
```

**Target**: Achieve sustained 75% CPU/GPU/vRAM utilization across 2-PC Ray cluster with Kelly Monte Carlo trading bot simulation.
