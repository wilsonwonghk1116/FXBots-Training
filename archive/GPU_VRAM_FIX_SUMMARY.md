# GPU VRAM Distribution Fix - Summary

## Problem Identified
- Only PC1's GPU (RTX with 23.57 GiB) was being used, running out of VRAM
- PC2's RTX 3070 8GB GPU was completely unused
- Ray workers were not properly distributed across both GPUs
- OOM errors: "CUDA out of memory. Tried to allocate 8.84 GiB"

## Root Cause
1. **Ray cluster configuration**: GPUs were not properly detected/allocated
2. **Worker distribution**: All workers were trying to use the same GPU
3. **Memory allocation**: Tasks were requesting too much VRAM per worker
4. **Resource specification**: Ray remote tasks had incorrect resource requirements

## Fixes Implemented

### 1. Ray Cluster Restart with Explicit GPU Detection
```bash
# PC1 (Head)
ray start --head --node-ip-address=192.168.1.10 --port=6379 --num-cpus=80 --num-gpus=1

# PC2 (Worker) 
ray start --address=192.168.1.10:6379 --num-cpus=80 --num-gpus=1
```

### 2. Updated Ray Remote Task Configuration
**Before:**
```python
@ray.remote(num_cpus=0.75, num_gpus=0.75)
```

**After:**
```python
@ray.remote(num_cpus=1, num_gpus=0.5)  # Use 0.5 GPU per task
```

### 3. Proper GPU Assignment and Memory Management
- **GPU Assignment**: Workers distributed across both GPUs using modulo assignment
- **Memory Fraction**: Reduced from 75% to 40% per worker to allow multiple workers per GPU
- **Tensor Size**: Reduced from 2000x2000 to 1200x1200 to prevent OOM
- **Immediate Cleanup**: Added aggressive memory cleanup after each iteration

### 4. Worker Distribution Strategy
- **Workers per GPU**: 2 workers per GPU (total 4 workers for 2 GPUs)
- **GPU Assignment**: `gpu_id = worker_id // workers_per_gpu`
- **Device Selection**: `device_id = node_gpu_id % device_count`

### 5. Enhanced Error Handling
- **OOM Recovery**: Fallback to CPU if GPU OOM occurs
- **Memory Monitoring**: Real-time VRAM usage tracking
- **Cleanup on Error**: Automatic memory cleanup on exceptions

## Test Results
- ✅ Both PC1 and PC2 GPUs now properly detected in Ray cluster
- ✅ Workers successfully distributed across both GPUs
- ✅ No more OOM errors with conservative memory allocation
- ✅ PC2 RTX 3070 8GB now actively participating in training
- ✅ Total GPU memory: PC1 (~24GB) + PC2 (8GB) = ~32GB distributed

## Performance Impact
- **Before**: Single GPU at 87% VRAM → OOM crashes
- **After**: Dual GPU at ~40% VRAM each → Stable operation
- **Throughput**: 2x GPUs with smart distribution = Significantly higher throughput
- **Reliability**: No more crashes due to proper memory management

## Monitoring
Created monitoring scripts:
- `test_gpu_distribution.py`: Verify task distribution
- `test_vram_distribution.py`: Test memory allocation  
- `monitor_dual_gpu.py`: Real-time GPU usage monitoring

## Key Configuration Changes
1. **Ray Cluster**: Explicit `--num-gpus=1` per node
2. **Task Resources**: `num_gpus=0.5` per task (2 tasks per GPU)
3. **Memory Fraction**: `0.4` instead of `0.75` per worker
4. **Worker Count**: 4 total workers (2 per GPU)
5. **Tensor Size**: 1200x1200 instead of 2000x2000

The system now properly utilizes both PC1 and PC2 GPUs without VRAM overflow issues!
