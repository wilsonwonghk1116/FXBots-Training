# CONNECTING WORKER PC 2 TO RAY CLUSTER
## Step-by-Step Instructions

### Current Network Configuration:
- **Head PC 1 IP:** 192.168.1.10 (Ray Head Node)
- **Worker PC 2 IP:** 192.168.1.11 (Ray Worker Node)
- **Connection:** Direct LAN cable

### On Worker PC 2, run these commands:

#### Option 1: Use the automated script
```bash
# Copy the script to Worker PC 2 and run:
./connect_worker_pc2.sh
```

#### Option 2: Manual connection
```bash
# 1. Activate the correct conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate BotsTraining_env

# 2. Check Python version (should be 3.12.x)
python --version

# 3. Test network connection to Head PC 1
ping -c 3 192.168.1.10

# 4. Stop any existing Ray processes
ray stop --force

# 5. Connect to the Ray cluster
ray start --address='192.168.1.10:6379' --node-ip-address=192.168.1.11 --redis-password=mypassword
```

### Verification Steps:

#### On Head PC 1, check cluster status:
```bash
ray status
```

**Expected output should show:**
- 2 active nodes
- Combined CPU count (80 + Worker PC 2 CPUs)
- 2 GPUs total (RTX 3090 + RTX 3070)

### Common Issues and Solutions:

1. **Connection refused:**
   - Check if Ray head is running on PC 1: `ray status`
   - Verify firewall settings allow port 6379
   - Ensure LAN cable is properly connected

2. **Python version mismatch:**
   - Both PCs must use Python 3.12.x
   - Ensure same conda environment on both PCs

3. **CUDA/PyTorch issues:**
   - Install same PyTorch version on both PCs
   - Verify CUDA drivers on Worker PC 2

### Performance Testing:

Once both PCs are connected, test with:
```bash
# Run dual-GPU optimization test
python rtx3090_smart_compute_optimizer_v2.py --duration=1
```

**Expected results with dual-GPU setup:**
- RTX 3090 Worker: ~50-60 TFLOPS
- RTX 3070 Worker: ~30-40 TFLOPS
- **Combined:** ~80-100 TFLOPS peak performance
- Massive improvement in parallel processing capability

### Ray Dashboard Access:
- URL: http://192.168.1.10:8265
- Monitor both nodes in real-time
- Track GPU utilization across both systems
