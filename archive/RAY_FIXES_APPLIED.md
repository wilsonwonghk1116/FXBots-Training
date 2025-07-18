# Ray Cluster Command Fixes Applied
==========================================

## Fixed Commands

### 1. Ray Head Node Start Command
**Before:** `ray start --head --port=10001`
**After:** `ray start --head --node-ip-address=192.168.1.10`

### 2. Ray Worker Connection Command  
**Before:** 
```bash
sshpass -p 'w' ssh w1@192.168.1.11 'ray start --address=192.168.1.10:10001'
```

**After:**
```bash
sshpass -p 'w' ssh w2@192.168.1.11 "source /home/w2/miniconda3/etc/profile.d/conda.sh && conda activate Training_env && ray start --address='192.168.1.10:6379' --node-ip-address=192.168.1.11"
```

### 3. Ray Client Connection
**Before:** `ray.init(address='ray://192.168.1.10:10001')`
**After:** `ray.init(address='ray://192.168.1.10:10001')` (Port 10001 correct for client)

## Key Changes

1. **SSH User:** Changed from `w1` to `w2` for PC2
2. **Conda Path:** Updated to `/home/w2/miniconda3/etc/profile.d/conda.sh` for PC2
3. **Head Node:** Added `--node-ip-address=192.168.1.10` parameter
4. **Worker Node:** Added `--node-ip-address=192.168.1.11` parameter
5. **Ray Address:** Changed from port 10001 to 6379 for worker connection
6. **Shell:** Force bash execution with `executable='/bin/bash'`

## Files Updated

✅ **automated_cluster_training.py** - All commands fixed
✅ **launch_real_training.py** - All commands fixed  
✅ **fixed_cluster_training.py** - All commands fixed
✅ **direct_training.py** - Already had correct client address

## Ready to Test

All scripts now use the correct Ray cluster commands as specified:

```bash
# Test with any of these scripts:
python automated_cluster_training.py
python launch_real_training.py  
python fixed_cluster_training.py
```

The scripts should now properly:
1. Start Ray head node on PC1 with correct IP binding
2. Connect PC2 worker using correct SSH user and conda path
3. Use proper Ray addresses for cluster formation
