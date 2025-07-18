# 🚀 DUAL PC CLUSTER TESTING INSTRUCTIONS

## 📋 OVERVIEW

You now have **3 scripts** to test and verify your dual PC cluster setup:

1. **`test_ray_cluster_connection.py`** - Test Ray cluster connectivity
2. **`run_dual_pc_training.py`** - Run distributed training across both PCs  
3. **`verify_dual_pc_gpu_usage.py`** - Monitor GPU usage in real-time

---

## 🔧 STEP 1: Test Ray Cluster Connection

**Run this first to verify your cluster is working:**

```bash
python test_ray_cluster_connection.py
```

**What it does:**
- ✅ Connects to your Ray cluster at `192.168.1.10:6379`
- ✅ Shows cluster resources (CPUs, GPUs, Memory)
- ✅ Lists all connected nodes 
- ✅ Tests task execution across different machines
- ✅ Verifies dual PC setup is detected

**Expected output for working dual PC:**
```
🚀 === RAY CLUSTER CONNECTION TEST ===
✅ Successfully connected to Ray cluster!
📊 === CLUSTER INFORMATION ===
📡 Total Nodes: 2
🖥️ Total CPUs: 32
🎮 Total GPUs: 2
🎉 === DUAL PC SETUP DETECTED ===
✅ Found 2 GPUs across cluster
🎉 SUCCESS: Tasks executed on 2 different machines!
   Hostnames: PC1-hostname, PC2-hostname
✅ Your dual PC cluster is working correctly!
```

---

## 🎮 STEP 2: Run Dual PC Training

**Once cluster test passes, run distributed training:**

```bash
python run_dual_pc_training.py
```

**What it does:**
- ✅ Connects to your existing Ray cluster
- ✅ Creates 2,000 trading bots distributed across both PCs
- ✅ Runs parallel evaluation using both GPUs
- ✅ Shows real-time cluster GPU monitoring
- ✅ Saves champion bots automatically

**Expected output:**
```
🚀 === DUAL PC DISTRIBUTED TRAINING ===
🔗 Connecting to Ray cluster at 192.168.1.10:6379...
✅ Connected to Ray cluster!
📡 Nodes: 2
🎮 GPUs: 2
🎉 Dual PC setup detected!
🧬 Creating population of 2000 bots...
🎮 === GPU CLUSTER STATUS ===
   🖥️ PC1-hostname:
      🎮 NVIDIA GeForce RTX 3090: 75.2% 🔥 ACTIVE
   🖥️ PC2-hostname:  
      🎮 NVIDIA GeForce RTX 3070: 68.1% 🔥 ACTIVE
```

---

## 📊 STEP 3: Monitor GPU Usage (Optional)

**In a separate terminal, monitor real-time GPU usage:**

```bash
python verify_dual_pc_gpu_usage.py
```

**What it shows:**
- 🔥 Live GPU utilization across both PCs
- 📈 VRAM usage and temperatures
- ✅ Verification that both PCs are working
- 📊 Real-time cluster status

---

## 🎯 WHAT TO LOOK FOR

### ✅ **SUCCESS INDICATORS:**

1. **Cluster Test:**
   - Multiple nodes detected
   - Multiple GPUs found
   - Tasks execute on different hostnames

2. **Training:**
   - Both GPUs show >50% utilization
   - Different hostnames in cluster status
   - Bots distributed across nodes

3. **GPU Monitor:**
   - Multiple GPUs showing "🔥 ACTIVE"
   - "VERIFIED - Both PCs using GPUs!" message

### ⚠️ **TROUBLESHOOTING:**

**If only 1 GPU detected:**
- Check PC2 worker node is connected: `ray status`
- Restart worker on PC2: `ray start --address=192.168.1.10:6379`

**If Python version mismatch:**
- Use same Python version on both PCs
- Or restart cluster with: `ray stop && ray start --head`

**If GPUs idle during training:**
- Training may not have started properly
- Check Ray dashboard: `http://192.168.1.10:8265`

---

## 🏃‍♂️ QUICK TEST SEQUENCE

```bash
# 1. Test cluster (should show 2 nodes, 2 GPUs)
python test_ray_cluster_connection.py

# 2. Start training (should use both GPUs)  
python run_dual_pc_training.py

# 3. Monitor in separate terminal (should show both PCs active)
python verify_dual_pc_gpu_usage.py
```

---

## 🎉 SUCCESS CONFIRMATION

**You'll know your dual PC setup is working when you see:**

✅ **2 Nodes detected** in cluster test  
✅ **2 GPUs detected** in cluster resources  
✅ **Different hostnames** in task execution  
✅ **Both GPUs >50% load** during training  
✅ **"VERIFIED - Both PCs using GPUs!"** in monitor  

**At this point, your dual PC cluster is fully operational!** 🚀

---

## 📞 IF YOU NEED HELP

**Share the output of:**
```bash
python test_ray_cluster_connection.py
```

This will show exactly what's detected and help diagnose any issues.

**Your dual PC setup should give you roughly 2x the training performance compared to single PC!** 🎯 