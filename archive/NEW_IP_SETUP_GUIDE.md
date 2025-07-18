# 🚀 NEW LAN CABLE CLUSTER SETUP GUIDE

## 🔧 NEW NETWORK CONFIGURATION
- **Head PC (RTX 3090)**: 192.168.1.10
- **Worker PC (RTX 3070)**: 192.168.1.11  
- **Connection**: Direct LAN cable (end-to-end)

---

## 📋 STEP-BY-STEP SETUP INSTRUCTIONS

### 🎯 STEP 1: KILL ALL EXISTING RAY PROCESSES (BOTH PCs)

```bash
# Run on BOTH Head PC AND Worker PC
ray stop --force
pkill -f "ray::" 2>/dev/null || true
pkill -f "raylet" 2>/dev/null || true
sleep 3

# Double check no Ray processes remain
ps aux | grep ray
```

### 🎯 STEP 2: START HEAD NODE (192.168.1.10 ONLY)

```bash
# Run ONLY on Head PC (192.168.1.10)
ray start --head \
    --node-ip-address=192.168.1.10 \
    --port=6379 \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265 \
    --object-manager-port=8076 \
    --gcs-server-port=8077 \
    --raylet-port=8078 \
    --min-worker-port=10002 \
    --max-worker-port=19999 \
    --verbose
```

**✅ Expected output:**
```
Ray runtime started.
View the Ray dashboard at http://192.168.1.10:8265
```

### 🎯 STEP 3: CONNECT WORKER NODE (192.168.1.11 ONLY)

```bash
# Run ONLY on Worker PC (192.168.1.11)
ray start --address="192.168.1.10:6379" \
    --node-ip-address=192.168.1.11 \
    --object-manager-port=8076 \
    --raylet-port=8078 \
    --min-worker-port=10002 \
    --max-worker-port=19999 \
    --verbose
```

**✅ Expected output:**
```
Ray runtime started. Using existing Ray cluster.
```

### 🎯 STEP 4: VERIFY CLUSTER STATUS

```bash
# Run on Head PC (192.168.1.10) to check cluster
ray status
```

**✅ Expected output:**
```
======== Autoscaler status: 2024-xx-xx xx:xx:xx ========
Node status
---------------------------------------------------------------
Healthy:
 1 node_xxx (192.168.1.10): xxx CPU, 1 GPU
 1 node_yyy (192.168.1.11): xxx CPU, 1 GPU
...
Resources
---------------------------------------------------------------
Total Usage:
 X.0/XX CPUs
 0.0/2.0 GPUs
```

### 🎯 STEP 5: TEST CLUSTER WITH AUTOMATED SCRIPT

```bash
# Run on Head PC
python test_new_ray_cluster.py
```

**✅ Expected output:**
```
🔬 NEW RAY CLUSTER TEST STARTING...
🖥️  Local machine IP: 192.168.1.10
🎯 Running on HEAD PC
✅ Connected to Ray cluster!
📊 Cluster resources: {'CPU': 96.0, 'GPU': 2.0, ...}
🏷️  Available nodes: 2
🎊 RAY CLUSTER TEST COMPLETED SUCCESSFULLY!
```

---

## 🚀 AUTOMATED SETUP (RECOMMENDED)

Instead of manual steps, use the automated script:

```bash
# Make executable and run
chmod +x setup_new_ray_cluster.sh
./setup_new_ray_cluster.sh
```

This script automatically detects whether you're on Head PC or Worker PC and runs the appropriate commands.

---

## 🎯 START FOREX TRAINING

Once cluster is verified working:

```bash
# Set environment variable and start training
RAY_CLUSTER=1 python run_stable_85_percent_trainer.py
```

---

## 🔍 TROUBLESHOOTING

### ❌ Problem: "Connection refused"
**Solution:**
```bash
# Check firewall (Ubuntu)
sudo ufw status
sudo ufw allow 6379
sudo ufw allow 8265
sudo ufw allow 8076-8078
sudo ufw allow 10002:19999/tcp
```

### ❌ Problem: "Address already in use"
**Solution:**
```bash
# Kill all processes using Ray ports
sudo lsof -ti:6379 | xargs sudo kill -9
sudo lsof -ti:8265 | xargs sudo kill -9
sudo lsof -ti:8076 | xargs sudo kill -9
sudo lsof -ti:8077 | xargs sudo kill -9
sudo lsof -ti:8078 | xargs sudo kill -9
```

### ❌ Problem: "Worker disconnects immediately"
**Solution:**
```bash
# Check network connectivity
ping 192.168.1.10  # From worker
ping 192.168.1.11  # From head
telnet 192.168.1.10 6379  # Test Ray port
```

---

## 🎊 SUCCESS CRITERIA

Your cluster is ready when:
- ✅ `ray status` shows 2 healthy nodes
- ✅ Dashboard accessible at http://192.168.1.10:8265
- ✅ Test script completes without errors
- ✅ Both GPUs (RTX 3090 + RTX 3070) visible in cluster resources
- ✅ Training starts without "connection refused" errors

---

## 💡 QUICK COMMANDS CHEATSHEET

```bash
# Kill everything Ray-related
ray stop --force && pkill -f "ray::" && pkill -f "raylet"

# Start head (192.168.1.10)
ray start --head --node-ip-address=192.168.1.10 --port=6379

# Connect worker (192.168.1.11)  
ray start --address="192.168.1.10:6379" --node-ip-address=192.168.1.11

# Check status
ray status

# Test cluster
python test_new_ray_cluster.py

# Start training
RAY_CLUSTER=1 python run_stable_85_percent_trainer.py
```

---

**🔥 REMEMBER: 你而家用緊新IP，所以一定要用新嘅commands！舊嘅IP (192.168.1.10) 已經無效！🔥** 