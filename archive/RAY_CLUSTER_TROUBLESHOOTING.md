# 🔧 RAY CLUSTER TROUBLESHOOTING GUIDE

## Problem: Connection Failed to 192.168.1.10:6379

**Error Message:**
```
Failed to connect to GCS at address 192.168.1.10:6379 within 5 seconds
```

## 🚀 QUICK SOLUTIONS

### Option 1: Use the Helper Script
```bash
python start_local_ray_cluster.py
```
This will:
- Auto-detect your IP address
- Start Ray head node properly
- Show connection info for PC2

### Option 2: Start Ray Manually
```bash
ray start --head --node-ip-address=192.168.1.10
```

### Option 3: Check Current Ray Status
```bash
ray status
```

## 🔍 DIAGNOSTIC STEPS

### Step 1: Check if Ray is Running
```bash
ray status
```

**Expected Output (if working):**
```
Node status:
======================================================================
Active nodes:
-----------------------------------------------------------------------
NODE_IP:       192.168.1.10
NODE_NAME:     HOSTNAME-HERE
Alive:         True
Resources:     {'CPU': 8.0, 'GPU': 1.0, 'memory': 16384.0, 'object_store_memory': 8192.0}
```

### Step 2: Check Network Connectivity
```bash
ping 192.168.1.10
telnet 192.168.1.10 6379
```

### Step 3: Check Your IP Address
```bash
hostname -I
ip route get 1.1.1.1 | awk '{print $7; exit}'
```

### Step 4: Test Flexible Connection
```bash
python test_ray_cluster_flexible.py
```

## 🛠️ COMMON FIXES

### Fix 1: Restart Ray Completely
```bash
ray stop
ray start --head --node-ip-address=192.168.1.10
```

### Fix 2: Use Your Actual IP
Get your IP:
```bash
hostname -I
```
Then start Ray with that IP:
```bash
ray start --head --node-ip-address=YOUR_ACTUAL_IP
```

### Fix 3: Check Firewall
```bash
sudo ufw status
sudo ufw allow 6379
sudo ufw allow 8265
```

### Fix 4: Check for Port Conflicts
```bash
netstat -tulpn | grep 6379
lsof -i :6379
```

## 🖥️ DUAL PC SETUP

### On PC1 (Head Node):
```bash
# Start head node
ray start --head --node-ip-address=192.168.1.10
```

### On PC2 (Worker Node):
```bash
# Connect to head node
ray start --address=192.168.1.10:6379
```

### Verify Dual PC Setup:
```bash
python test_ray_cluster_flexible.py
```

**Expected Output:**
```
✅ Nodes: 2
✅ GPUs: 2
✅ Hosts: PC1-hostname, PC2-hostname
🚀 === DUAL PC SETUP DETECTED ===
```

## 📊 Dashboard Access

**Local Access:**
```
http://127.0.0.1:8265
```

**Network Access:**
```
http://192.168.1.10:8265
```

## 🚨 EMERGENCY RESET

If everything fails:
```bash
# Kill all Ray processes
pkill -f ray
ray stop --force

# Clear Ray temp files
rm -rf /tmp/ray/

# Restart fresh
ray start --head --node-ip-address=192.168.1.10
```

## 🔄 ALTERNATIVE CONNECTION MODES

### Mode 1: Auto-connect (if Ray is local)
```python
ray.init(address='auto')
```

### Mode 2: Specific IP
```python
ray.init(address='192.168.1.10:6379')
```

### Mode 3: Localhost (single PC)
```python
ray.init(address='localhost:6379')
```

## 🎯 SUCCESS VERIFICATION

Run this after any fix:
```bash
python test_ray_cluster_flexible.py
```

**Success indicators:**
- ✅ Connection successful
- ✅ 2 nodes detected (dual PC)
- ✅ 2 GPUs detected
- ✅ Different hostnames

## 📋 FINAL CHECKLIST

- [ ] Ray head node running on PC1
- [ ] Ray worker connected from PC2
- [ ] Both PCs show in `ray status`
- [ ] Dashboard accessible
- [ ] Connection test passes
- [ ] Ready for `python run_dual_pc_training.py`

## 💡 TIPS

1. **Always start head node first** on PC1
2. **Wait 5-10 seconds** before connecting PC2
3. **Use actual IP addresses**, not 127.0.0.1
4. **Check dashboard** for visual cluster status
5. **Test connection** before starting training 