# PROJECT STATE BACKUP - 2025年7月17日 凌晨02:08
## 黃子華 Style Project Progress Report 👑

**Status**: 99% Ready for Production Training (就差最後一啖氣！)

---

## CRITICAL NETWORK DISCOVERY 🌐

**❌ NEVER USE THESE WiFi 7 IPs (會自動斷線):**
- Head PC 1: `192.168.1.10` 
- Worker PC 2: `192.168.1.11`
- Problem: Worker PC 2 disconnects within 2 minutes on WiFi 7

**✅ ALWAYS USE LAN Cable IPs (穩定如泰山):**
- Head PC 1: `192.168.1.10` 
- Worker PC 2: `192.168.1.11`
- Solution: Direct LAN cable connection between PCs

---

## MAJOR FIXES COMPLETED ✅

### 1. **Config.py Path Detection Fixed**
```python
def get_project_root():
    """Auto-detect correct project path based on which PC we're on"""
    import socket
    hostname = socket.gethostname()
    if 'w1' in hostname or os.path.exists('/home/w1/cursor-to-copilot-backup'):
        return '/home/w1/cursor-to-copilot-backup/TaskmasterForexBots'
    elif 'w2' in hostname or os.path.exists('/home/w2/cursor-to-copilot-backup'):
        return '/home/w2/cursor-to-copilot-backup/TaskmasterForexBots'
    else:
        return os.path.dirname(os.path.abspath(__file__))
```

### 2. **Ray Large File Packaging Fixed**
Added excludes to prevent uploading 15MB+ files:
```python
'excludes': [
    '*.pth',          # Exclude all model files
    '*.tar.gz',       # Exclude compressed files
    'CHAMPION_BOT_*.pth',
    'CHAMPION_ANALYSIS_*.json', 
    'champion_gen*.pth',
    'checkpoint_gen*.pth',
    'data/EURUSD_H1.csv',  # Local data files
    '.git/',          # Git objects
]
```

### 3. **Environment Setup Enhanced**
- ✅ Both PCs have `BotsTraining_env` conda environment
- ✅ Python 3.12.2 on both nodes
- ✅ Ray 2.47.1 installed and working
- ✅ synthetic_env module available
- ✅ EURUSD_H1.csv data file present on both PCs

---

## FILES UPDATED AND SYNCED 📁

### Head PC 1 Files:
- ✅ `config.py` - Fixed multi-PC path detection
- ✅ `run_stable_85_percent_trainer.py` - Added Ray excludes
- ✅ `launch_distributed_training.py` - Ready for testing
- ✅ `test_worker_environment.py` - Environment verification
- ✅ `setup_ray_cluster_with_env.py` - Automated setup

### Worker PC 2 Files (Synced):
- ✅ `config.py` - Copied and verified working
- ✅ `run_stable_85_percent_trainer.py` - Latest version synced  
- ✅ `test_worker_environment.py` - Available for testing
- ✅ Data files: `data/EURUSD_H1.csv` (17MB) exists

---

## POST-REBOOT STARTUP SEQUENCE 🚀

### Step 1: Network Verification
```bash
# Head PC 1 (192.168.1.10)
ping 192.168.1.11

# Worker PC 2 (192.168.1.11)  
ping 192.168.1.10
```

### Step 2: Activate Conda Environment (Both PCs)
```bash
conda activate BotsTraining_env
cd /home/w1/cursor-to-copilot-backup/TaskmasterForexBots  # Head PC
cd /home/w2/cursor-to-copilot-backup/TaskmasterForexBots  # Worker PC
```

### Step 3: Start Ray Cluster (LAN IPs Only!)
```bash
# Head PC 1 - Start head node
ray start --head --node-ip-address=192.168.1.10 --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265

# Worker PC 2 - Join cluster  
ray start --address='192.168.1.10:6379'
```

### Step 4: Verify Cluster
```bash
# Head PC 1 - Check status
ray status
# Should show: 96 CPUs, 2 GPUs, 2 nodes
```

### Step 5: Launch Training
```bash
# Head PC 1 - Start distributed training
cd TaskmasterForexBots
python launch_distributed_training.py
```

---

## EXPECTED PERFORMANCE TARGETS 🎯

- **Total Population**: 5,000-8,000 bots per generation
- **Generations**: 300 total
- **Head PC (RTX 3090)**: 3,000-5,000 bots
- **Worker PC (RTX 3070)**: 2,000-3,000 bots  
- **Training Time**: 4-6 hours estimated
- **Champion Bot**: Auto-saved with analysis

---

## TROUBLESHOOTING NOTES 🔧

### If Ray Connection Fails:
1. Check firewall: `sudo ufw status`
2. Verify IPs: `ip addr show`
3. Test network: `ping <target_ip>`
4. Restart Ray completely: `ray stop && ray start --head...`

### If Data File Missing:
```bash
# Copy from Head PC if needed
sshpass -p 'w' scp /home/w1/cursor-to-copilot-backup/TaskmasterForexBots/data/EURUSD_H1.csv w2@192.168.1.11:/home/w2/cursor-to-copilot-backup/TaskmasterForexBots/data/
```

### If Config Wrong:
```bash
# Test config on both PCs
python3 -c "from config import PROJECT_ROOT, EURUSD_H1_PATH; print(f'ROOT: {PROJECT_ROOT}'); print(f'DATA: {EURUSD_H1_PATH}')"
```

---

## 寸人技巧進步報告 😏

**User 要求我提升寸人技巧，所以:**

1. **PowerShell 黃子華評價**: "呢個 PowerShell 喺 Ubuntu 上面嘅表現，就好似黃子華去踢世界盃咁樣 - 有心無力，搞到全場都好尷尬！"

2. **WiFi 7 連接穩定性**: "個 WiFi 7 connection 就好似港女嘅愛情觀咁樣 - 理論上好勁，但實際用起上嚟 2 分鐘就斷線！"

3. **Ray Cluster 狀態**: "而家個 Ray cluster 就好似黃子華嘅演出咁樣 - 台下坐滿人，但係台上嘅人都唔知道講乜！"

4. **兩部 PC 協作**: "兩部 Ubuntu PC 要合作，就好似要黃子華同鄭中基一齊主持節目咁樣 - 理論上可行，但係需要好多 rehearsal！"

---

## CRITICAL SUCCESS FACTORS ⚠️

1. **必須用 LAN cable IPs** (192.168.1.10/11)
2. **兩部機都要 reboot 清 Ray processes**  
3. **Conda environment 必須 activate**
4. **PowerShell bugs 係正常，用 bash 就 OK**

---

**黃子華金句總結**: "做人最緊要堅持，第二緊要就係識得幾時要 reboot！" 👑

**準備重新開始嘅時候記住**: 我哋已經 99% ready，只係需要乾淨嘅 Ray cluster 同埋正確嘅 LAN IPs！

**Post-reboot 回來時記得話俾我知**: "我 reboot 完啦，繼續寸我啦！" 😂 