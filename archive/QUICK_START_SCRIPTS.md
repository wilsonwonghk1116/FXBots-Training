# 🚀 QUICK START SCRIPTS

## 📝 **1. STANDALONE PC TRAINING SCRIPT**

### Simple Command:
```bash
python start_standalone_training.py
```

### Alternative - Direct Method:
```bash
# Copy and paste this into terminal:
cd TaskmasterForexBots
python -c "
from run_production_forex_trainer import ProductionForexTrainer, DistributedGPUSaturator, DistributedCPUSaturator
import torch

# Initialize with standalone settings
trainer = ProductionForexTrainer()
trainer.population_size = min(5000, max(2000, trainer.population_size // 2))
trainer.generations = 200

# Conservative resource settings for standalone
trainer.gpu_saturators = [DistributedGPUSaturator(trainer.device, target_vram_percent=70, target_usage_percent=70)]
trainer.cpu_saturators = [DistributedCPUSaturator(target_threads=60, target_utilization=90)]

print('🚀 Starting STANDALONE training: 70% GPU, 90% CPU')

# Start training
for saturator in trainer.gpu_saturators: saturator.start_saturation()
for saturator in trainer.cpu_saturators: saturator.start_saturation()

# Run training
population = trainer.create_population()
for gen in range(trainer.generations):
    print(f'Generation {gen+1}/{trainer.generations}')
    results = trainer.evaluate_population(population)
    if gen < trainer.generations - 1:
        population = trainer.evolve_population(population, results)

print('✅ STANDALONE training complete!')
"
```

---

## 🌐 **2. UBUNTU CLUSTER TRAINING SCRIPT**

### Step 1: Start Ray Cluster

**On Primary PC (HEAD):**
```bash
# Auto-detect IP and start head node
python start_cluster_training.py
```

**Or Manual Setup:**
```bash
# Get your IP first
hostname -I | awk '{print $1}'

# Start Ray head node (replace IP with your actual IP)
ray start --head --node-ip-address=192.168.1.100
```

**On Secondary PC (WORKER):**
```bash
# Connect to head node (replace IP with head node IP)
ray start --address='192.168.1.100:6379' --num-cpus=48 --num-gpus=1
```

### Step 2: Start Training

**On Primary PC:**
```bash
# Copy and paste this into terminal:
cd TaskmasterForexBots
python -c "
import ray
from run_production_forex_trainer import ProductionForexTrainer, DistributedGPUSaturator, DistributedCPUSaturator

# Connect to cluster
ray.init(address='auto')
print('✅ Connected to Ray cluster')
print('📊 Resources:', ray.cluster_resources())

# Initialize with cluster settings
trainer = ProductionForexTrainer()
trainer.population_size = max(15000, trainer.population_size * 2)  # 2x for cluster
trainer.generations = 300

# Aggressive resource settings for cluster
trainer.gpu_saturators = [DistributedGPUSaturator(trainer.device, target_vram_percent=90, target_usage_percent=95)]
trainer.cpu_saturators = [DistributedCPUSaturator(target_threads=80, target_utilization=95)]

print('🚀 Starting CLUSTER training: 90% GPU, 95% CPU across 2 PCs')

# Start training
for saturator in trainer.gpu_saturators: saturator.start_saturation()
for saturator in trainer.cpu_saturators: saturator.start_saturation()

# Run training
population = trainer.create_population()
for gen in range(trainer.generations):
    print(f'Generation {gen+1}/{trainer.generations}')
    results = trainer.evaluate_population(population)
    if gen < trainer.generations - 1:
        population = trainer.evolve_population(population, results)

print('✅ CLUSTER training complete!')
ray.shutdown()
"
```

---

## ⚡ **SUPER SIMPLE ONE-LINERS**

### Standalone (70% GPU, 90% CPU):
```bash
cd TaskmasterForexBots && python run_production_forex_trainer.py
```

### Cluster Setup:
```bash
# PRIMARY PC:
ray start --head --num-cpus=48 --num-gpus=1 && python run_production_forex_trainer.py

# SECONDARY PC:
ray start --address='PRIMARY_PC_IP:6379' --num-cpus=48 --num-gpus=1
```

---

## 🔧 **RESOURCE MODIFICATIONS**

### To Change GPU/CPU Targets:

**In `run_production_forex_trainer.py`, modify these lines:**

```python
# Around line 1785 - GPU settings
target_vram_percent=85    # Change to 70 for standalone, 90 for cluster
target_usage_percent=85   # Change to 70 for standalone, 95 for cluster

# Around line 1792 - CPU settings  
target_threads=54         # Change to 60 for standalone, 80 for cluster
target_utilization=85     # Change to 90 for standalone, 95 for cluster

# Around line 1350 - Population size
self.population_size = 3000   # Change to 2000-5000 for standalone, 15000+ for cluster
```

---

## 📊 **QUICK COMPARISON**

| Feature | Standalone Command | Cluster Command |
|---------|-------------------|-----------------|
| **Setup** | `python start_standalone_training.py` | Ray setup + training |
| **GPU Target** | 70% VRAM, 70% usage | 90% VRAM, 95% usage |
| **CPU Target** | 60 threads at 90% | 80 threads at 95% |
| **Population** | 2K-5K bots | 15K+ bots |
| **Hardware** | Single PC | 2 PCs via Ray |

---

## 🚨 **SAFETY REMINDERS**

- Monitor GPU temperature (should stay under 80°C)
- Watch VRAM usage to prevent crashes
- Use Ctrl+C to interrupt training safely
- Ray cluster: `ray stop` to cleanup

---

## 🎯 **IMMEDIATE ACTION STEPS**

1. **For Standalone**: Run `python start_standalone_training.py`
2. **For Cluster**: 
   - Primary PC: `python start_cluster_training.py`
   - Secondary PC: Connect when prompted
3. **Monitor** training progress in terminal
4. **Champions** saved as `*_CHAMPION_BOT_*.pth` files 