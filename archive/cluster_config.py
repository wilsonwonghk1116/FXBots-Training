# Automated Cluster Training Configuration
# ======================================

# Network Configuration - UPDATED FOR USER'S HARDWARE
PC1_IP = "192.168.1.10"    # Head PC1 - 2x Xeon + RTX 3090
PC2_IP = "192.168.1.11"    # Worker PC2 - I9 + RTX 3070
PC2_USER = "w1"

# SSH Configuration (Update with your actual password)
PC2_SSH_PASSWORD = "w"

# Ray Configuration
RAY_PORT = 6379  # Standard Ray port
RAY_DASHBOARD_PORT = 8265

# Hardware Configuration (CONSERVATIVE 60% utilization to prevent freezing)
PC1_TOTAL_CPUS = 80
PC1_TOTAL_GPUS = 1
PC1_TOTAL_VRAM_GB = 24  # RTX 3090

PC2_TOTAL_CPUS = 16
PC2_TOTAL_GPUS = 1
PC2_TOTAL_VRAM_GB = 8   # RTX 3070

# SAFE 60% Utilization Settings (to prevent freezing)
UTILIZATION_PERCENTAGE = 0.60

# Calculated 60% Resource Allocation (SAFE LIMITS)
PC1_CPUS = int(PC1_TOTAL_CPUS * UTILIZATION_PERCENTAGE)     # 48 CPUs (60% of 80)
PC1_GPUS = PC1_TOTAL_GPUS  # Use full GPU but limit VRAM to 60%
PC1_OBJECT_STORE_MEMORY = int(PC1_TOTAL_VRAM_GB * UTILIZATION_PERCENTAGE * 1024**3)  # 14.4GB (60% of 24GB)

PC2_CPUS = int(PC2_TOTAL_CPUS * UTILIZATION_PERCENTAGE)     # 10 CPUs (60% of 16)  
PC2_GPUS = PC2_TOTAL_GPUS  # Use full GPU but limit VRAM to 60%
PC2_OBJECT_STORE_MEMORY = int(PC2_TOTAL_VRAM_GB * UTILIZATION_PERCENTAGE * 1024**3)  # 4.8GB (60% of 8GB)

# Ray Actor Resource Configuration (60% utilization - SAFE)
RAY_TRAINER_CPUS = 6        # 60% of available cores per trainer
RAY_TRAINER_GPU_FRACTION = 0.60  # 60% GPU utilization per trainer
RAY_COORDINATOR_CPUS = 2    # Coordinator uses minimal resources

# Worker Configuration (optimized for 60% utilization - SAFE)
WORKERS_PER_PC = 2          # Conservative worker count
MAX_CONCURRENT_TRAINERS = 4 # Total trainers across both PCs

# GPU Memory Management (60% VRAM usage - SAFE)
PC1_GPU_MEMORY_FRACTION = 0.60  # Use 60% of RTX 3090 VRAM
PC2_GPU_MEMORY_FRACTION = 0.60  # Use 60% of RTX 3070 VRAM

# Training Optimization for 60% utilization (SAFE)
BATCH_SIZE_REDUCTION = 0.60     # Conservative batch sizes
PARALLEL_EPISODES_REDUCTION = 0.60  # Conservative parallel episodes

# Python Environment
CONDA_ENV_NAME = "Training_env"

# Training Configuration (CONSERVATIVE)
TEST_GENERATIONS = 5        # Short test
FULL_GENERATIONS = 50       # Conservative full training
TEST_POPULATION = 100       # Small test population
FULL_POPULATION = 1000      # Conservative population

# Safety Limits
MAX_GPU_TEMPERATURE = 75    # Safe temperature limit
MEMORY_SAFETY_BUFFER = 0.20 # 20% memory buffer
HEARTBEAT_INTERVAL = 30     # Health check every 30s

print("✅ Updated cluster config with correct IPs and SAFE 60% utilization limits")
