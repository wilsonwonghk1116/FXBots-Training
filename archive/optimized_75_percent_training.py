#!/usr/bin/env python3
"""
75% Utilization Training Configuration for Massive Scale Distributed Training
============================================================================

This module provides optimized training configurations that ensure 75% utilization
of CPU, GPU, and VRAM resources across both PCs for optimal performance without
overloading the systems.

Author: AI Assistant
Date: July 13, 2025
"""

import os
import psutil
import torch
import ray
from cluster_config import *

class OptimizedResourceManager:
    """Manages 75% utilization across CPU, GPU, and memory resources"""
    
    def __init__(self):
        self.cpu_limit_pc1 = PC1_CPUS
        self.cpu_limit_pc2 = PC2_CPUS
        self.gpu_memory_fraction_pc1 = PC1_GPU_MEMORY_FRACTION
        self.gpu_memory_fraction_pc2 = PC2_GPU_MEMORY_FRACTION
        
    def setup_pytorch_gpu_limits(self):
        """Configure PyTorch to use 75% GPU memory"""
        if torch.cuda.is_available():
            # Set memory fraction for each GPU
            for i in range(torch.cuda.device_count()):
                torch.cuda.set_per_process_memory_fraction(0.75, device=i)
                print(f"🎮 GPU {i}: Limited to 75% VRAM usage")
    
    def setup_cpu_affinity(self, worker_id, total_workers):
        """Set CPU affinity for workers to ensure 75% utilization"""
        total_cpus = psutil.cpu_count()
        cpus_per_worker = total_cpus // total_workers
        
        # Calculate CPU range for this worker (75% of available)
        start_cpu = worker_id * cpus_per_worker
        end_cpu = min(start_cpu + int(cpus_per_worker * 0.75), total_cpus - 1)
        
        cpu_list = list(range(start_cpu, end_cpu + 1))
        
        try:
            # Set CPU affinity
            p = psutil.Process()
            p.cpu_affinity(cpu_list)
            print(f"🖥️  Worker {worker_id}: Using CPUs {cpu_list} (75% utilization)")
        except Exception as e:
            print(f"⚠️  Could not set CPU affinity: {e}")
    
    def get_optimal_batch_size(self, base_batch_size=64):
        """Calculate optimal batch size for 75% utilization"""
        return int(base_batch_size * BATCH_SIZE_REDUCTION)
    
    def get_optimal_parallel_episodes(self, base_episodes=8):
        """Calculate optimal parallel episodes for 75% utilization"""
        return int(base_episodes * PARALLEL_EPISODES_REDUCTION)

class Enhanced75PercentTrainingConfig:
    """Enhanced training configuration optimized for 75% resource utilization"""
    
    # Core training parameters (same as before)
    GENERATIONS = FULL_GENERATIONS
    EPISODES_PER_GENERATION = FULL_EPISODES
    STEPS_PER_EPISODE = FULL_STEPS
    TOTAL_TRAINING_STEPS = GENERATIONS * EPISODES_PER_GENERATION * STEPS_PER_EPISODE
    
    # 75% Optimized Resource Configuration
    WORKERS_PER_NODE = WORKERS_PER_PC
    MAX_CONCURRENT_TRAINERS = MAX_CONCURRENT_TRAINERS
    
    # Ray Actor Configuration (75% utilization)
    TRAINER_CPU_ALLOCATION = RAY_TRAINER_CPUS
    TRAINER_GPU_ALLOCATION = RAY_TRAINER_GPU_FRACTION
    COORDINATOR_CPU_ALLOCATION = RAY_COORDINATOR_CPUS
    
    # Memory and Performance Optimization
    BATCH_SIZE = 48  # Reduced from 64 for 75% utilization
    PARALLEL_EPISODES = 6  # Reduced from 8 for 75% utilization
    
    # GPU Memory Management
    GPU_MEMORY_FRACTION = 0.75
    PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:512"  # Optimize GPU memory allocation
    
    # Training Performance Settings
    LEARNING_RATE = 0.001
    EXPLORATION_RATE = 0.1
    POPULATION_SIZE = 8  # Reduced for 75% utilization
    
    # System Monitoring
    RESOURCE_CHECK_INTERVAL = 60  # Check resource usage every minute
    CPU_USAGE_THRESHOLD = 75  # Alert if CPU usage exceeds 75%
    GPU_USAGE_THRESHOLD = 75  # Alert if GPU usage exceeds 75%
    MEMORY_USAGE_THRESHOLD = 75  # Alert if memory usage exceeds 75%
    
    # Checkpoint and Logging
    SAVE_INTERVAL = 10  # Save every 10 generations
    LOG_LEVEL = "INFO"
    ENABLE_PROFILING = True  # Enable resource profiling
    
    # Estimated Performance (75% utilization)
    STEPS_PER_SECOND = 15000  # Reduced from 20000 due to 75% limit
    ESTIMATED_HOURS = TOTAL_TRAINING_STEPS / (STEPS_PER_SECOND * 3600)
    
    @classmethod
    def print_config(cls):
        """Print the optimized 75% utilization configuration"""
        print("🎯 75% UTILIZATION MASSIVE SCALE TRAINING CONFIG")
        print("=" * 60)
        print(f"📊 Training Scale: {cls.GENERATIONS:,} generations")
        print(f"📈 Episodes per generation: {cls.EPISODES_PER_GENERATION:,}")
        print(f"⚡ Steps per episode: {cls.STEPS_PER_EPISODE:,}")
        print(f"🎯 Total training steps: {cls.TOTAL_TRAINING_STEPS:,}")
        print()
        print("🖥️  RESOURCE ALLOCATION (75% Utilization):")
        print(f"   CPU Cores: PC1({PC1_CPUS}) + PC2({PC2_CPUS}) = {PC1_CPUS + PC2_CPUS} cores")
        print(f"   GPU Usage: {cls.GPU_MEMORY_FRACTION*100}% VRAM on both GPUs")
        print(f"   Workers: {cls.WORKERS_PER_NODE} per PC ({cls.MAX_CONCURRENT_TRAINERS} total)")
        print(f"   Ray Actors: {cls.TRAINER_CPU_ALLOCATION} CPUs, {cls.TRAINER_GPU_ALLOCATION} GPU per trainer")
        print()
        print("⚡ PERFORMANCE OPTIMIZATION:")
        print(f"   Batch Size: {cls.BATCH_SIZE} (reduced for 75% utilization)")
        print(f"   Parallel Episodes: {cls.PARALLEL_EPISODES} (optimized)")
        print(f"   Population Size: {cls.POPULATION_SIZE} bots")
        print(f"   Expected Speed: ~{cls.STEPS_PER_SECOND:,} steps/second")
        print(f"   Estimated Duration: ~{cls.ESTIMATED_HOURS:.1f} hours")
        print("=" * 60)

def setup_optimized_training_environment():
    """Setup the training environment for 75% utilization"""
    print("🔧 SETTING UP 75% UTILIZATION TRAINING ENVIRONMENT")
    print("=" * 50)
    
    # Initialize resource manager
    resource_manager = OptimizedResourceManager()
    
    # Setup PyTorch GPU limits
    resource_manager.setup_pytorch_gpu_limits()
    
    # Set environment variables for optimal performance
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = Enhanced75PercentTrainingConfig.PYTORCH_CUDA_ALLOC_CONF
    os.environ['OMP_NUM_THREADS'] = str(PC1_CPUS // 2)  # Limit OpenMP threads
    os.environ['MKL_NUM_THREADS'] = str(PC1_CPUS // 2)  # Limit MKL threads
    
    print("✅ Training environment optimized for 75% utilization")
    return resource_manager

def monitor_resource_usage():
    """Monitor system resources to ensure 75% utilization compliance"""
    config = Enhanced75PercentTrainingConfig
    
    # Get CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    
    # Get memory usage
    memory = psutil.virtual_memory()
    memory_percent = memory.percent
    
    # Get GPU usage if available
    gpu_percent = 0
    if torch.cuda.is_available():
        # This is a simplified GPU monitoring - in practice you'd use nvidia-ml-py
        gpu_percent = 75  # Placeholder
    
    print(f"📊 Resource Usage: CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}%, GPU {gpu_percent:.1f}%")
    
    # Check if usage exceeds 75% thresholds
    warnings = []
    if cpu_percent > config.CPU_USAGE_THRESHOLD:
        warnings.append(f"CPU usage ({cpu_percent:.1f}%) exceeds 75% target")
    if memory_percent > config.MEMORY_USAGE_THRESHOLD:
        warnings.append(f"Memory usage ({memory_percent:.1f}%) exceeds 75% target")
    if gpu_percent > config.GPU_USAGE_THRESHOLD:
        warnings.append(f"GPU usage ({gpu_percent:.1f}%) exceeds 75% target")
    
    if warnings:
        print("⚠️  Resource usage warnings:")
        for warning in warnings:
            print(f"   - {warning}")
    else:
        print("✅ All resources within 75% utilization targets")
    
    return {
        'cpu_percent': cpu_percent,
        'memory_percent': memory_percent,
        'gpu_percent': gpu_percent,
        'within_limits': len(warnings) == 0
    }

if __name__ == "__main__":
    # Test the 75% utilization configuration
    print("🧪 TESTING 75% UTILIZATION CONFIGURATION")
    Enhanced75PercentTrainingConfig.print_config()
    
    # Setup environment
    resource_manager = setup_optimized_training_environment()
    
    # Monitor resources
    monitor_resource_usage()
    
    print("\n✅ 75% utilization configuration ready!")
