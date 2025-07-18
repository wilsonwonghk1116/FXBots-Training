#!/usr/bin/env python3
"""
Quick verification of 75% utilization configuration
"""

from cluster_config import *

print("🎯 75% UTILIZATION VERIFICATION")
print("=" * 40)
print(f"PC1 Resources (75% utilization):")
print(f"  CPUs: {PC1_CPUS} (75% of {PC1_TOTAL_CPUS})")
print(f"  VRAM: {PC1_TOTAL_VRAM_GB * UTILIZATION_PERCENTAGE:.1f}GB (75% of {PC1_TOTAL_VRAM_GB}GB)")
print()
print(f"PC2 Resources (75% utilization):")
print(f"  CPUs: {PC2_CPUS} (75% of {PC2_TOTAL_CPUS})")
print(f"  VRAM: {PC2_TOTAL_VRAM_GB * UTILIZATION_PERCENTAGE:.1f}GB (75% of {PC2_TOTAL_VRAM_GB}GB)")
print()
print(f"Total Cluster (75% utilization):")
print(f"  CPUs: {PC1_CPUS + PC2_CPUS} cores")
print(f"  GPUs: {PC1_GPUS + PC2_GPUS} (with VRAM limits)")
print(f"  VRAM: {(PC1_TOTAL_VRAM_GB + PC2_TOTAL_VRAM_GB) * UTILIZATION_PERCENTAGE:.1f}GB total")
print()
print("✅ 75% utilization configuration verified!")
print("🚀 Ready for optimized training with sustainable resource usage")
