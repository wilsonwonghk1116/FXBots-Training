#!/usr/bin/env python3
"""
Manual Cluster Connectivity Test Script
======================================
This script performs all cluster connectivity tests step by step
"""

import subprocess
import time
import sys
import os
from datetime import datetime

# Configuration
PC1_IP = "192.168.1.10"
PC2_IP = "192.168.1.11"
PC2_USER = "w1"
PC2_PASSWORD = "w"
CONDA_ENV = "Training_env"

def log(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

def run_command(cmd, description):
    log(f"TEST: {description}")
    log(f"CMD: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            log(f"✅ SUCCESS: {description}")
            if result.stdout:
                print(result.stdout)
        else:
            log(f"❌ FAILED: {description}")
            if result.stderr:
                print(f"ERROR: {result.stderr}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        log(f"⏱️ TIMEOUT: {description}")
        return False
    except Exception as e:
        log(f"❌ EXCEPTION: {e}")
        return False

def main():
    print("=" * 60)
    print("🧪 MANUAL CLUSTER CONNECTIVITY TESTS")
    print("=" * 60)
    
    # Test 1: Basic Network Connectivity
    log("Test 1: Basic network connectivity to PC2")
    run_command(f"ping -c 2 {PC2_IP}", "Ping PC2")
    
    # Test 2: SSH Connectivity
    log("Test 2: SSH connectivity to PC2")
    run_command(f"sshpass -p '{PC2_PASSWORD}' ssh -o ConnectTimeout=10 {PC2_USER}@{PC2_IP} 'echo SSH_SUCCESS'", "SSH to PC2")
    
    # Test 3: Conda Environment Check on PC1
    log("Test 3: Check conda environment on PC1")
    run_command(f"conda env list | grep {CONDA_ENV}", f"Check {CONDA_ENV} on PC1")
    
    # Test 4: Conda Environment Check on PC2
    log("Test 4: Check conda environment on PC2")
    run_command(f"sshpass -p '{PC2_PASSWORD}' ssh {PC2_USER}@{PC2_IP} 'conda env list | grep {CONDA_ENV}'", f"Check {CONDA_ENV} on PC2")
    
    # Test 5: Python Version Check
    log("Test 5: Python version compatibility")
    run_command(f"source ~/miniconda3/etc/profile.d/conda.sh && conda activate {CONDA_ENV} && python --version", "PC1 Python version")
    run_command(f"sshpass -p '{PC2_PASSWORD}' ssh {PC2_USER}@{PC2_IP} 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate {CONDA_ENV} && python --version'", "PC2 Python version")
    
    # Test 6: Ray Installation Check
    log("Test 6: Ray installation check")
    run_command(f"source ~/miniconda3/etc/profile.d/conda.sh && conda activate {CONDA_ENV} && python -c 'import ray; print(ray.__version__)'", "PC1 Ray version")
    run_command(f"sshpass -p '{PC2_PASSWORD}' ssh {PC2_USER}@{PC2_IP} 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate {CONDA_ENV} && python -c \"import ray; print(ray.__version__)\"'", "PC2 Ray version")
    
    # Test 7: Current Ray Status
    log("Test 7: Current Ray cluster status")
    run_command(f"source ~/miniconda3/etc/profile.d/conda.sh && conda activate {CONDA_ENV} && ray status", "Current Ray status")
    
    # Test 8: System Resources
    log("Test 8: System resources check")
    run_command("nproc", "PC1 CPU count")
    run_command("nvidia-smi --query-gpu=count --format=csv,noheader,nounits", "PC1 GPU count")
    run_command(f"sshpass -p '{PC2_PASSWORD}' ssh {PC2_USER}@{PC2_IP} 'nproc'", "PC2 CPU count")
    run_command(f"sshpass -p '{PC2_PASSWORD}' ssh {PC2_USER}@{PC2_IP} 'nvidia-smi --query-gpu=count --format=csv,noheader,nounits'", "PC2 GPU count")
    
    print("\n" + "=" * 60)
    print("🏁 MANUAL TESTS COMPLETED")
    print("=" * 60)
    
    # Final recommendation
    log("If all tests passed, you can run:")
    log("python automated_cluster_training.py")

if __name__ == "__main__":
    main()
