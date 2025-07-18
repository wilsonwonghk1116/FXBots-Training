#!/usr/bin/env python3
"""
Quick Cluster Test Script
========================
Simple test to verify cluster connectivity with corrected PC2 settings
"""

import subprocess
import time

def run_cmd(cmd, desc):
    print(f"\n🔧 {desc}")
    print(f"Command: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(f"✅ Success: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ Failed: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("🚀 QUICK CLUSTER CONNECTIVITY TEST")
    print("=" * 50)
    
    # Configuration
    PC1_IP = "192.168.1.10"
    PC2_IP = "192.168.1.11"  # Corrected IP
    PC2_USER = "w1"
    PC2_PASSWORD = "w"        # Corrected password
    CONDA_ENV = "Training_env"
    
    print(f"PC1 IP: {PC1_IP}")
    print(f"PC2 IP: {PC2_IP}")
    print(f"PC2 User: {PC2_USER}")
    print(f"PC2 Password: {'*' * len(PC2_PASSWORD)}")
    print(f"Conda Environment: {CONDA_ENV}")
    
    # Test 1: Check conda environment on PC1
    run_cmd(f"conda env list | grep {CONDA_ENV}", "Check conda environment on PC1")
    
    # Test 2: Check SSH connectivity to PC2
    run_cmd(f"sshpass -p '{PC2_PASSWORD}' ssh -o ConnectTimeout=10 {PC2_USER}@{PC2_IP} 'echo Connected to PC2'", 
            "Test SSH connection to PC2")
    
    # Test 3: Check conda environment on PC2
    run_cmd(f"sshpass -p '{PC2_PASSWORD}' ssh {PC2_USER}@{PC2_IP} 'conda env list | grep {CONDA_ENV}'", 
            "Check conda environment on PC2")
    
    # Test 4: Check Python versions
    run_cmd(f"python --version", "Check PC1 Python version")
    
    run_cmd(f"sshpass -p '{PC2_PASSWORD}' ssh {PC2_USER}@{PC2_IP} 'python --version'", 
            "Check PC2 Python version")
    
    # Test 5: Check Ray installation
    run_cmd(f"python -c 'import ray; print(f\"Ray version: {{ray.__version__}}\")'", 
            "Check Ray installation on PC1")
    
    run_cmd(f"sshpass -p '{PC2_PASSWORD}' ssh {PC2_USER}@{PC2_IP} 'python -c \"import ray; print(f\\\"Ray version: {{ray.__version__}}\\\")\"'", 
            "Check Ray installation on PC2")
    
    print("\n" + "=" * 50)
    print("🎯 QUICK TEST COMPLETED")
    print("If all tests passed, run: python automated_cluster_training.py")
    print("=" * 50)

if __name__ == "__main__":
    main()
