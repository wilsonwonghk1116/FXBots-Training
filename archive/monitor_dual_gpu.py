#!/usr/bin/env python3
"""
Monitor GPU usage on both PC1 and PC2 during distributed training
"""

import time
import subprocess
import threading

def monitor_pc1_gpu():
    """Monitor PC1 GPU usage"""
    print("Monitoring PC1 GPU...")
    while True:
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                gpu_info = result.stdout.strip()
                timestamp = time.strftime("%H:%M:%S")
                print(f"[{timestamp}] PC1: {gpu_info}")
            time.sleep(5)
        except Exception as e:
            print(f"PC1 monitoring error: {e}")
            time.sleep(10)

def monitor_pc2_gpu():
    """Monitor PC2 GPU usage via SSH"""
    print("Monitoring PC2 GPU...")
    while True:
        try:
            cmd = ['sshpass', '-p', 'w', 'ssh', 'w2@192.168.1.11', 
                   'nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if result.returncode == 0:
                gpu_info = result.stdout.strip()
                timestamp = time.strftime("%H:%M:%S")
                print(f"[{timestamp}] PC2: {gpu_info}")
            time.sleep(5)
        except Exception as e:
            print(f"PC2 monitoring error: {e}")
            time.sleep(10)

def main():
    print("=== Dual PC GPU Monitoring ===")
    print("Monitoring GPU usage on both PC1 and PC2...")
    print("PC1: Primary GPU")
    print("PC2: RTX 3070 8GB")
    print("Press Ctrl+C to stop monitoring\n")
    
    # Start monitoring threads
    thread1 = threading.Thread(target=monitor_pc1_gpu, daemon=True)
    thread2 = threading.Thread(target=monitor_pc2_gpu, daemon=True)
    
    thread1.start()
    thread2.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

if __name__ == "__main__":
    main()
