#!/usr/bin/env python3
"""
Monitor the training progress and system resources
"""

import psutil
import time
import os
import GPUtil

def monitor_training():
    print("Training Progress Monitor")
    print("=" * 40)
    
    # Find the training process
    training_process = None
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'python' in proc.info['name'] and 'run_smart_real_training.py' in ' '.join(proc.info['cmdline']):
                training_process = proc
                break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    if not training_process:
        print("❌ Training process not found")
        return
    
    print(f"✅ Found training process: PID {training_process.pid}")
    print()
    
    for i in range(10):  # Monitor for 10 iterations
        try:
            # Process info
            cpu_percent = training_process.cpu_percent()
            memory_info = training_process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            # System GPU info
            gpu_info = ""
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    gpu_info = f"GPU: {gpu.load*100:.1f}% | VRAM: {gpu.memoryUsed}/{gpu.memoryTotal}MB ({gpu.memoryUtil*100:.1f}%)"
            except:
                gpu_info = "GPU: N/A"
            
            # System CPU and RAM
            system_cpu = psutil.cpu_percent()
            system_ram = psutil.virtual_memory()
            
            print(f"Iteration {i+1}/10:")
            print(f"  Process CPU: {cpu_percent:.1f}% | RAM: {memory_mb:.1f}MB")
            print(f"  System CPU: {system_cpu:.1f}% | RAM: {system_ram.percent:.1f}%")
            print(f"  {gpu_info}")
            
            # Check for output files
            files_created = []
            for filename in os.listdir('.'):
                if filename.startswith('CHAMPION_') and filename.endswith('.pth'):
                    files_created.append(filename)
            
            if files_created:
                print(f"  Files created: {len(files_created)} champion models")
                print(f"  Latest: {max(files_created, key=os.path.getctime)}")
            else:
                print("  No champion models created yet")
            
            print()
            time.sleep(30)  # Wait 30 seconds
            
        except psutil.NoSuchProcess:
            print("❌ Training process ended")
            break
        except KeyboardInterrupt:
            print("\n⏹️  Monitoring stopped by user")
            break
    
    print("Monitoring complete")

if __name__ == "__main__":
    monitor_training()
