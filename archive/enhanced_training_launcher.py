#!/usr/bin/env python3
"""
Enhanced Ray Kelly Monte Carlo Training with Maximum GPU/CPU Utilization
Focus: Push both PCs to 75% CPU, 75% GPU, 75% VRAM utilization
"""

import argparse
import subprocess
import sys
import time
import json
import os
from pathlib import Path
from datetime import datetime
import signal
import threading

def parse_args():
    parser = argparse.ArgumentParser(description='Kelly Monte Carlo Ray Training with Maximum Resource Utilization')
    parser.add_argument('--n-bots', type=int, default=2000, help='Number of bots in fleet')
    parser.add_argument('--duration-hours', type=int, default=24, help='Training duration in hours')
    parser.add_argument('--session-dir', type=str, default=None, help='Session directory')
    parser.add_argument('--auto-save-interval', type=int, default=3600, help='Auto-save interval in seconds')
    parser.add_argument('--gpu-utilization-target', type=float, default=0.75, help='GPU utilization target')
    parser.add_argument('--cpu-utilization-target', type=float, default=0.75, help='CPU utilization target')
    parser.add_argument('--head-node-ip', type=str, default='192.168.1.100', help='Head node IP address')
    parser.add_argument('--ray-port', type=int, default=8265, help='Ray dashboard port')
    return parser.parse_args()

def start_gui_monitoring():
    """Start GUI monitoring components in background"""
    print("�️  Starting GUI monitoring...")
    
    try:
        # Start resource monitor
        subprocess.Popen([
            'python3', 'ray_cluster_monitor_75_percent.py'
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("   ✅ Resource monitor started")
        
        # Generate fresh demo data for bot dashboard
        subprocess.run(['python3', 'generate_demo_fleet_data.py'], 
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Start bot dashboard
        subprocess.Popen([
            'python3', 'kelly_bot_dashboard.py'
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("   ✅ Bot dashboard started")
        
        time.sleep(2)  # Give GUIs time to initialize
        
    except Exception as e:
        print(f"   ⚠️  GUI monitoring failed to start: {e}")

def check_system_resources():
    """Check and report current system capabilities"""
    print("🔍 Checking system resources...")
    
    try:
        # Check CPU cores
        import multiprocessing
        cpu_cores = multiprocessing.cpu_count()
        print(f"   • CPU Cores: {cpu_cores}")
        
        # Check GPU availability
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                print(f"   • CUDA GPUs: {gpu_count}")
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                    print(f"     - GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
            else:
                print("   • CUDA GPUs: Not available")
        except ImportError:
            print("   • PyTorch not available for GPU check")
            
    except Exception as e:
        print(f"   ⚠️  Resource check failed: {e}")

def main():
    args = parse_args()
    
    print("🚀 Enhanced Kelly Monte Carlo Training - Maximum Resource Utilization")
    print("=" * 80)
    print(f"🎯 Target: {args.cpu_utilization_target*100}% CPU, {args.gpu_utilization_target*100}% GPU, {args.gpu_utilization_target*100}% VRAM")
    print(f"🤖 Fleet Size: {args.n_bots} bots")
    print(f"⏱️  Duration: {args.duration_hours} hours")
    print()
    
    # Check system resources
    check_system_resources()
    print()
    
    # Start GUI monitoring
    start_gui_monitoring()
    print()
    
    print("🎯 Starting maximum resource utilization training...")
    print("   Focus: Push GPUs and CPUs to 75% utilization")
    print("   Optimized for overclocked VRAM on both GPUs")
    print()
    
    try:
        # Import and run the Ray training directly
        print("📦 Loading Ray Kelly Monte Carlo training module...")
        
        # Run the main Ray training script directly
        import ray_kelly_ultimate_75_percent
        
        print("🚀 Launching distributed training with maximum resource saturation...")
        
        # Call the main function from the ray training module
        ray_kelly_ultimate_75_percent.main()
        
        print("✅ Training completed successfully!")
        
    except ImportError as e:
        print(f"❌ Error importing Ray training module: {e}")
        print("   Make sure ray_kelly_ultimate_75_percent.py is available")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("   Check Ray cluster connection and GPU availability")
        sys.exit(1)

if __name__ == "__main__":
    main()
