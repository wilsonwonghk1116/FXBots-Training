#!/usr/bin/env python3
"""
QUICK GPU CHECK - Simple GPU monitoring without Ray dependencies
Use this for quick verification that your GPUs are being used
"""

import time
import sys
import os
from datetime import datetime

def check_gpu_basic():
    """Basic GPU check without heavy dependencies"""
    print("🔍 QUICK GPU VERIFICATION")
    print("=" * 50)
    
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        
        if not gpus:
            print("❌ No GPUs detected!")
            return False
        
        print(f"✅ Found {len(gpus)} GPU(s):")
        
        any_active = False
        for i, gpu in enumerate(gpus):
            load = gpu.load * 100
            vram = (gpu.memoryUsed / gpu.memoryTotal) * 100
            temp = gpu.temperature
            
            status = "🔥 ACTIVE" if load > 30 else "💤 IDLE"
            temp_status = "🌡️ HOT" if temp > 75 else "❄️ COOL"
            
            print(f"\n🎮 GPU {i}: {gpu.name}")
            print(f"   📈 Load: {load:.1f}% {status}")
            print(f"   💾 VRAM: {vram:.1f}% ({gpu.memoryUsed:.0f}/{gpu.memoryTotal:.0f} MB)")
            print(f"   🌡️ Temp: {temp}°C {temp_status}")
            
            if load > 30:
                any_active = True
        
        print("\n" + "=" * 50)
        if any_active:
            print("✅ SUCCESS: At least one GPU is active!")
        else:
            print("⚠️ All GPUs appear idle")
            print("💡 Start your training to see GPU activity")
        
        return any_active
        
    except ImportError:
        print("❌ GPUtil not installed. Install with: pip install GPUtil")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def check_pytorch_gpu():
    """Check PyTorch GPU availability"""
    try:
        import torch
        print("\n🔥 PYTORCH GPU CHECK:")
        print(f"   CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA Devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   Device {i}: {torch.cuda.get_device_name(i)}")
    except ImportError:
        print("⚠️ PyTorch not available")

def monitor_continuous():
    """Continuous monitoring mode"""
    print("\n🔄 Starting continuous monitoring (Press Ctrl+C to stop)...")
    
    try:
        while True:
            os.system('clear' if os.name == 'posix' else 'cls')
            print(f"🕐 {datetime.now().strftime('%H:%M:%S')} - GPU Monitor")
            check_gpu_basic()
            time.sleep(2)
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped")

def main():
    print("🚀 QUICK GPU CHECK")
    print("Choose an option:")
    print("1. Quick check (one-time)")
    print("2. Continuous monitoring")
    print("3. Both")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice in ['1', '3']:
        check_gpu_basic()
        check_pytorch_gpu()
    
    if choice in ['2', '3']:
        monitor_continuous()

if __name__ == "__main__":
    main() 