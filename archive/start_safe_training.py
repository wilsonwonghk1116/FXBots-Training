#!/usr/bin/env python3
"""
SAFE TRAINING STARTUP SCRIPT
============================
Launches the safe dual PC training system with freeze prevention
"""

import os
import sys
import time
import subprocess
import psutil
import signal

def cleanup_processes():
    """Clean up any hanging processes"""
    print("🧹 Cleaning up processes...")
    
    # Kill any existing Ray processes
    try:
        subprocess.run(["ray", "stop", "--force"], capture_output=True)
        print("✅ Ray processes cleaned")
    except:
        pass
    
    # Kill any Python training processes
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            if proc.info['name'] == 'python' or proc.info['name'] == 'python3':
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if 'trainer' in cmdline.lower() or 'forex' in cmdline.lower():
                    proc.terminate()
                    print(f"✅ Terminated training process {proc.info['pid']}")
    except:
        pass
    
    print("✅ Process cleanup complete")

def check_system_health():
    """Check system health before starting"""
    print("🔍 Checking system health...")
    
    # Check CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    if cpu_percent > 50:
        print(f"⚠️ High CPU usage detected: {cpu_percent:.1f}%")
        return False
    
    # Check memory usage
    memory = psutil.virtual_memory()
    if memory.percent > 80:
        print(f"⚠️ High memory usage detected: {memory.percent:.1f}%")
        return False
    
    # Check GPU temperature if possible
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            if gpu.temperature > 75:
                print(f"⚠️ High GPU temperature: {gpu.temperature}°C")
                return False
    except:
        pass
    
    print("✅ System health OK")
    return True

def start_safe_training():
    """Start the safe training system"""
    print("🚀 === SAFE DUAL PC FOREX TRAINER STARTUP ===")
    print("Freeze-resistant training with conservative resource usage")
    print("=" * 55)
    
    # Step 1: Cleanup first
    cleanup_processes()
    time.sleep(2)
    
    # Step 2: Health check
    if not check_system_health():
        print("❌ System health check failed - please wait for resources to free up")
        return False
    
    # Step 3: Start training
    print("🎯 Starting safe training system...")
    
    try:
        # Change to project directory
        project_dir = "/home/w1/cursor-to-copilot-backup/TaskmasterForexBots"
        if os.path.exists(project_dir):
            os.chdir(project_dir)
            print(f"📁 Changed to project directory: {project_dir}")
        
        # Start the safe trainer
        cmd = [sys.executable, "safe_dual_pc_trainer.py"]
        
        print("🚀 Launching safe trainer...")
        print("💡 Use Ctrl+C to stop safely")
        print("📊 Monitor resource usage with system monitor")
        print("=" * 55)
        
        # Run with proper signal handling
        def signal_handler(sig, frame):
            print("\n🛑 Shutdown signal received - cleaning up...")
            cleanup_processes()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start the training process
        result = subprocess.run(cmd, check=True)
        
        print("✅ Training completed successfully")
        return True
        
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
        cleanup_processes()
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Training process failed with code {e.returncode}")
        cleanup_processes()
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        cleanup_processes()
        return False

def show_usage():
    """Show usage instructions"""
    print("""
🛡️ SAFE TRAINING STARTUP SCRIPT
==============================

BEFORE RUNNING:
1. Ensure both PCs are on and connected
2. Check that GPU temperatures are below 70°C  
3. Close any other intensive applications

TO START TRAINING:
    python start_safe_training.py

FEATURES:
✅ Automatic process cleanup
✅ System health checks
✅ Safe resource limits (60% utilization)
✅ Proper shutdown handling
✅ Freeze prevention

HARDWARE CONFIGURATION:
📡 Head PC1: 192.168.1.10 (2x Xeon + RTX 3090)
📡 Worker PC2: 192.168.1.11 (I9 + RTX 3070)

SAFETY FEATURES:
🌡️ Temperature monitoring
💾 Memory leak prevention  
🔄 Automatic cleanup on exit
⚡ Conservative resource usage
""")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        show_usage()
    else:
        success = start_safe_training()
        if success:
            print("🎉 Startup script completed successfully!")
        else:
            print("❌ Startup script encountered issues")
            sys.exit(1) 