#!/usr/bin/env python3
"""
PRODUCTION FIX: Modified training script with subprocess CPU saturation
Replace the ineffective threading approach with subprocess workers
"""

# Find and replace the CPU saturation section in the training script
import subprocess
import sys
import os
import tempfile

def create_cpu_saturation_fix():
    """Create the fixed CPU saturation code to replace threading approach"""
    
    fixed_code = '''
            def cpu_intensive_work_subprocess(worker_id, duration=0.075):
                """Single CPU-intensive computation for subprocess worker"""
                import time
                import math
                import sys
                
                start_time = time.time()
                counter = 0
                
                while time.time() - start_time < duration:
                    # CPU-intensive mathematical operations
                    for i in range(500):  # Reduced iterations for shorter bursts
                        result = math.sqrt(i ** 2 + math.sin(i) * math.cos(i))
                        result += math.log(i + 1) * math.exp(i / 500)
                        counter += int(result) % 7
                
                return counter

            def create_cpu_worker_script():
                """Create temporary CPU worker script"""
                worker_script = \'''
import time
import math
import sys

def cpu_work(duration):
    start_time = time.time()
    counter = 0
    
    while time.time() - start_time < duration:
        for i in range(500):
            result = math.sqrt(i ** 2 + math.sin(i) * math.cos(i))
            result += math.log(i + 1) * math.exp(i / 500)
            counter += int(result) % 7
    
    return counter

if __name__ == "__main__":
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 0.075
    result = cpu_work(duration)
    sys.exit(0)
\'''
                
                # Create temporary script file
                import tempfile
                fd, script_path = tempfile.mkstemp(suffix='.py', text=True)
                try:
                    with os.fdopen(fd, 'w') as f:
                        f.write(worker_script)
                    return script_path
                except:
                    os.close(fd)
                    raise

            def sustained_cpu_load():
                """Run sustained CPU load using subprocess workers (GIL-free)"""
                import subprocess
                import psutil
                import time
                
                num_cores = psutil.cpu_count(logical=True)
                target_workers = int(num_cores * 0.75)  # 75% of available cores
                print(f"🔥 CPU: Starting {target_workers} subprocess workers on {node_hostname} ({num_cores} cores)")
                
                # Create worker script
                script_path = create_cpu_worker_script()
                
                try:
                    while True:
                        # Launch subprocess workers for CPU bursts
                        processes = []
                        for i in range(target_workers):
                            cmd = [sys.executable, script_path, "0.075"]  # 75ms burst
                            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                            processes.append(proc)
                        
                        # Wait for all workers to complete
                        for proc in processes:
                            proc.wait()
                        
                        # Brief pause to maintain 75% (not 100%) utilization  
                        time.sleep(0.025)  # 25ms pause
                        
                finally:
                    # Cleanup script file
                    try:
                        os.remove(script_path)
                    except:
                        pass
'''
    
    return fixed_code

def apply_fix_to_training_script():
    """Apply the subprocess CPU fix to the training script"""
    
    print("🔧 APPLYING CPU SATURATION FIX...")
    
    # Read the current training script
    script_path = "/home/w1/cursor-to-copilot-backup/TaskmasterForexBots/fixed_integrated_training_75_percent.py"
    
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Find the section to replace (the threading-based CPU saturation)
    start_marker = "def sustained_cpu_load():"
    end_marker = "# GPU Saturation Functions"
    
    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker)
    
    if start_idx == -1 or end_idx == -1:
        print("❌ Could not find CPU saturation section to replace")
        return False
    
    # Replace the section
    fixed_code = create_cpu_saturation_fix()
    
    new_content = (
        content[:start_idx] + 
        fixed_code + 
        "\n            " + 
        content[end_idx:]
    )
    
    # Create backup
    backup_path = script_path + ".backup_threading"
    with open(backup_path, 'w') as f:
        f.write(content)
    print(f"✅ Created backup: {backup_path}")
    
    # Write fixed version
    with open(script_path, 'w') as f:
        f.write(new_content)
    
    print("✅ Applied subprocess CPU saturation fix!")
    print("🎯 The training script now uses subprocess workers instead of threading")
    
    return True

def main():
    print("=== APPLYING FINAL CPU SATURATION FIX ===")
    
    success = apply_fix_to_training_script()
    
    if success:
        print("\n🎉 FIX APPLIED SUCCESSFULLY!")
        print("\n📋 WHAT WAS CHANGED:")
        print("   ✅ Replaced ThreadPoolExecutor with subprocess workers")
        print("   ✅ CPU workers now bypass Python's GIL completely") 
        print("   ✅ Should achieve 75% CPU utilization on both PC1 and PC2")
        
        print("\n🚀 NEXT STEPS:")
        print("1. Test the updated training script with Ray cluster")
        print("2. Monitor CPU usage on both PC1 and PC2")
        print("3. Verify PC2's 16-core i9 now shows proper utilization")
        
        print("\n💡 TECHNICAL DETAILS:")
        print("   - Threading limited by Python GIL (Global Interpreter Lock)")
        print("   - Subprocess workers run as separate processes (no GIL)")
        print("   - Each worker saturates 1 CPU core independently")
        print("   - 75% target = 12 workers on PC2's 16 cores")
        
    else:
        print("\n❌ Fix application failed")
        print("Manual modification may be required")
    
    print("\n=== FIX APPLICATION COMPLETE ===")

if __name__ == "__main__":
    main()
