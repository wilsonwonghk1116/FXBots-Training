#!/usr/bin/env python3
"""
FINAL VERIFICATION: Test the fixed CPU saturation on Ray cluster
"""
import ray
import psutil
import subprocess
import tempfile
import os
import sys
import time

def create_subprocess_worker_script():
    """Create the CPU worker script for subprocess execution"""
    worker_script = '''
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
'''
    
    # Create temporary script file
    fd, script_path = tempfile.mkstemp(suffix='.py', text=True)
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(worker_script)
        return script_path
    except:
        os.close(fd)
        raise

@ray.remote(num_cpus=12, resources={"node:192.168.1.11": 1})
def pc2_fixed_cpu_worker():
    """PC2 Ray task using FIXED subprocess CPU saturation"""
    import subprocess
    import psutil
    import time
    import os
    import sys
    
    print(f"[PC2] FIXED CPU worker running on: {ray._private.services.get_node_ip_address()}")
    print(f"[PC2] Process ID: {os.getpid()}")
    
    # Create worker script
    script_path = create_subprocess_worker_script()
    
    # Monitor CPU usage
    cpu_readings = []
    
    def monitor_cpu():
        for i in range(12):  # 12 seconds
            cpu_usage = psutil.cpu_percent(interval=1)
            cpu_readings.append(cpu_usage)
            print(f"[PC2] CPU usage: {cpu_usage:.1f}%")
    
    # Start monitoring in background
    import threading
    monitor_thread = threading.Thread(target=monitor_cpu)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    try:
        print("[PC2] Starting subprocess CPU saturation...")
        num_cores = psutil.cpu_count(logical=True)
        target_workers = int(num_cores * 0.75)  # 75% of cores
        
        # Run subprocess workers for 10 seconds
        start_time = time.time()
        while time.time() - start_time < 10:
            # Launch subprocess workers
            processes = []
            for i in range(target_workers):
                cmd = [sys.executable, script_path, "0.075"]  # 75ms burst
                proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                processes.append(proc)
            
            # Wait for all workers to complete
            for proc in processes:
                proc.wait()
            
            # Brief pause (25ms)
            time.sleep(0.025)
            
    finally:
        # Cleanup
        try:
            os.remove(script_path)
        except:
            pass
    
    # Wait for monitoring to complete
    monitor_thread.join()
    
    avg_cpu = sum(cpu_readings) / len(cpu_readings) if cpu_readings else 0
    max_cpu = max(cpu_readings) if cpu_readings else 0
    
    print(f"[PC2] COMPLETED - Avg CPU: {avg_cpu:.1f}%, Max CPU: {max_cpu:.1f}%")
    
    return {
        'avg_cpu': avg_cpu,
        'max_cpu': max_cpu,
        'cpu_readings': cpu_readings
    }

@ray.remote(num_cpus=12, resources={"node:192.168.1.10": 1})
def pc1_fixed_cpu_worker():
    """PC1 Ray task using FIXED subprocess CPU saturation"""
    import subprocess
    import psutil
    import time
    import os
    import sys
    
    print(f"[PC1] FIXED CPU worker running on: {ray._private.services.get_node_ip_address()}")
    print(f"[PC1] Process ID: {os.getpid()}")
    
    # Create worker script  
    script_path = create_subprocess_worker_script()
    
    # Monitor CPU usage
    cpu_readings = []
    
    def monitor_cpu():
        for i in range(12):  # 12 seconds
            cpu_usage = psutil.cpu_percent(interval=1)
            cpu_readings.append(cpu_usage)
            print(f"[PC1] CPU usage: {cpu_usage:.1f}%")
    
    # Start monitoring in background
    import threading
    monitor_thread = threading.Thread(target=monitor_cpu)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    try:
        print("[PC1] Starting subprocess CPU saturation...")
        num_cores = psutil.cpu_count(logical=True)
        target_workers = int(num_cores * 0.75)  # 75% of cores
        
        # Run subprocess workers for 10 seconds
        start_time = time.time()
        while time.time() - start_time < 10:
            # Launch subprocess workers
            processes = []
            for i in range(target_workers):
                cmd = [sys.executable, script_path, "0.075"]  # 75ms burst
                proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                processes.append(proc)
            
            # Wait for all workers to complete
            for proc in processes:
                proc.wait()
            
            # Brief pause (25ms)
            time.sleep(0.025)
            
    finally:
        # Cleanup
        try:
            os.remove(script_path)
        except:
            pass
    
    # Wait for monitoring to complete
    monitor_thread.join()
    
    avg_cpu = sum(cpu_readings) / len(cpu_readings) if cpu_readings else 0
    max_cpu = max(cpu_readings) if cpu_readings else 0
    
    print(f"[PC1] COMPLETED - Avg CPU: {avg_cpu:.1f}%, Max CPU: {max_cpu:.1f}%")
    
    return {
        'avg_cpu': avg_cpu,
        'max_cpu': max_cpu,
        'cpu_readings': cpu_readings
    }

def main():
    print("=== FINAL VERIFICATION: FIXED CPU SATURATION ===")
    
    # Initialize Ray
    if not ray.is_initialized():
        try:
            ray.init(address='ray://192.168.1.10:10001')
            print("✅ Ray connected successfully")
        except Exception as e:
            print(f"❌ Failed to connect to Ray: {e}")
            return
    
    print("\n🧪 Testing FIXED subprocess CPU saturation on both nodes...")
    print("This should show high CPU usage on BOTH PC1 and PC2!")
    
    # Launch both tasks
    start_time = time.time()
    
    pc2_future = pc2_fixed_cpu_worker.remote()
    pc1_future = pc1_fixed_cpu_worker.remote()
    
    # Get results
    print("\n⏳ Waiting for both tasks to complete...")
    pc2_result = ray.get(pc2_future)
    pc1_result = ray.get(pc1_future)
    
    total_time = time.time() - start_time
    
    print(f"\n📊 FINAL RESULTS:")
    print(f"   Total time: {total_time:.1f}s")
    print(f"   PC1 (80 cores): Avg {pc1_result['avg_cpu']:.1f}%, Max {pc1_result['max_cpu']:.1f}%")
    print(f"   PC2 (16 cores): Avg {pc2_result['avg_cpu']:.1f}%, Max {pc2_result['max_cpu']:.1f}%")
    
    # Success criteria
    pc1_success = pc1_result['max_cpu'] > 40  # Should be high on PC1
    pc2_success = pc2_result['max_cpu'] > 40  # This is the key test for PC2
    
    print(f"\n🎯 VERIFICATION RESULTS:")
    print(f"   PC1 CPU Saturation: {'✅ SUCCESS' if pc1_success else '❌ FAILED'}")
    print(f"   PC2 CPU Saturation: {'✅ SUCCESS' if pc2_success else '❌ FAILED'}")
    
    if pc1_success and pc2_success:
        print(f"\n🎉 PROBLEM SOLVED!")
        print(f"   Both PC1 and PC2 now achieve proper CPU utilization")
        print(f"   PC2's 16-core i9 CPU is finally being used effectively")
        print(f"   The subprocess fix successfully bypassed Python's GIL")
    elif pc2_success:
        print(f"\n🎯 PC2 FIXED!")
        print(f"   PC2's CPU utilization issue has been resolved")
    else:
        print(f"\n⚠️  Further investigation needed")
        print(f"   The subprocess approach may need additional tuning")
    
    print("=== VERIFICATION COMPLETE ===")

if __name__ == "__main__":
    main()
