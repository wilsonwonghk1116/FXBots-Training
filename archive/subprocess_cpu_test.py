#!/usr/bin/env python3
"""
FINAL FIX: Replace ineffective threading with multiprocess-based CPU saturation
This bypasses Python's GIL limitation using subprocess workers
"""
import subprocess
import sys
import os
import time
import psutil
import concurrent.futures

def create_cpu_worker_script():
    """Create a separate Python script for CPU-intensive work"""
    worker_script = '''
import time
import sys
import math

def cpu_intensive_computation(duration):
    """Pure CPU computation that saturates a single core"""
    start_time = time.time()
    counter = 0
    
    while time.time() - start_time < duration:
        # CPU-intensive mathematical operations
        for i in range(1000):
            # Complex mathematical computations
            result = math.sqrt(i ** 2 + math.sin(i) * math.cos(i))
            result += math.log(i + 1) * math.exp(i / 1000)
            counter += int(result) % 7
    
    return counter

if __name__ == "__main__":
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
    worker_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    
    print(f"Worker {worker_id} starting for {duration}s")
    result = cpu_intensive_computation(duration)
    print(f"Worker {worker_id} completed: {result}")
    sys.exit(0)
'''
    
    # Write the worker script
    script_path = "/tmp/cpu_worker.py"
    with open(script_path, 'w') as f:
        f.write(worker_script)
    
    return script_path

def subprocess_cpu_saturation(num_processes, duration=10):
    """Use subprocess to achieve true CPU saturation"""
    script_path = create_cpu_worker_script()
    
    print(f"Starting {num_processes} subprocess workers for {duration} seconds...")
    
    # Launch subprocess workers
    processes = []
    for i in range(num_processes):
        cmd = [sys.executable, script_path, str(duration), str(i)]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        processes.append(proc)
    
    # Monitor CPU while workers run
    cpu_readings = []
    start_time = time.time()
    
    while time.time() - start_time < duration + 2:  # Monitor for slightly longer
        cpu_usage = psutil.cpu_percent(interval=1)
        cpu_readings.append(cpu_usage)
        print(f"CPU usage: {cpu_usage:.1f}%")
        
        # Check if any processes have finished
        active_processes = sum(1 for p in processes if p.poll() is None)
        if active_processes == 0:
            break
    
    # Wait for all processes to complete
    for proc in processes:
        proc.wait()
    
    # Cleanup
    try:
        os.remove(script_path)
    except:
        pass
    
    return cpu_readings

def main():
    print("=== SUBPROCESS CPU SATURATION TEST ===")
    print("This approach uses subprocess to bypass Python's GIL completely")
    
    # Test on current machine first
    num_cores = psutil.cpu_count(logical=True)
    target_processes = int(num_cores * 0.75)  # 75% utilization
    
    print(f"System has {num_cores} logical cores")
    print(f"Testing with {target_processes} subprocess workers")
    
    # Run the test
    cpu_readings = subprocess_cpu_saturation(target_processes, 8)
    
    # Analyze results
    avg_cpu = sum(cpu_readings) / len(cpu_readings) if cpu_readings else 0
    max_cpu = max(cpu_readings) if cpu_readings else 0
    
    print(f"\n📊 Results:")
    print(f"   Average CPU: {avg_cpu:.1f}%")
    print(f"   Peak CPU: {max_cpu:.1f}%")
    print(f"   CPU readings: {cpu_readings}")
    
    if max_cpu > 50:
        print("\n🎯 SUCCESS: Subprocess approach achieves high CPU utilization!")
        print("This method should work for the training script.")
    else:
        print("\n⚠️  Still low CPU usage - may need different approach")
    
    print("\n💡 IMPLEMENTATION PLAN:")
    print("1. Replace ThreadPoolExecutor with subprocess workers in training script")
    print("2. Use subprocess.Popen to launch CPU-intensive workers")
    print("3. Monitor and manage worker processes")
    
    print("=== TEST COMPLETE ===")

if __name__ == "__main__":
    main()
