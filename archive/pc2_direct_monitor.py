#!/usr/bin/env python3
"""
Direct PC2 CPU monitoring script
This script should be run on PC2 to monitor its own CPU usage
"""
import ray
import psutil
import concurrent.futures
import time
import os
import numpy as np

def cpu_work_intensive(thread_id, duration=10):
    """CPU-intensive work"""
    start_time = time.time()
    iterations = 0
    
    while time.time() - start_time < duration:
        # Pure Python CPU work (should be affected by GIL)
        for i in range(10000):
            x = sum(j**2 for j in range(100))
        iterations += 1
    
    return iterations

@ray.remote(num_cpus=12, resources={"node:192.168.1.11": 1})
def pc2_cpu_saturation_task():
    """PC2 task that monitors its own CPU usage"""
    print(f"Task running on PC2 node: {ray._private.services.get_node_ip_address()}")
    print(f"Process ID: {os.getpid()}")
    
    # Start background CPU monitoring
    cpu_readings = []
    
    def monitor_cpu():
        for _ in range(15):  # 15 seconds
            cpu_usage = psutil.cpu_percent(interval=1)
            cpu_readings.append(cpu_usage)
            print(f"[PC2] CPU usage: {cpu_usage:.1f}%")
    
    # Start monitoring in background
    import threading
    monitor_thread = threading.Thread(target=monitor_cpu)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # Start CPU intensive work with threading
    print("Starting 12 threads for CPU work...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        futures = []
        
        for i in range(12):
            future = executor.submit(cpu_work_intensive, i, 10)  # 10 seconds each
            futures.append(future)
        
        # Wait for completion
        results = []
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            result = future.result()
            results.append(result)
            print(f"[PC2] Thread {i} completed {result} iterations")
    
    # Wait for monitoring to complete
    monitor_thread.join()
    
    avg_cpu = sum(cpu_readings) / len(cpu_readings) if cpu_readings else 0
    max_cpu = max(cpu_readings) if cpu_readings else 0
    
    return {
        'total_iterations': sum(results),
        'avg_cpu': avg_cpu,
        'max_cpu': max_cpu,
        'cpu_readings': cpu_readings
    }

def main():
    print("=== PC2 DIRECT CPU MONITORING TEST ===")
    print("This script will monitor PC2's CPU usage from PC2 itself")
    
    # Initialize Ray
    if not ray.is_initialized():
        try:
            ray.init(address='ray://192.168.1.10:10001')
            print("✅ Ray connected successfully")
        except Exception as e:
            print(f"❌ Failed to connect to Ray: {e}")
            return
    
    print("\n🧪 Starting PC2 CPU saturation test...")
    print("The task will run on PC2 and monitor its own CPU usage")
    
    # Launch the task
    start_time = time.time()
    pc2_future = pc2_cpu_saturation_task.remote()
    
    # Get results
    print("\n⏳ Waiting for PC2 task to complete...")
    result = ray.get(pc2_future)
    
    total_time = time.time() - start_time
    
    print(f"\n✅ PC2 Test Results:")
    print(f"   Total iterations: {result['total_iterations']}")
    print(f"   Average CPU: {result['avg_cpu']:.1f}%")
    print(f"   Peak CPU: {result['max_cpu']:.1f}%")
    print(f"   Total time: {total_time:.1f}s")
    
    print(f"\n📊 PC2 CPU readings: {result['cpu_readings']}")
    
    if result['max_cpu'] > 50:
        print("\n🎯 SUCCESS: PC2 CPU is being utilized!")
    elif result['max_cpu'] > 20:
        print("\n⚠️  MODERATE: Some PC2 CPU usage detected")
    else:
        print("\n❌ PROBLEM: PC2 CPU usage still very low")
        print("The threading approach may not be working on PC2")
    
    print("=== PC2 DIRECT TEST COMPLETE ===")

if __name__ == "__main__":
    main()
