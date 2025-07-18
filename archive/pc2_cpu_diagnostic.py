#!/usr/bin/env python3
"""
Targeted PC2 CPU diagnostic
"""
import ray
import psutil
import concurrent.futures
import time
import threading
import os

@ray.remote(num_cpus=12, resources={"node:192.168.1.11": 1})
def pc2_cpu_task():
    """Task specifically targeted to PC2 with threading"""
    print(f"Task running on node: {ray.get_runtime_context().node_id}")
    print(f"Process ID: {os.getpid()}")
    print(f"Thread ID: {threading.get_ident()}")
    
    def cpu_work():
        """CPU intensive work"""
        start_time = time.time()
        counter = 0
        while time.time() - start_time < 10:  # 10 seconds
            counter += sum(i**2 for i in range(1000))
        return counter
    
    print("Starting 12 threads for CPU saturation...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        futures = [executor.submit(cpu_work) for _ in range(12)]
        
        # Wait and monitor
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            result = future.result()
            print(f"Thread {i} completed with result: {result}")
    
    return "PC2 task completed"

@ray.remote(num_cpus=12, resources={"node:192.168.1.10": 1})  
def pc1_cpu_task():
    """Task specifically targeted to PC1 with threading"""
    print(f"Task running on node: {ray.get_runtime_context().node_id}")
    print(f"Process ID: {os.getpid()}")
    print(f"Thread ID: {threading.get_ident()}")
    
    def cpu_work():
        """CPU intensive work"""
        start_time = time.time()
        counter = 0
        while time.time() - start_time < 10:  # 10 seconds
            counter += sum(i**2 for i in range(1000))
        return counter
    
    print("Starting 12 threads for CPU saturation...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        futures = [executor.submit(cpu_work) for _ in range(12)]
        
        # Wait and monitor
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            result = future.result()
            print(f"Thread {i} completed with result: {result}")
    
    return "PC1 task completed"

def main():
    print("=== TARGETED PC2 CPU DIAGNOSTIC ===")
    
    # Initialize Ray
    if not ray.is_initialized():
        try:
            ray.init(address='ray://192.168.1.10:10001')
            print("✅ Ray connected successfully")
        except Exception as e:
            print(f"❌ Failed to connect to Ray: {e}")
            return
    
    # Get cluster resources
    resources = ray.cluster_resources()
    print(f"📊 Cluster resources: {resources}")
    
    print("\n🧪 Starting targeted PC2 and PC1 tests...")
    print("Monitor task manager on both PCs!")
    
    # Launch targeted tasks
    pc2_future = pc2_cpu_task.remote()
    pc1_future = pc1_cpu_task.remote()
    
    # Monitor PC1 CPU (this machine)
    print("\n🖥️  Monitoring PC1 CPU usage:")
    for i in range(12):  # 12 seconds
        cpu_usage = psutil.cpu_percent(interval=1)
        print(f"  Second {i+1}: CPU usage: {cpu_usage:.1f}%")
    
    # Wait for completion
    print("\n⏳ Waiting for tasks to complete...")
    pc2_result = ray.get(pc2_future)
    pc1_result = ray.get(pc1_future)
    
    print(f"✅ PC2 result: {pc2_result}")
    print(f"✅ PC1 result: {pc1_result}")
    
    print("=== DIAGNOSTIC COMPLETE ===")
    print("Check PC2 task manager to see if CPU usage increased during the test!")

if __name__ == "__main__":
    main()
