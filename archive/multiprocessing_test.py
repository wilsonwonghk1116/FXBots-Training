#!/usr/bin/env python3
"""
Test multiprocessing vs threading for CPU saturation in Ray
"""
import ray
import psutil
import concurrent.futures
import multiprocessing as mp
import time
import os

@ray.remote(num_cpus=12, resources={"node:192.168.1.11": 1})
def pc2_multiprocessing_task():
    """PC2 task using multiprocessing instead of threading"""
    print(f"PC2 Multiprocessing task running on node: {ray._private.services.get_node_ip_address()}")
    print(f"Process ID: {os.getpid()}")
    
    def cpu_work(duration=8):
        """CPU intensive work"""
        start_time = time.time()
        counter = 0
        while time.time() - start_time < duration:
            counter += sum(i**2 for i in range(2000))  # Increased workload
        return counter
    
    print("Starting 12 processes for CPU saturation...")
    
    # Use multiprocessing instead of threading
    with mp.Pool(processes=12) as pool:
        results = pool.map(cpu_work, [8] * 12)  # 8 seconds each
    
    print(f"Multiprocessing completed with {len(results)} results")
    return f"PC2 multiprocessing completed: {sum(results)}"

@ray.remote(num_cpus=12, resources={"node:192.168.1.10": 1})  
def pc1_multiprocessing_task():
    """PC1 task using multiprocessing instead of threading"""
    print(f"PC1 Multiprocessing task running on node: {ray._private.services.get_node_ip_address()}")
    print(f"Process ID: {os.getpid()}")
    
    def cpu_work(duration=8):
        """CPU intensive work"""
        start_time = time.time()
        counter = 0
        while time.time() - start_time < duration:
            counter += sum(i**2 for i in range(2000))  # Increased workload
        return counter
    
    print("Starting 12 processes for CPU saturation...")
    
    # Use multiprocessing instead of threading
    with mp.Pool(processes=12) as pool:
        results = pool.map(cpu_work, [8] * 12)  # 8 seconds each
    
    print(f"Multiprocessing completed with {len(results)} results")
    return f"PC1 multiprocessing completed: {sum(results)}"

@ray.remote(num_cpus=12, resources={"node:192.168.1.11": 1})
def pc2_threading_task():
    """PC2 task using threading for comparison"""
    print(f"PC2 Threading task running on node: {ray._private.services.get_node_ip_address()}")
    print(f"Process ID: {os.getpid()}")
    
    def cpu_work():
        """CPU intensive work"""
        start_time = time.time()
        counter = 0
        while time.time() - start_time < 8:
            counter += sum(i**2 for i in range(2000))  # Increased workload
        return counter
    
    print("Starting 12 threads for CPU saturation...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        futures = [executor.submit(cpu_work) for _ in range(12)]
        results = [future.result() for future in futures]
    
    return f"PC2 threading completed: {sum(results)}"

def main():
    print("=== MULTIPROCESSING VS THREADING CPU TEST ===")
    
    # Initialize Ray
    if not ray.is_initialized():
        try:
            ray.init(address='ray://192.168.1.10:10001')
            print("✅ Ray connected successfully")
        except Exception as e:
            print(f"❌ Failed to connect to Ray: {e}")
            return
    
    print("\n🧪 Test 1: Multiprocessing approach...")
    print("This should saturate CPU cores properly!")
    
    # Test multiprocessing approach
    start_time = time.time()
    
    pc2_mp_future = pc2_multiprocessing_task.remote()
    pc1_mp_future = pc1_multiprocessing_task.remote()
    
    # Monitor CPU usage
    print("\n🖥️  Monitoring PC1 CPU usage (multiprocessing):")
    cpu_readings = []
    for i in range(10):  # 10 seconds
        cpu_usage = psutil.cpu_percent(interval=1)
        cpu_readings.append(cpu_usage)
        print(f"  Second {i+1}: CPU usage: {cpu_usage:.1f}%")
    
    # Get results
    pc2_mp_result = ray.get(pc2_mp_future)
    pc1_mp_result = ray.get(pc1_mp_future)
    
    mp_time = time.time() - start_time
    avg_cpu_mp = sum(cpu_readings) / len(cpu_readings)
    max_cpu_mp = max(cpu_readings)
    
    print(f"✅ Multiprocessing results:")
    print(f"   PC2: {pc2_mp_result}")
    print(f"   PC1: {pc1_mp_result}")
    print(f"   Time: {mp_time:.1f}s, Avg CPU: {avg_cpu_mp:.1f}%, Max CPU: {max_cpu_mp:.1f}%")
    
    print("\n🧪 Test 2: Threading approach (for comparison)...")
    
    # Test threading approach
    start_time = time.time()
    
    pc2_th_future = pc2_threading_task.remote()
    
    # Monitor CPU usage
    print("\n🖥️  Monitoring PC1 CPU usage (threading):")
    cpu_readings = []
    for i in range(10):  # 10 seconds
        cpu_usage = psutil.cpu_percent(interval=1)
        cpu_readings.append(cpu_usage)
        print(f"  Second {i+1}: CPU usage: {cpu_usage:.1f}%")
    
    # Get results
    pc2_th_result = ray.get(pc2_th_future)
    
    th_time = time.time() - start_time
    avg_cpu_th = sum(cpu_readings) / len(cpu_readings)
    max_cpu_th = max(cpu_readings)
    
    print(f"✅ Threading results:")
    print(f"   PC2: {pc2_th_result}")
    print(f"   Time: {th_time:.1f}s, Avg CPU: {avg_cpu_th:.1f}%, Max CPU: {max_cpu_th:.1f}%")
    
    print("\n📊 COMPARISON:")
    print(f"   Multiprocessing: Avg {avg_cpu_mp:.1f}%, Max {max_cpu_mp:.1f}%")
    print(f"   Threading:       Avg {avg_cpu_th:.1f}%, Max {max_cpu_th:.1f}%")
    
    if max_cpu_mp > max_cpu_th * 1.5:
        print("🎯 SOLUTION: Use multiprocessing instead of threading for CPU saturation!")
    else:
        print("🤔 Both approaches show similar CPU usage - deeper investigation needed")
    
    print("=== TEST COMPLETE ===")

if __name__ == "__main__":
    main()
