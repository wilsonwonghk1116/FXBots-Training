#!/usr/bin/env python3
"""
SOLUTION: Fixed CPU saturation using numpy/numba for CPU-intensive work
This bypasses Python's GIL limitation by using C-optimized operations
"""
import ray
import psutil
import concurrent.futures
import time
import os
import numpy as np

try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("Warning: numba not available, using numpy only")

@jit(nopython=True) if HAS_NUMBA else lambda x: x
def cpu_intensive_math(size=10000, iterations=1000):
    """CPU-intensive mathematical operations that bypass GIL"""
    result = 0
    for i in range(iterations):
        # Create large arrays and perform operations
        a = np.random.rand(size)
        b = np.random.rand(size)
        
        # CPU-intensive operations
        c = np.dot(a, b)
        d = np.sum(a ** 2)
        e = np.sqrt(np.abs(b))
        
        result += c + d + np.sum(e)
    
    return result

def cpu_work_numpy(thread_id, duration=5):
    """CPU-intensive work using numpy operations"""
    start_time = time.time()
    iterations = 0
    
    while time.time() - start_time < duration:
        if HAS_NUMBA:
            # Numba-optimized version
            result = cpu_intensive_math(5000, 10)
        else:
            # Pure numpy version
            size = 5000
            a = np.random.rand(size)
            b = np.random.rand(size)
            c = np.dot(a, b)
            d = np.sum(a ** 2)
            e = np.sqrt(np.abs(b))
            result = c + d + np.sum(e)
        
        iterations += 1
    
    return iterations

@ray.remote(num_cpus=12, resources={"node:192.168.1.11": 1})
def pc2_fixed_cpu_task():
    """PC2 task with fixed CPU saturation using numpy/numba"""
    print(f"PC2 FIXED task running on node: {ray._private.services.get_node_ip_address()}")
    print(f"Process ID: {os.getpid()}")
    print(f"Numba available: {HAS_NUMBA}")
    
    # Use 12 threads with numpy-based CPU work
    print("Starting 12 threads with numpy/numba CPU saturation...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        futures = []
        
        # Start all threads
        for i in range(12):
            future = executor.submit(cpu_work_numpy, i, 8)  # 8 seconds each
            futures.append(future)
        
        # Collect results
        results = []
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            result = future.result()
            results.append(result)
            print(f"Thread {i} completed {result} iterations")
    
    return f"PC2 FIXED completed: {sum(results)} total iterations"

@ray.remote(num_cpus=12, resources={"node:192.168.1.10": 1})  
def pc1_fixed_cpu_task():
    """PC1 task with fixed CPU saturation using numpy/numba"""
    print(f"PC1 FIXED task running on node: {ray._private.services.get_node_ip_address()}")
    print(f"Process ID: {os.getpid()}")
    print(f"Numba available: {HAS_NUMBA}")
    
    # Use 12 threads with numpy-based CPU work
    print("Starting 12 threads with numpy/numba CPU saturation...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        futures = []
        
        # Start all threads
        for i in range(12):
            future = executor.submit(cpu_work_numpy, i, 8)  # 8 seconds each
            futures.append(future)
        
        # Collect results
        results = []
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            result = future.result()
            results.append(result)
            print(f"Thread {i} completed {result} iterations")
    
    return f"PC1 FIXED completed: {sum(results)} total iterations"

def main():
    print("=== FIXED CPU SATURATION TEST ===")
    print("Using numpy/numba to bypass Python GIL limitations")
    
    # Initialize Ray
    if not ray.is_initialized():
        try:
            ray.init(address='ray://192.168.1.10:10001')
            print("✅ Ray connected successfully")
        except Exception as e:
            print(f"❌ Failed to connect to Ray: {e}")
            return
    
    print("\n🧪 Testing FIXED approach with numpy/numba...")
    print("This should properly saturate CPU cores on both nodes!")
    
    # Launch fixed CPU tasks
    start_time = time.time()
    
    pc2_future = pc2_fixed_cpu_task.remote()
    pc1_future = pc1_fixed_cpu_task.remote()
    
    # Monitor CPU usage
    print("\n🖥️  Monitoring PC1 CPU usage:")
    cpu_readings = []
    for i in range(10):  # 10 seconds
        cpu_usage = psutil.cpu_percent(interval=1)
        cpu_readings.append(cpu_usage)
        print(f"  Second {i+1}: CPU usage: {cpu_usage:.1f}%")
    
    # Get results
    print("\n⏳ Waiting for tasks to complete...")
    pc2_result = ray.get(pc2_future)
    pc1_result = ray.get(pc1_future)
    
    total_time = time.time() - start_time
    avg_cpu = sum(cpu_readings) / len(cpu_readings)
    max_cpu = max(cpu_readings)
    
    print(f"\n✅ FIXED Results:")
    print(f"   PC2: {pc2_result}")
    print(f"   PC1: {pc1_result}")
    print(f"   Time: {total_time:.1f}s")
    print(f"   Average CPU: {avg_cpu:.1f}%")
    print(f"   Peak CPU: {max_cpu:.1f}%")
    
    if max_cpu > 50:
        print("\n🎯 SUCCESS: CPU saturation working!")
        print("This approach should work for the training script.")
    else:
        print("\n⚠️  Still low CPU usage - further investigation needed")
    
    print("\n💡 SOLUTION SUMMARY:")
    print("Replace pure Python loops with numpy/numba operations")
    print("This bypasses Python's GIL and enables true multi-core utilization")
    
    print("=== TEST COMPLETE ===")

if __name__ == "__main__":
    main()
