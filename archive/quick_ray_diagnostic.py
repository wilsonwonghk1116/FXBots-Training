#!/usr/bin/env python3
"""
Quick Ray CPU diagnostic for connected cluster
"""
import ray
import psutil
import concurrent.futures
import time
import threading

def cpu_intensive_task(duration=5):
    """CPU intensive task for testing"""
    start_time = time.time()
    counter = 0
    while time.time() - start_time < duration:
        counter += sum(i**2 for i in range(1000))
    return counter

@ray.remote(num_cpus=1)
def ray_cpu_task(duration=5):
    """Ray remote CPU intensive task"""
    return cpu_intensive_task(duration)

@ray.remote(num_cpus=12)
def ray_heavy_cpu_task(duration=5):
    """Ray remote CPU intensive task with 12 CPU allocation"""
    
    # Use threading to saturate allocated CPUs
    def thread_work():
        return cpu_intensive_task(duration)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        futures = [executor.submit(thread_work) for _ in range(12)]
        results = [future.result() for future in futures]
    
    return sum(results)

def main():
    print("=== QUICK RAY CPU DIAGNOSTIC ===")
    
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
    
    total_cpus = resources.get('CPU', 0)
    total_gpus = resources.get('GPU', 0)
    
    print(f"🔧 Total CPUs: {total_cpus}, Total GPUs: {total_gpus}")
    
    # Test 1: Single CPU tasks on both nodes
    print("\n🧪 Test 1: Single CPU tasks...")
    start_time = time.time()
    cpu_usage_before = psutil.cpu_percent(interval=1)
    
    # Launch tasks
    futures = []
    for i in range(int(total_cpus)):
        future = ray_cpu_task.remote(duration=3)
        futures.append(future)
    
    # Monitor CPU while tasks run
    cpu_readings = []
    for _ in range(4):  # 4 seconds of monitoring
        cpu_usage = psutil.cpu_percent(interval=1)
        cpu_readings.append(cpu_usage)
        print(f"  🖥️  CPU usage: {cpu_usage:.1f}%")
    
    # Wait for completion
    results = ray.get(futures)
    end_time = time.time()
    
    print(f"✅ Test 1 completed in {end_time - start_time:.1f} seconds")
    print(f"📈 Average CPU during test: {sum(cpu_readings)/len(cpu_readings):.1f}%")
    print(f"🔥 Peak CPU during test: {max(cpu_readings):.1f}%")
    
    # Test 2: Heavy CPU tasks (12 CPUs each)
    print("\n🧪 Test 2: Heavy CPU tasks (12 CPUs each)...")
    start_time = time.time()
    
    # Launch 2 heavy tasks (one should go to each node)
    futures = []
    for i in range(2):
        future = ray_heavy_cpu_task.remote(duration=5)
        futures.append(future)
    
    # Monitor CPU while tasks run
    cpu_readings = []
    for _ in range(6):  # 6 seconds of monitoring
        cpu_usage = psutil.cpu_percent(interval=1)
        cpu_readings.append(cpu_usage)
        print(f"  🖥️  CPU usage: {cpu_usage:.1f}%")
    
    # Wait for completion
    results = ray.get(futures)
    end_time = time.time()
    
    print(f"✅ Test 2 completed in {end_time - start_time:.1f} seconds")
    print(f"📈 Average CPU during test: {sum(cpu_readings)/len(cpu_readings):.1f}%")
    print(f"🔥 Peak CPU during test: {max(cpu_readings):.1f}%")
    
    # Get final cluster status
    print("\n📊 Final cluster status:")
    try:
        import subprocess
        result = subprocess.run(['ray', 'status'], capture_output=True, text=True, timeout=10)
        print(result.stdout)
    except Exception as e:
        print(f"Could not get ray status: {e}")
    
    print("=== DIAGNOSTIC COMPLETE ===")

if __name__ == "__main__":
    main()
