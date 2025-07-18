#!/usr/bin/env python3
"""
COMPREHENSIVE DIAGNOSTIC SCRIPT for PC2 CPU Utilization Issue
==============================================================
This script performs deep analysis of the CPU utilization problem
and provides detailed diagnostics to identify the root cause.
"""

import ray
import psutil
import time
import os
import threading
import concurrent.futures
import subprocess
import numpy as np
from datetime import datetime

def check_system_resources():
    """Check system resources and Ray cluster status"""
    print("=" * 60)
    print("SYSTEM RESOURCES DIAGNOSTIC")
    print("=" * 60)
    
    # Basic system info
    print(f"Hostname: {os.uname().nodename}")
    print(f"CPU cores (logical): {psutil.cpu_count(logical=True)}")
    print(f"CPU cores (physical): {psutil.cpu_count(logical=False)}")
    print(f"Current CPU usage: {psutil.cpu_percent(interval=1):.1f}%")
    
    # Memory info
    memory = psutil.virtual_memory()
    print(f"Memory total: {memory.total / (1024**3):.1f} GB")
    print(f"Memory usage: {memory.percent:.1f}%")
    
    # Ray cluster info
    if ray.is_initialized():
        cluster_resources = ray.cluster_resources()
        cluster_nodes = ray.nodes()
        print(f"\nRay cluster resources: {cluster_resources}")
        print(f"Ray active nodes: {len([n for n in cluster_nodes if n['Alive']])}")
        
        for node in cluster_nodes:
            if node['Alive']:
                print(f"  Node: {node['NodeManagerAddress']} - Resources: {node['Resources']}")
    else:
        print("Ray is not initialized")

def test_threading_cpu_saturation():
    """Test threading-based CPU saturation"""
    print("\n" + "=" * 60)
    print("THREADING CPU SATURATION TEST")
    print("=" * 60)
    
    def cpu_work(thread_id, duration=5):
        """CPU intensive work"""
        print(f"Thread {thread_id} starting CPU work...")
        end_time = time.time() + duration
        iterations = 0
        while time.time() < end_time:
            # CPU intensive operations
            data = np.random.rand(300, 300)
            result = np.dot(data, data.T)
            iterations += 1
        print(f"Thread {thread_id} completed {iterations} iterations")
        return iterations
    
    # Test with different thread counts
    num_cores = psutil.cpu_count(logical=True)
    target_threads = int(num_cores * 0.75)
    
    print(f"Testing {target_threads} threads on {num_cores} cores...")
    
    # Measure CPU before
    cpu_before = psutil.cpu_percent(interval=1)
    print(f"CPU usage before test: {cpu_before:.1f}%")
    
    # Run threading test
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=target_threads) as executor:
        futures = []
        for i in range(target_threads):
            future = executor.submit(cpu_work, i, 10)  # 10 seconds of work
            futures.append(future)
        
        # Monitor CPU during execution
        monitor_duration = 8  # Monitor for 8 seconds
        cpu_readings = []
        for _ in range(monitor_duration):
            cpu_usage = psutil.cpu_percent(interval=1)
            cpu_readings.append(cpu_usage)
            print(f"  CPU usage during test: {cpu_usage:.1f}%")
        
        # Wait for completion
        results = concurrent.futures.as_completed(futures)
        total_iterations = sum([f.result() for f in futures])
    
    end_time = time.time()
    avg_cpu = sum(cpu_readings) / len(cpu_readings)
    max_cpu = max(cpu_readings)
    
    print(f"Test completed in {end_time - start_time:.1f} seconds")
    print(f"Total iterations: {total_iterations}")
    print(f"Average CPU usage: {avg_cpu:.1f}%")
    print(f"Peak CPU usage: {max_cpu:.1f}%")
    
    return avg_cpu, max_cpu

@ray.remote(num_cpus=12, num_gpus=0)
def test_ray_cpu_allocation(worker_id, test_duration=15):
    """Test Ray CPU allocation and threading within Ray task"""
    import psutil
    import time
    import numpy as np
    import concurrent.futures
    import os
    import threading
    
    node_name = os.uname().nodename
    pid = os.getpid()
    
    print(f"Ray worker {worker_id} started on {node_name} (PID: {pid})")
    
    # Check process CPU affinity
    try:
        process = psutil.Process(pid)
        cpu_affinity = process.cpu_affinity()
        print(f"Worker {worker_id} CPU affinity: {cpu_affinity} (cores: {len(cpu_affinity)})")
    except Exception as e:
        print(f"Worker {worker_id} could not check CPU affinity: {e}")
    
    def cpu_intensive_work(thread_id, duration=1):
        """CPU-intensive work for threading test"""
        end_time = time.time() + duration
        iterations = 0
        while time.time() < end_time:
            data = np.random.rand(200, 200)
            result1 = np.dot(data, data.T)
            result2 = np.fft.fft2(result1[:50, :50])
            iterations += 1
        return iterations
    
    # CPU saturation test within Ray task
    num_cores = psutil.cpu_count(logical=True)
    target_threads = min(12, int(num_cores * 0.75))  # Use allocated CPUs or 75% of available
    
    print(f"Worker {worker_id} starting {target_threads} threads for CPU saturation...")
    
    cpu_readings = []
    total_iterations = 0
    
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=target_threads) as executor:
        # Submit continuous work
        active_futures = []
        
        while time.time() - start_time < test_duration:
            # Submit new work if needed
            if len(active_futures) < target_threads:
                future = executor.submit(cpu_intensive_work, len(active_futures), 1)
                active_futures.append(future)
            
            # Check completed work
            done_futures = []
            for future in active_futures:
                if future.done():
                    try:
                        iterations = future.result()
                        total_iterations += iterations
                    except Exception as e:
                        print(f"Worker {worker_id} thread error: {e}")
                    done_futures.append(future)
            
            # Remove completed futures
            for future in done_futures:
                active_futures.remove(future)
            
            # Monitor CPU usage
            cpu_usage = psutil.cpu_percent(interval=0.5)
            cpu_readings.append(cpu_usage)
            
            elapsed = time.time() - start_time
            if elapsed % 2 < 0.5:  # Log every 2 seconds
                print(f"Worker {worker_id} on {node_name}: {elapsed:.1f}s, CPU: {cpu_usage:.1f}%, active threads: {len(active_futures)}")
        
        # Wait for remaining work to complete
        for future in active_futures:
            try:
                iterations = future.result()
                total_iterations += iterations
            except Exception as e:
                print(f"Worker {worker_id} final thread error: {e}")
    
    end_time = time.time()
    duration = end_time - start_time
    avg_cpu = sum(cpu_readings) / len(cpu_readings) if cpu_readings else 0
    max_cpu = max(cpu_readings) if cpu_readings else 0
    
    result = {
        'worker_id': worker_id,
        'node': node_name,
        'pid': pid,
        'duration': duration,
        'total_iterations': total_iterations,
        'avg_cpu': avg_cpu,
        'max_cpu': max_cpu,
        'target_threads': target_threads,
        'cpu_readings_count': len(cpu_readings)
    }
    
    print(f"Worker {worker_id} on {node_name} completed:")
    print(f"  Duration: {duration:.1f}s")
    print(f"  Total iterations: {total_iterations}")
    print(f"  Average CPU: {avg_cpu:.1f}%")
    print(f"  Peak CPU: {max_cpu:.1f}%")
    print(f"  Target threads: {target_threads}")
    
    return result

def main():
    """Main diagnostic function"""
    print("STARTING COMPREHENSIVE CPU UTILIZATION DIAGNOSTIC")
    print("Date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # Step 1: Check system resources
    check_system_resources()
    
    # Step 2: Test threading outside Ray
    print("\n" + "=" * 60)
    print("STEP 2: Testing threading CPU saturation (outside Ray)")
    print("=" * 60)
    avg_cpu, max_cpu = test_threading_cpu_saturation()
    
    # Step 3: Test Ray cluster if available
    if ray.is_initialized():
        print("\n" + "=" * 60)
        print("STEP 3: Testing Ray CPU allocation")
        print("=" * 60)
        
        # Get cluster info
        cluster_resources = ray.cluster_resources()
        total_cpus = cluster_resources.get('CPU', 0)
        total_gpus = cluster_resources.get('GPU', 0)
        
        print(f"Cluster has {total_cpus} CPUs and {total_gpus} GPUs")
        
        if total_cpus >= 12:
            # Create placement group for testing
            from ray.util.placement_group import placement_group
            
            bundles = [{"CPU": 12}]  # Request 12 CPUs for testing
            pg = placement_group(bundles, strategy="STRICT_SPREAD")
            ray.get(pg.ready())
            print("Placement group created successfully")
            
            # Launch test task
            print("Launching Ray CPU test task...")
            future = test_ray_cpu_allocation.options(
                placement_group=pg,
                placement_group_bundle_index=0
            ).remote(0, 15)  # 15 seconds test
            
            result = ray.get(future)
            
            print("\nRay CPU Test Results:")
            print(f"  Node: {result['node']}")
            print(f"  PID: {result['pid']}")
            print(f"  Duration: {result['duration']:.1f}s")
            print(f"  Total iterations: {result['total_iterations']}")
            print(f"  Average CPU: {result['avg_cpu']:.1f}%")
            print(f"  Peak CPU: {result['max_cpu']:.1f}%")
            print(f"  Target threads: {result['target_threads']}")
            
            # Clean up placement group
            ray.util.remove_placement_group(pg)
            
        else:
            print("Insufficient CPUs in cluster for Ray testing")
    else:
        print("\nRay is not initialized - skipping Ray tests")
    
    # Step 4: System-level diagnostics
    print("\n" + "=" * 60)
    print("STEP 4: System-level diagnostics")
    print("=" * 60)
    
    try:
        # Check taskset availability
        result = subprocess.run(['which', 'taskset'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ taskset command is available")
        else:
            print("❌ taskset command not found")
            
        # Check CPU frequency scaling
        if os.path.exists('/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor'):
            with open('/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor', 'r') as f:
                governor = f.read().strip()
                print(f"CPU frequency governor: {governor}")
        else:
            print("CPU frequency scaling info not available")
            
        # Check CPU isolation
        if os.path.exists('/proc/cmdline'):
            with open('/proc/cmdline', 'r') as f:
                cmdline = f.read().strip()
                if 'isolcpus' in cmdline:
                    print(f"CPU isolation detected: {cmdline}")
                else:
                    print("No CPU isolation detected")
        
        # Check running processes
        python_processes = []
        ray_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent']):
            try:
                if 'python' in proc.info['name'].lower():
                    python_processes.append(proc.info)
                if 'ray' in ' '.join(proc.info['cmdline'] or []).lower():
                    ray_processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        print(f"\nFound {len(python_processes)} Python processes")
        print(f"Found {len(ray_processes)} Ray processes")
        
    except Exception as e:
        print(f"System diagnostics error: {e}")
    
    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
