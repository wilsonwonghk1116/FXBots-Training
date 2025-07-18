#!/usr/bin/env python3
"""
Quick Ray Cluster Test for NEW IP Configuration
Tests both head and worker nodes are connected and responsive
"""

import ray
import time
import socket
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_local_ip():
    """Get the local IP address"""
    try:
        # Connect to a dummy address to get local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "unknown"

@ray.remote
def test_node_function(node_id, test_data="hello_world"):
    """Simple function to test Ray remote execution"""
    import platform
    import psutil
    import torch
    
    result = {
        "node_id": node_id,
        "hostname": platform.node(),
        "test_data": test_data,
        "cpu_count": psutil.cpu_count(),
        "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "timestamp": time.time()
    }
    
    if torch.cuda.is_available():
        result["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    
    return result

@ray.remote(num_cpus=4)
def test_cpu_intensive_task(task_id, iterations=1000000):
    """Test CPU-intensive task to verify resource allocation"""
    import time
    start_time = time.time()
    
    # Simple CPU-bound calculation
    total = 0
    for i in range(iterations):
        total += i * i
    
    end_time = time.time()
    
    return {
        "task_id": task_id,
        "iterations": iterations,
        "duration": end_time - start_time,
        "result": total % 1000000  # Just to use the result
    }

@ray.remote(num_gpus=0.1)  # Small GPU allocation for testing
def test_gpu_task(task_id):
    """Test GPU availability and basic operations"""
    import torch
    
    if not torch.cuda.is_available():
        return {"task_id": task_id, "error": "CUDA not available"}
    
    try:
        # Create a small tensor on GPU
        x = torch.randn(100, 100).cuda()
        y = torch.randn(100, 100).cuda()
        z = torch.mm(x, y)
        
        return {
            "task_id": task_id,
            "gpu_device": torch.cuda.current_device(),
            "gpu_name": torch.cuda.get_device_name(torch.cuda.current_device()),
            "tensor_result": z.sum().item(),
            "memory_allocated": torch.cuda.memory_allocated(torch.cuda.current_device()),
            "memory_reserved": torch.cuda.memory_reserved(torch.cuda.current_device())
        }
    except Exception as e:
        return {"task_id": task_id, "error": str(e)}

def main():
    print("🔬 NEW RAY CLUSTER TEST STARTING...")
    print("=" * 50)
    
    local_ip = get_local_ip()
    print(f"🖥️  Local machine IP: {local_ip}")
    
    # Expected IPs
    head_ip = "192.168.1.10"
    worker_ip = "192.168.1.11"
    
    if local_ip == head_ip:
        print("🎯 Running on HEAD PC")
    elif local_ip == worker_ip:
        print("🎯 Running on WORKER PC")
    else:
        print(f"⚠️  Unknown IP {local_ip} - expected {head_ip} or {worker_ip}")
    
    try:
        # Connect to Ray cluster
        print("\n🔌 Connecting to Ray cluster...")
        ray.init(address='auto')
        
        # Get cluster info
        print(f"✅ Connected to Ray cluster!")
        print(f"📊 Cluster resources: {ray.cluster_resources()}")
        print(f"🏷️  Available nodes: {len(ray.nodes())}")
        
        # Test basic remote function
        print("\n🧪 Testing basic remote functions...")
        futures = []
        for i in range(6):  # Test 6 parallel tasks
            future = test_node_function.remote(f"node_test_{i}", f"test_data_{i}")
            futures.append(future)
        
        results = ray.get(futures)
        
        print("📋 Node test results:")
        for result in results:
            print(f"   Node: {result['hostname']}, CPUs: {result['cpu_count']}, "
                  f"Memory: {result['memory_gb']}GB, GPUs: {result['gpu_count']}")
            if 'gpu_names' in result:
                print(f"      GPU: {result['gpu_names']}")
        
        # Test CPU-intensive tasks
        print("\n💪 Testing CPU-intensive tasks...")
        cpu_futures = []
        for i in range(4):
            future = test_cpu_intensive_task.remote(f"cpu_test_{i}", 500000)
            cpu_futures.append(future)
        
        cpu_results = ray.get(cpu_futures)
        avg_duration = sum(r['duration'] for r in cpu_results) / len(cpu_results)
        print(f"   Average CPU task duration: {avg_duration:.3f}s")
        
        # Test GPU tasks if available
        gpu_count = ray.cluster_resources().get('GPU', 0)
        if gpu_count > 0:
            print(f"\n🚀 Testing GPU tasks ({gpu_count} GPUs available)...")
            gpu_futures = []
            for i in range(min(2, int(gpu_count))):  # Test up to 2 GPU tasks
                future = test_gpu_task.remote(f"gpu_test_{i}")
                gpu_futures.append(future)
            
            gpu_results = ray.get(gpu_futures)
            
            for result in gpu_results:
                if 'error' in result:
                    print(f"   GPU Test {result['task_id']}: ERROR - {result['error']}")
                else:
                    print(f"   GPU Test {result['task_id']}: {result['gpu_name']}, "
                          f"Memory: {result['memory_allocated']/1024**2:.1f}MB")
        else:
            print("\n⚠️  No GPUs detected in cluster")
        
        # Final status check
        print("\n📊 Final cluster status:")
        print(f"   Total nodes: {len(ray.nodes())}")
        print(f"   Active nodes: {len([n for n in ray.nodes() if n['Alive']])}")
        print(f"   Resources: {ray.cluster_resources()}")
        
        print("\n🎊 RAY CLUSTER TEST COMPLETED SUCCESSFULLY!")
        print("   Your cluster is ready for forex bot training! 🚀")
        
    except Exception as e:
        print(f"\n❌ RAY CLUSTER TEST FAILED!")
        print(f"   Error: {e}")
        print("   Please check your Ray cluster setup!")
        return False
    
    finally:
        try:
            ray.shutdown()
        except:
            pass
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Ready to run: RAY_CLUSTER=1 python run_stable_85_percent_trainer.py")
        exit(0)
    else:
        print("\n❌ Fix Ray cluster issues before training!")
        exit(1) 