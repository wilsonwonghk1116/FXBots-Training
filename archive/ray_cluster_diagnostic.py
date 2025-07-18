#!/usr/bin/env python3
"""
Ray Cluster Diagnostic Tool
===========================
Comprehensive investigation to identify why Worker PC 2 is not being utilized.
"""

import ray
import time
import torch
import socket
import os

@ray.remote(num_cpus=1, num_gpus=0.5)
class DiagnosticWorker:
    def __init__(self, worker_id):
        self.worker_id = worker_id
    
    def get_system_info(self):
        """Get comprehensive system information"""
        info = {
            "worker_id": self.worker_id,
            "hostname": socket.gethostname(),
            "node_id": ray.runtime_context.get_runtime_context().get_node_id(),
            "worker_id_ray": ray.runtime_context.get_runtime_context().get_worker_id(),
            "current_task_id": ray.runtime_context.get_runtime_context().get_task_id(),
        }
        
        # GPU information
        try:
            import torch
            if torch.cuda.is_available():
                info["cuda_available"] = True
                info["gpu_count"] = torch.cuda.device_count()
                info["current_device"] = torch.cuda.current_device()
                info["gpu_name"] = torch.cuda.get_device_name(0)
                info["gpu_memory_total"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
            else:
                info["cuda_available"] = False
        except:
            info["cuda_available"] = "error"
        
        # CPU information
        import psutil
        info["cpu_count"] = psutil.cpu_count()
        info["cpu_percent"] = psutil.cpu_percent(interval=1)
        info["memory_percent"] = psutil.virtual_memory().percent
        
        return info

def main():
    print("🔍 RAY CLUSTER DIAGNOSTIC TOOL")
    print("=" * 50)
    
    # Connect to Ray cluster
    try:
        ray.init(address='192.168.1.10:6379', ignore_reinit_error=True)
        print("✅ Connected to Ray cluster")
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        return
    
    # Get cluster information
    print("\n📊 CLUSTER OVERVIEW:")
    cluster_resources = ray.cluster_resources()
    print(f"Total Resources: {cluster_resources}")
    
    available_resources = ray.available_resources()
    print(f"Available Resources: {available_resources}")
    
    # Get node information
    print("\n🖥️ NODE DETAILS:")
    nodes = ray.nodes()
    for i, node in enumerate(nodes):
        print(f"Node {i+1}:")
        print(f"  ID: {node['NodeID'][:12]}...")
        print(f"  Address: {node['NodeManagerAddress']}")
        print(f"  Alive: {node['Alive']}")
        print(f"  Resources: {node['Resources']}")
        print()
    
    # Test worker placement
    print("🚀 TESTING WORKER PLACEMENT:")
    print("Creating 2 diagnostic workers...")
    
    workers = []
    for i in range(2):
        worker = DiagnosticWorker.remote(i)
        workers.append(worker)
        print(f"✅ Created worker {i}")
    
    # Get system information from each worker
    print("\n🧪 COLLECTING SYSTEM INFO:")
    futures = [worker.get_system_info.remote() for worker in workers]
    
    try:
        results = ray.get(futures, timeout=30)
        
        for i, result in enumerate(results):
            print(f"\n📋 WORKER {i} DETAILS:")
            for key, value in result.items():
                print(f"  {key}: {value}")
        
        # Analysis
        print("\n🔬 ANALYSIS:")
        node_ids = [r['node_id'] for r in results]
        hostnames = [r['hostname'] for r in results]
        
        if len(set(node_ids)) == 1:
            print("❌ PROBLEM FOUND: Both workers running on the same node!")
            print(f"   Both workers on node: {node_ids[0][:12]}...")
        else:
            print("✅ Workers distributed across different nodes")
            
        if len(set(hostnames)) == 1:
            print("❌ PROBLEM FOUND: Both workers running on the same machine!")
            print(f"   Both workers on hostname: {hostnames[0]}")
        else:
            print("✅ Workers distributed across different machines")
            
        # GPU analysis
        gpu_names = [r.get('gpu_name', 'No GPU') for r in results]
        print(f"\nGPU Distribution:")
        for i, gpu in enumerate(gpu_names):
            print(f"  Worker {i}: {gpu}")
            
    except Exception as e:
        print(f"❌ Failed to get worker info: {e}")
    
    ray.shutdown()
    print("\n🔗 Ray cluster disconnected")

if __name__ == "__main__":
    main()
