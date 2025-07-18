#!/usr/bin/env python3
"""
SIMPLIFIED DUAL-GPU TEST - Node Placement Verification
======================================================
Tests explicit node placement to ensure Worker PC 2 utilization.
"""

import ray
import time
import socket
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@ray.remote(num_cpus=2, num_gpus=1.0)  # FULL GPU per worker
class TestWorker:
    def __init__(self, worker_id):
        self.worker_id = worker_id
    
    def get_info_and_work(self):
        """Get system info and do some GPU work"""
        import torch
        import socket
        
        # System info
        hostname = socket.gethostname()
        node_id = ray.runtime_context.get_runtime_context().get_node_id()
        
        # GPU info
        gpu_name = "No GPU"
        gpu_work_result = "No GPU work"
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            
            # Do some actual GPU work to utilize resources
            logger.info(f"Worker {self.worker_id}: Starting GPU work on {hostname}")
            
            # Create tensors and do computation for 30 seconds
            start_time = time.time()
            iterations = 0
            
            device = torch.device("cuda:0")
            torch.cuda.set_device(0)
            
            # Create large tensors to use GPU memory and compute
            A = torch.randn(2048, 2048, device=device, dtype=torch.float16)
            B = torch.randn(2048, 2048, device=device, dtype=torch.float16)
            
            while time.time() - start_time < 30:  # Run for 30 seconds
                # Matrix multiplication (uses GPU compute)
                C = torch.mm(A, B)
                A = torch.nn.functional.relu(C)  # Activation function
                iterations += 1
                
                if iterations % 100 == 0:
                    logger.info(f"Worker {self.worker_id} on {hostname}: {iterations} iterations")
            
            gpu_work_result = f"Completed {iterations} GPU iterations"
            
        return {
            "worker_id": self.worker_id,
            "hostname": hostname,
            "node_id": node_id[:12],
            "gpu_name": gpu_name,
            "gpu_work": gpu_work_result
        }

def main():
    logger.info("🧪 SIMPLIFIED DUAL-GPU TEST - NODE PLACEMENT VERIFICATION")
    logger.info("=" * 60)
    
    # Connect to Ray
    ray.init(address='192.168.1.10:6379', ignore_reinit_error=True)
    logger.info("✅ Connected to Ray cluster")
    
    # Get available nodes
    nodes = ray.nodes()
    available_nodes = [node['NodeID'] for node in nodes if node['Alive']]
    logger.info(f"📍 Available nodes: {len(available_nodes)}")
    
    # Create workers with explicit placement
    workers = []
    
    if len(available_nodes) >= 2:
        logger.info("🎯 Creating workers with EXPLICIT NODE PLACEMENT:")
        
        # Worker 0 on Node 0
        worker0 = TestWorker.options(
            resources={"node:" + available_nodes[0]: 0.01}
        ).remote(0)
        workers.append(worker0)
        logger.info(f"✅ Worker 0 placed on node {available_nodes[0][:12]}")
        
        # Worker 1 on Node 1
        worker1 = TestWorker.options(
            resources={"node:" + available_nodes[1]: 0.01}
        ).remote(1)
        workers.append(worker1)
        logger.info(f"✅ Worker 1 placed on node {available_nodes[1][:12]}")
        
    else:
        logger.warning("⚠️ Not enough nodes for explicit placement")
        return
    
    logger.info("🚀 Starting 30-second GPU utilization test...")
    logger.info("📊 Check Worker PC 2 for CPU/GPU usage now!")
    
    # Run workers
    futures = [worker.get_info_and_work.remote() for worker in workers]
    
    try:
        results = ray.get(futures, timeout=60)
        
        logger.info("\n📋 RESULTS:")
        for result in results:
            logger.info(f"Worker {result['worker_id']}:")
            logger.info(f"  Hostname: {result['hostname']}")
            logger.info(f"  Node: {result['node_id']}")
            logger.info(f"  GPU: {result['gpu_name']}")
            logger.info(f"  Work: {result['gpu_work']}")
            logger.info("")
        
        # Check if distributed
        hostnames = [r['hostname'] for r in results]
        nodes = [r['node_id'] for r in results]
        
        if len(set(hostnames)) > 1:
            logger.info("✅ SUCCESS: Workers distributed across different machines!")
        else:
            logger.info("❌ ISSUE: All workers on same machine")
            
        if len(set(nodes)) > 1:
            logger.info("✅ SUCCESS: Workers distributed across different nodes!")
        else:
            logger.info("❌ ISSUE: All workers on same node")
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
    
    ray.shutdown()
    logger.info("🔗 Disconnected from Ray cluster")

if __name__ == "__main__":
    main()
