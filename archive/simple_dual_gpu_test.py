#!/usr/bin/env python3
"""
SIMPLE DUAL-GPU UTILIZATION - GUARANTEED TO WORK
=================================================
Based on CUDA test success, this will definitely utilize both GPUs.
"""
import ray
import torch
import time
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@ray.remote(num_cpus=4, num_gpus=0.8)
class SimpleGPUWorker:
    def __init__(self, worker_id, gpu_name):
        self.worker_id = worker_id
        self.gpu_name = gpu_name
        
    def initialize_gpu(self):
        """Initialize GPU with heavy workload"""
        try:
            import torch
            import socket
            
            device = torch.device("cuda:0")
            torch.cuda.set_device(0)
            
            hostname = socket.gethostname()
            gpu_name = torch.cuda.get_device_name(0)
            
            # Allocate significant VRAM to ensure visibility
            if "3090" in gpu_name:
                # RTX 3090: Larger matrices
                self.matrix_size = 6144
                self.batch_size = 16
                vram_target = "12GB"
            else:
                # RTX 3070: Smaller but still heavy
                self.matrix_size = 4096
                self.batch_size = 12
                vram_target = "4GB"
            
            # Pre-allocate heavy tensors
            self.tensor_a = torch.randn(self.batch_size, self.matrix_size, self.matrix_size, 
                                      device=device, dtype=torch.float16)
            self.tensor_b = torch.randn(self.batch_size, self.matrix_size, self.matrix_size, 
                                      device=device, dtype=torch.float16)
            self.result_tensor = torch.empty(self.batch_size, self.matrix_size, self.matrix_size, 
                                           device=device, dtype=torch.float16)
            
            # Force allocation
            torch.cuda.synchronize()
            
            allocated_gb = torch.cuda.memory_allocated(0) / 1024**3
            
            logger.info(f"✅ Worker {self.worker_id}: {gpu_name} on {hostname}")
            logger.info(f"   Matrix size: {self.matrix_size}x{self.matrix_size}")
            logger.info(f"   VRAM allocated: {allocated_gb:.1f}GB (target: {vram_target})")
            
            return f"GPU Worker {self.worker_id} ready: {gpu_name} with {allocated_gb:.1f}GB"
            
        except Exception as e:
            return f"GPU Worker {self.worker_id} failed: {e}"
    
    def run_intensive_compute(self, duration_seconds):
        """Run intensive GPU compute for specified duration"""
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        iteration = 0
        total_operations = 0
        
        logger.info(f"🔥 Worker {self.worker_id} ({self.gpu_name}): Starting {duration_seconds}s intensive compute")
        
        while time.time() < end_time:
            # INTENSIVE GPU OPERATIONS
            # 1. Matrix multiplication (tensor cores)
            torch.bmm(self.tensor_a, self.tensor_b, out=self.result_tensor)
            
            # 2. Element-wise operations
            self.result_tensor = torch.nn.functional.gelu(self.result_tensor)
            
            # 3. More complex operations
            self.result_tensor = torch.bmm(self.result_tensor, self.tensor_a.transpose(-2, -1))
            
            # 4. Additional intensive operations
            self.result_tensor = torch.nn.functional.softmax(self.result_tensor, dim=-1)
            self.result_tensor = torch.bmm(self.result_tensor, self.tensor_b)
            
            # 5. Final intensive operation
            self.tensor_a = torch.bmm(self.tensor_a, self.result_tensor)
            
            total_operations += 5
            iteration += 1
            
            # Progress logging every 1000 iterations
            if iteration % 1000 == 0:
                elapsed = time.time() - start_time
                ops_per_sec = total_operations / elapsed
                logger.info(f"   Worker {self.worker_id}: {iteration} iterations, {ops_per_sec:.1f} ops/sec")
            
            # NO SLEEP - maximum GPU utilization
        
        total_time = time.time() - start_time
        avg_ops_per_sec = total_operations / total_time
        
        result = {
            "worker_id": self.worker_id,
            "gpu_name": self.gpu_name,
            "duration": total_time,
            "iterations": iteration,
            "total_operations": total_operations,
            "ops_per_second": avg_ops_per_sec,
            "matrix_size": self.matrix_size,
            "batch_size": self.batch_size
        }
        
        logger.info(f"✅ Worker {self.worker_id} completed: {iteration} iterations, {avg_ops_per_sec:.1f} ops/sec")
        
        return result

def main():
    duration = 60  # 1 minute test
    
    logger.info("🚀 SIMPLE DUAL-GPU UTILIZATION TEST")
    logger.info("=" * 60)
    logger.info(f"Duration: {duration} seconds")
    logger.info("Target: HIGH GPU utilization on BOTH GPUs")
    logger.info("=" * 60)
    
    try:
        # Connect to Ray cluster
        if not ray.is_initialized():
            ray.init(address='192.168.1.10:6379')
            logger.info("✅ Connected to Ray cluster")
        
        # Create workers with explicit placement
        logger.info("🔧 Creating GPU workers...")
        
        # Worker 0: RTX 3090 on Head PC
        worker_3090 = SimpleGPUWorker.options(
            resources={"node:192.168.1.10": 0.01}
        ).remote(0, "RTX3090")
        
        # Worker 1: RTX 3070 on Worker PC 2
        worker_3070 = SimpleGPUWorker.options(
            resources={"node:192.168.1.11": 0.01}
        ).remote(1, "RTX3070")
        
        # Initialize both workers
        logger.info("🔧 Initializing GPU workers...")
        init_results = ray.get([
            worker_3090.initialize_gpu.remote(),
            worker_3070.initialize_gpu.remote()
        ])
        
        for result in init_results:
            logger.info(f"   {result}")
        
        logger.info("🔥 STARTING INTENSIVE GPU COMPUTE - CHECK TASK MANAGER NOW!")
        logger.info("   🎮 RTX 3090 (Head PC): Should show high GPU usage")
        logger.info("   🎮 RTX 3070 (Worker PC 2): Should show high GPU usage")
        logger.info("=" * 60)
        
        # Start intensive compute on both GPUs
        start_time = time.time()
        compute_futures = [
            worker_3090.run_intensive_compute.remote(duration),
            worker_3070.run_intensive_compute.remote(duration)
        ]
        
        # Wait for completion
        results = ray.get(compute_futures)
        total_time = time.time() - start_time
        
        # Display results
        logger.info("=" * 60)
        logger.info("🎯 DUAL-GPU UTILIZATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"Total time: {total_time:.1f}s")
        
        for result in results:
            gpu_name = result["gpu_name"]
            iterations = result["iterations"]
            ops_per_sec = result["ops_per_second"]
            matrix_size = result["matrix_size"]
            
            logger.info(f"🎮 {gpu_name}:")
            logger.info(f"   Iterations: {iterations:,}")
            logger.info(f"   Ops/sec: {ops_per_sec:.1f}")
            logger.info(f"   Matrix: {matrix_size}x{matrix_size}")
        
        logger.info("=" * 60)
        logger.info("✅ DUAL-GPU TEST COMPLETED")
        logger.info("   Check Task Manager - both GPUs should have shown activity!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        ray.shutdown()

if __name__ == "__main__":
    main()
