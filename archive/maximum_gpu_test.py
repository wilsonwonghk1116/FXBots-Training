#!/usr/bin/env python3
"""
MAXIMUM GPU UTILIZATION TEST - GUARANTEED VISIBLE
================================================
This will create sustained, heavy GPU load that MUST be visible in Task Manager
"""
import ray
import torch
import time
import logging
import numpy as np
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@ray.remote(num_cpus=8, num_gpus=0.9)
class MaxGPUWorker:
    def __init__(self, worker_id, gpu_name):
        self.worker_id = worker_id
        self.gpu_name = gpu_name
        
    def initialize_maximum_gpu_load(self):
        """Initialize GPU with MAXIMUM possible workload"""
        try:
            import torch
            import socket
            
            device = torch.device("cuda:0")
            torch.cuda.set_device(0)
            
            hostname = socket.gethostname()
            gpu_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            # MAXIMUM MEMORY ALLOCATION for each GPU
            if "3090" in gpu_name:
                # RTX 3090: Use 20GB+ for maximum visibility
                self.matrix_size = 8192
                self.batch_size = 20
                num_tensors = 15  # Many tensors
                vram_target = "20GB"
            else:
                # RTX 3070: Use 6GB+ for maximum visibility  
                self.matrix_size = 6144
                self.batch_size = 16
                num_tensors = 10  # Many tensors
                vram_target = "6GB"
            
            logger.info(f"🔥 Worker {self.worker_id}: Allocating MAXIMUM GPU memory...")
            logger.info(f"   Target: {vram_target} on {gpu_name}")
            logger.info(f"   Matrix: {self.matrix_size}x{self.matrix_size} x{num_tensors} tensors")
            
            # Allocate MANY heavy tensors for maximum VRAM usage
            self.tensors = []
            for i in range(num_tensors):
                tensor_set = {
                    'A': torch.randn(self.batch_size, self.matrix_size, self.matrix_size, 
                                   device=device, dtype=torch.float16),
                    'B': torch.randn(self.batch_size, self.matrix_size, self.matrix_size, 
                                   device=device, dtype=torch.float16),
                    'C': torch.empty(self.batch_size, self.matrix_size, self.matrix_size, 
                                   device=device, dtype=torch.float16)
                }
                self.tensors.append(tensor_set)
                
                # Force allocation
                for tensor in tensor_set.values():
                    tensor.zero_()
                
                torch.cuda.synchronize()
                
                # Check memory usage
                allocated_gb = torch.cuda.memory_allocated(0) / 1024**3
                logger.info(f"   Tensor set {i+1}/{num_tensors}: {allocated_gb:.1f}GB allocated")
                
                # Stop if we're getting close to memory limit
                if allocated_gb > total_memory * 0.85:  # 85% of total
                    logger.info(f"   Reached 85% memory limit, stopping allocation")
                    break
            
            final_allocated_gb = torch.cuda.memory_allocated(0) / 1024**3
            utilization_percent = final_allocated_gb / total_memory * 100
            
            logger.info(f"✅ Worker {self.worker_id}: {gpu_name} on {hostname}")
            logger.info(f"   FINAL VRAM: {final_allocated_gb:.1f}GB / {total_memory:.1f}GB ({utilization_percent:.1f}%)")
            logger.info(f"   Tensor sets: {len(self.tensors)}")
            
            return f"MAX GPU Worker {self.worker_id}: {gpu_name} with {final_allocated_gb:.1f}GB ({utilization_percent:.1f}%)"
            
        except Exception as e:
            return f"MAX GPU Worker {self.worker_id} failed: {e}"
    
    def run_maximum_compute_load(self, duration_seconds):
        """Run MAXIMUM GPU compute load for specified duration"""
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        iteration = 0
        
        logger.info(f"🔥🔥🔥 Worker {self.worker_id} ({self.gpu_name}): MAXIMUM GPU LOAD for {duration_seconds}s")
        logger.info(f"         THIS MUST BE VISIBLE IN TASK MANAGER!")
        
        while time.time() < end_time:
            # MAXIMUM INTENSITY: Process ALL tensor sets in parallel
            for tensor_set in self.tensors:
                # 1. Heavy matrix multiplication
                torch.bmm(tensor_set['A'], tensor_set['B'], out=tensor_set['C'])
                
                # 2. Complex activation
                tensor_set['C'] = torch.nn.functional.gelu(tensor_set['C'])
                
                # 3. Another heavy operation
                tensor_set['A'] = torch.bmm(tensor_set['C'], tensor_set['A'].transpose(-2, -1))
                
                # 4. More complex operations
                tensor_set['B'] = torch.nn.functional.softmax(tensor_set['A'], dim=-1)
                tensor_set['C'] = torch.bmm(tensor_set['B'], tensor_set['C'])
                
                # 5. Final intensive operation
                tensor_set['A'] = torch.bmm(tensor_set['C'], tensor_set['B'])
            
            iteration += 1
            
            # Progress every 100 iterations (more frequent)
            if iteration % 100 == 0:
                elapsed = time.time() - start_time
                remaining = duration_seconds - elapsed
                logger.info(f"🔥 Worker {self.worker_id}: {iteration} iterations, {remaining:.1f}s remaining")
            
            # NO SLEEP - MAXIMUM CONTINUOUS LOAD
        
        total_time = time.time() - start_time
        total_operations = iteration * len(self.tensors) * 5  # 5 ops per tensor set
        ops_per_sec = total_operations / total_time
        
        result = {
            "worker_id": self.worker_id,
            "gpu_name": self.gpu_name,
            "duration": total_time,
            "iterations": iteration,
            "total_operations": total_operations,
            "ops_per_second": ops_per_sec,
            "tensor_sets": len(self.tensors),
            "matrix_size": self.matrix_size
        }
        
        logger.info(f"✅ Worker {self.worker_id} MAXIMUM LOAD completed:")
        logger.info(f"   {iteration} iterations, {ops_per_sec:.1f} ops/sec")
        logger.info(f"   {len(self.tensors)} tensor sets processed")
        
        return result

def main():
    duration = 120  # 2 minutes for sustained visibility
    
    logger.info("🔥🔥🔥 MAXIMUM GPU UTILIZATION TEST 🔥🔥🔥")
    logger.info("=" * 80)
    logger.info(f"Duration: {duration} seconds (2 minutes)")
    logger.info("Target: MAXIMUM SUSTAINED GPU utilization on BOTH GPUs")
    logger.info("This WILL be visible in Task Manager GPU usage!")
    logger.info("=" * 80)
    
    try:
        # Connect to Ray cluster
        if not ray.is_initialized():
            ray.init(address='192.168.1.10:6379')
            logger.info("✅ Connected to Ray cluster")
        
        # Clear GPU memory first
        logger.info("🧹 Clearing GPU memory...")
        torch.cuda.empty_cache()
        
        # Create workers with maximum resource allocation
        logger.info("🔧 Creating MAXIMUM GPU workers...")
        
        # Worker 0: RTX 3090 on Head PC - MAXIMUM LOAD
        worker_3090 = MaxGPUWorker.options(
            resources={"node:192.168.1.10": 0.01},
            num_cpus=8,
            num_gpus=0.95  # Use almost all GPU
        ).remote(0, "RTX3090")
        
        # Worker 1: RTX 3070 on Worker PC 2 - MAXIMUM LOAD
        worker_3070 = MaxGPUWorker.options(
            resources={"node:192.168.1.11": 0.01},
            num_cpus=8,
            num_gpus=0.95  # Use almost all GPU
        ).remote(1, "RTX3070")
        
        # Initialize with MAXIMUM GPU allocation
        logger.info("🔧 Initializing MAXIMUM GPU allocation...")
        init_results = ray.get([
            worker_3090.initialize_maximum_gpu_load.remote(),
            worker_3070.initialize_maximum_gpu_load.remote()
        ], timeout=60)
        
        for result in init_results:
            logger.info(f"   {result}")
        
        logger.info("🔥🔥🔥 STARTING MAXIMUM GPU LOAD 🔥🔥🔥")
        logger.info("   ⚡ RTX 3090 (Head PC): MAXIMUM SUSTAINED LOAD")
        logger.info("   ⚡ RTX 3070 (Worker PC 2): MAXIMUM SUSTAINED LOAD")
        logger.info("   📊 CHECK TASK MANAGER NOW - GPU usage should be HIGH!")
        logger.info("=" * 80)
        
        # Start MAXIMUM compute load
        start_time = time.time()
        compute_futures = [
            worker_3090.run_maximum_compute_load.remote(duration),
            worker_3070.run_maximum_compute_load.remote(duration)
        ]
        
        # Monitor progress
        logger.info(f"⏰ Running for {duration} seconds...")
        check_interval = 15  # Check every 15 seconds
        
        while True:
            ready, not_ready = ray.wait(compute_futures, timeout=check_interval, num_returns=len(compute_futures))
            
            if ready:
                # Tasks completed
                break
            else:
                # Still running
                elapsed = time.time() - start_time
                remaining = duration - elapsed
                logger.info(f"⏰ Progress: {elapsed:.0f}s elapsed, {remaining:.0f}s remaining")
                logger.info(f"   📊 GPUs should be showing HIGH usage in Task Manager!")
        
        # Get results
        results = ray.get(compute_futures)
        total_time = time.time() - start_time
        
        # Display results
        logger.info("=" * 80)
        logger.info("🎯 MAXIMUM GPU UTILIZATION RESULTS")
        logger.info("=" * 80)
        logger.info(f"Total time: {total_time:.1f}s")
        
        total_ops = 0
        for result in results:
            gpu_name = result["gpu_name"]
            iterations = result["iterations"]
            ops_per_sec = result["ops_per_second"]
            tensor_sets = result["tensor_sets"]
            matrix_size = result["matrix_size"]
            total_ops += result["total_operations"]
            
            logger.info(f"🎮 {gpu_name}:")
            logger.info(f"   Iterations: {iterations:,}")
            logger.info(f"   Ops/sec: {ops_per_sec:.1f}")
            logger.info(f"   Tensor sets: {tensor_sets}")
            logger.info(f"   Matrix: {matrix_size}x{matrix_size}")
        
        combined_ops_per_sec = total_ops / total_time
        
        logger.info("=" * 80)
        logger.info(f"🔥 COMBINED PERFORMANCE: {combined_ops_per_sec:.1f} ops/sec")
        logger.info(f"🔥 TOTAL OPERATIONS: {total_ops:,}")
        logger.info("=" * 80)
        logger.info("✅ MAXIMUM GPU TEST COMPLETED!")
        logger.info("   If Worker PC 2 GPU didn't show activity, there may be a deeper issue.")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        torch.cuda.empty_cache()
        ray.shutdown()

if __name__ == "__main__":
    main()
