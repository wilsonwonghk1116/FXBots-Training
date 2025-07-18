#!/usr/bin/env python3
"""
SUSTAINED GPU UTILIZATION - BALANCED LOAD
=========================================
This will create steady, sustained GPU load without hitting memory limits
"""
import ray
import torch
import time
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@ray.remote(num_cpus=6, num_gpus=0.8)
class SustainedGPUWorker:
    def __init__(self, worker_id, gpu_name):
        self.worker_id = worker_id
        self.gpu_name = gpu_name
        
    def initialize_sustained_gpu_load(self):
        """Initialize GPU with sustained, safe workload"""
        try:
            import torch
            import socket
            
            device = torch.device("cuda:0")
            torch.cuda.set_device(0)
            
            hostname = socket.gethostname()
            gpu_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            # SAFE BUT SUBSTANTIAL MEMORY ALLOCATION
            if "3090" in gpu_name:
                # RTX 3090: Use ~50% memory for safety
                self.matrix_size = 4096
                self.batch_size = 12
                num_tensors = 6
                target_percent = 50
            else:
                # RTX 3070: Use ~60% memory for visibility
                self.matrix_size = 3072
                self.batch_size = 10  
                num_tensors = 5
                target_percent = 60
            
            logger.info(f"🔥 Worker {self.worker_id}: Allocating {target_percent}% GPU memory...")
            logger.info(f"   Target: {target_percent}% of {total_memory:.1f}GB on {gpu_name}")
            
            # Allocate tensor sets
            self.tensors = []
            for i in range(num_tensors):
                tensor_set = {
                    'A': torch.randn(self.batch_size, self.matrix_size, self.matrix_size, 
                                   device=device, dtype=torch.float16),
                    'B': torch.randn(self.batch_size, self.matrix_size, self.matrix_size, 
                                   device=device, dtype=torch.float16),
                    'C': torch.empty(self.batch_size, self.matrix_size, self.matrix_size, 
                                   device=device, dtype=torch.float16),
                    'workspace': torch.empty(self.batch_size, self.matrix_size, self.matrix_size, 
                                           device=device, dtype=torch.float16)
                }
                self.tensors.append(tensor_set)
                
                # Force allocation
                for tensor in tensor_set.values():
                    tensor.zero_()
                
                torch.cuda.synchronize()
                
                allocated_gb = torch.cuda.memory_allocated(0) / 1024**3
                percent_used = allocated_gb / total_memory * 100
                
                logger.info(f"   Tensor set {i+1}: {allocated_gb:.1f}GB ({percent_used:.1f}%)")
            
            final_allocated_gb = torch.cuda.memory_allocated(0) / 1024**3
            final_percent = final_allocated_gb / total_memory * 100
            
            logger.info(f"✅ Worker {self.worker_id}: {gpu_name}")
            logger.info(f"   VRAM: {final_allocated_gb:.1f}GB / {total_memory:.1f}GB ({final_percent:.1f}%)")
            logger.info(f"   Matrix: {self.matrix_size}x{self.matrix_size}")
            logger.info(f"   Tensor sets: {len(self.tensors)}")
            
            return f"Worker {self.worker_id}: {gpu_name} ready with {final_allocated_gb:.1f}GB ({final_percent:.1f}%)"
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            logger.error(f"Worker {self.worker_id} initialization failed: {error_details}")
            return f"Worker {self.worker_id} failed: {e}"
    
    def run_sustained_compute(self, duration_seconds):
        """Run sustained GPU compute with consistent load"""
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        iteration = 0
        
        logger.info(f"🔥 Worker {self.worker_id} ({self.gpu_name}): SUSTAINED LOAD for {duration_seconds}s")
        logger.info(f"   This should be clearly visible in Task Manager!")
        
        # Main compute loop
        while time.time() < end_time:
            # Process each tensor set with intensive operations
            for i, tensor_set in enumerate(self.tensors):
                # 1. Matrix multiplication (uses tensor cores)
                torch.bmm(tensor_set['A'], tensor_set['B'], out=tensor_set['C'])
                
                # 2. Activation function (compute intensive)
                tensor_set['workspace'] = torch.nn.functional.gelu(tensor_set['C'])
                
                # 3. Transpose and multiply (more intensive)
                tensor_set['C'] = torch.bmm(tensor_set['workspace'], tensor_set['A'].transpose(-2, -1))
                
                # 4. Softmax (memory and compute intensive)
                tensor_set['workspace'] = torch.nn.functional.softmax(tensor_set['C'], dim=-1)
                
                # 5. Final matrix operation
                tensor_set['A'] = torch.bmm(tensor_set['workspace'], tensor_set['B'])
                
                # Small sleep between tensor sets to create sustained load pattern
                # This helps with visibility in Task Manager
                time.sleep(0.001)  # 1ms pause
            
            iteration += 1
            
            # Progress every 50 iterations
            if iteration % 50 == 0:
                elapsed = time.time() - start_time
                remaining = duration_seconds - elapsed
                progress_percent = elapsed / duration_seconds * 100
                logger.info(f"   Worker {self.worker_id}: {iteration} iterations ({progress_percent:.1f}%), {remaining:.1f}s left")
            
            # Brief pause between iterations for sustained pattern
            time.sleep(0.005)  # 5ms pause for sustainability
        
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
        
        logger.info(f"✅ Worker {self.worker_id} sustained load completed:")
        logger.info(f"   {iteration} iterations, {ops_per_sec:.1f} ops/sec")
        
        return result

def main():
    duration = 180  # 3 minutes for extended visibility
    
    logger.info("🔥 SUSTAINED GPU UTILIZATION TEST 🔥")
    logger.info("=" * 70)
    logger.info(f"Duration: {duration} seconds (3 minutes)")
    logger.info("Target: SUSTAINED, VISIBLE GPU utilization")
    logger.info("Safe memory allocation to avoid OOM errors")
    logger.info("=" * 70)
    
    try:
        # Connect to Ray cluster
        if not ray.is_initialized():
            ray.init(address='192.168.1.10:6379')
            logger.info("✅ Connected to Ray cluster")
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        
        # Create workers
        logger.info("🔧 Creating sustained GPU workers...")
        
        worker_3090 = SustainedGPUWorker.options(
            resources={"node:192.168.1.10": 0.01}
        ).remote(0, "RTX3090")
        
        worker_3070 = SustainedGPUWorker.options(
            resources={"node:192.168.1.11": 0.01}
        ).remote(1, "RTX3070")
        
        # Initialize workers
        logger.info("🔧 Initializing sustained GPU load...")
        init_results = ray.get([
            worker_3090.initialize_sustained_gpu_load.remote(),
            worker_3070.initialize_sustained_gpu_load.remote()
        ], timeout=60)
        
        for result in init_results:
            logger.info(f"   {result}")
        
        logger.info("🔥 STARTING SUSTAINED GPU LOAD 🔥")
        logger.info("   📊 MONITOR TASK MANAGER on Worker PC 2 now!")
        logger.info("   📊 GPU 0 should show consistent utilization")
        logger.info("   📊 Dedicated GPU memory should show significant usage")
        logger.info("=" * 70)
        
        # Start sustained load
        start_time = time.time()
        compute_futures = [
            worker_3090.run_sustained_compute.remote(duration),
            worker_3070.run_sustained_compute.remote(duration)
        ]
        
        # Monitor progress with regular updates
        logger.info("⏰ Starting sustained load monitoring...")
        
        # Check every 30 seconds
        check_interval = 30
        
        while True:
            ready, not_ready = ray.wait(compute_futures, timeout=check_interval, num_returns=len(compute_futures))
            
            if ready:
                break
            else:
                elapsed = time.time() - start_time
                remaining = duration - elapsed
                progress = elapsed / duration * 100
                logger.info(f"⏰ Progress: {elapsed:.0f}s ({progress:.1f}%), {remaining:.0f}s remaining")
                logger.info(f"   📊 Check Task Manager: GPU usage should be visible!")
        
        # Get results
        results = ray.get(compute_futures)
        total_time = time.time() - start_time
        
        # Display results
        logger.info("=" * 70)
        logger.info("🎯 SUSTAINED GPU UTILIZATION RESULTS")
        logger.info("=" * 70)
        logger.info(f"Total duration: {total_time:.1f}s")
        
        total_ops = 0
        for result in results:
            gpu_name = result["gpu_name"]
            iterations = result["iterations"]
            ops_per_sec = result["ops_per_second"]
            matrix_size = result["matrix_size"]
            total_ops += result["total_operations"]
            
            logger.info(f"🎮 {gpu_name}:")
            logger.info(f"   Iterations: {iterations:,}")
            logger.info(f"   Ops/sec: {ops_per_sec:.1f}")
            logger.info(f"   Matrix: {matrix_size}x{matrix_size}")
        
        logger.info("=" * 70)
        logger.info("✅ SUSTAINED GPU TEST COMPLETED!")
        logger.info("   Did you see GPU activity on Worker PC 2?")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        torch.cuda.empty_cache()
        ray.shutdown()

if __name__ == "__main__":
    main()
