#!/usr/bin/env python3
"""
Test RTX 3090 Smart Compute Optimizer
=====================================
Simple test version to validate the GPU optimization concept works.
"""

import os
import sys
import time
import logging
import ray
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# GPU optimization environment
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

@ray.remote(num_cpus=2, num_gpus=0.5)
class TestGPUWorker:
    """Simple GPU test worker"""
    
    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self.device = None
        self.initialized = False
        
    def initialize_gpu(self):
        """Initialize GPU for testing"""
        try:
            import torch  # Import locally to avoid serialization issues
            
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                torch.cuda.set_device(0)
                
                # Enable optimizations
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cuda.matmul.allow_tf32 = True
                
                # Test tensor
                test_tensor = torch.randn(1024, 1024, device=self.device)
                result = torch.sum(test_tensor).item()
                
                self.initialized = True
                
                gpu_name = torch.cuda.get_device_name(0)
                vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                allocated_gb = torch.cuda.memory_allocated(0) / 1024**3
                
                logger.info(f"🚀 Worker {self.worker_id}: {gpu_name} initialized ({allocated_gb:.1f}GB/{vram_gb:.1f}GB)")
                return f"Worker {self.worker_id} GPU initialized: {result:.4f}"
            else:
                return f"Worker {self.worker_id}: CUDA not available"
        except Exception as e:
            return f"Worker {self.worker_id}: GPU initialization failed: {e}"
    
    def run_compute_test(self, iterations: int = 100):
        """Run simple compute test"""
        if not self.initialized:
            init_result = self.initialize_gpu()
            if "failed" in init_result or "not available" in init_result:
                return {"error": init_result}
        
        try:
            import torch  # Import locally to avoid serialization issues
            
            start_time = time.time()
            total_ops = 0
            
            for i in range(iterations):
                # Create random tensors
                a = torch.randn(512, 512, device=self.device, dtype=torch.float16)
                b = torch.randn(512, 512, device=self.device, dtype=torch.float16)
                
                # Matrix multiplication (tensor cores)
                c = torch.mm(a, b)
                
                # Activation
                c = torch.relu(c)
                
                # More operations
                c = torch.mm(c, a.T)
                
                total_ops += 3  # 2 matmuls + 1 relu
            
            torch.cuda.synchronize()
            operation_time = time.time() - start_time
            
            # Calculate metrics
            allocated_gb = torch.cuda.memory_allocated(0) / 1024**3
            ops_per_second = total_ops / operation_time
            
            return {
                "worker_id": self.worker_id,
                "iterations": iterations,
                "total_operations": total_ops,
                "operation_time": operation_time,
                "ops_per_second": ops_per_second,
                "allocated_gb": allocated_gb,
                "status": "success"
            }
            
        except Exception as e:
            return {"worker_id": self.worker_id, "error": f"Compute test failed: {e}"}

def test_gpu_optimization():
    """Test GPU optimization with Ray"""
    logger.info("🧠 TESTING RTX 3090 SMART COMPUTE OPTIMIZATION")
    logger.info("=" * 60)
    
    try:
        # Connect to Ray
        if not ray.is_initialized():
            ray.init(address='auto', ignore_reinit_error=True)
        
        logger.info("✅ Connected to Ray cluster")
        
        # Check resources
        resources = ray.cluster_resources()
        logger.info(f"📊 Available CPUs: {resources.get('CPU', 0)}")
        logger.info(f"📊 Available GPUs: {resources.get('GPU', 0)}")
        
        # Create test workers
        num_workers = 2
        workers = []
        
        logger.info(f"🔥 Creating {num_workers} test workers...")
        
        for i in range(num_workers):
            worker = TestGPUWorker.remote(i)
            workers.append(worker)
            logger.info(f"✅ Worker {i} created")
        
        # Run compute tests
        logger.info("🚀 Starting compute tests...")
        futures = []
        
        for worker in workers:
            future = worker.run_compute_test.remote(200)  # 200 iterations per worker
            futures.append(future)
        
        # Wait for results
        start_time = time.time()
        results = ray.get(futures, timeout=60)
        total_time = time.time() - start_time
        
        # Display results
        logger.info("📊 TEST RESULTS:")
        logger.info("=" * 40)
        
        successful_workers = [r for r in results if r.get("status") == "success"]
        failed_workers = [r for r in results if r.get("status") != "success"]
        
        total_ops = sum(r.get("total_operations", 0) for r in successful_workers)
        total_ops_per_sec = sum(r.get("ops_per_second", 0) for r in successful_workers)
        
        logger.info(f"Duration: {total_time:.2f} seconds")
        logger.info(f"Successful workers: {len(successful_workers)}/{len(results)}")
        logger.info(f"Total operations: {total_ops:,}")
        logger.info(f"Combined ops/second: {total_ops_per_sec:.1f}")
        
        for result in successful_workers:
            logger.info(f"Worker {result['worker_id']}:")
            logger.info(f"  Iterations: {result['iterations']:,}")
            logger.info(f"  Ops/second: {result['ops_per_second']:.1f}")
            logger.info(f"  VRAM allocated: {result['allocated_gb']:.2f} GB")
        
        for result in failed_workers:
            logger.error(f"❌ Worker {result.get('worker_id', 'Unknown')}: {result.get('error', 'Unknown error')}")
        
        # Save test results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"gpu_optimization_test_{timestamp}.json"
        
        import json
        test_data = {
            "test_type": "GPU Optimization Test",
            "timestamp": timestamp,
            "duration": total_time,
            "total_operations": total_ops,
            "combined_ops_per_second": total_ops_per_sec,
            "workers": len(results),
            "successful_workers": len(successful_workers),
            "worker_results": results
        }
        
        with open(results_file, 'w') as f:
            json.dump(test_data, f, indent=2)
        
        logger.info(f"📊 Test results saved to: {results_file}")
        logger.info("✅ GPU optimization test completed!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        if ray.is_initialized():
            ray.shutdown()

if __name__ == "__main__":
    test_gpu_optimization()
