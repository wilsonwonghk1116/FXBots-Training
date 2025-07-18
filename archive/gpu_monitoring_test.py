#!/usr/bin/env python3
"""
GPU MONITORING TEST with nvidia-smi
==================================
This will run GPU workload while monitoring with nvidia-smi
"""
import ray
import torch
import time
import logging
import subprocess
import threading
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@ray.remote(num_cpus=4, num_gpus=0.7)
class MonitoredGPUWorker:
    def __init__(self, worker_id):
        self.worker_id = worker_id
        
    def run_gpu_with_monitoring(self, duration_seconds):
        """Run GPU workload while monitoring with nvidia-smi"""
        try:
            import torch
            import socket
            import subprocess
            import time
            
            hostname = socket.gethostname()
            device = torch.device("cuda:0")
            gpu_name = torch.cuda.get_device_name(0)
            
            # Allocate moderate GPU memory
            matrix_size = 2048
            batch_size = 8
            
            tensor_a = torch.randn(batch_size, matrix_size, matrix_size, device=device, dtype=torch.float16)
            tensor_b = torch.randn(batch_size, matrix_size, matrix_size, device=device, dtype=torch.float16)
            result_tensor = torch.empty(batch_size, matrix_size, matrix_size, device=device, dtype=torch.float16)
            
            allocated_gb = torch.cuda.memory_allocated(0) / 1024**3
            
            logger.info(f"🎮 Worker {self.worker_id} on {hostname} ({gpu_name})")
            logger.info(f"   Allocated: {allocated_gb:.1f}GB")
            
            # Run nvidia-smi to check current status
            try:
                nvidia_cmd = ["nvidia-smi", "--query-gpu=name,memory.used,utilization.gpu", "--format=csv,noheader,nounits"]
                result = subprocess.run(nvidia_cmd, capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    logger.info(f"   nvidia-smi output: {result.stdout.strip()}")
                else:
                    logger.warning(f"   nvidia-smi failed: {result.stderr}")
            except Exception as e:
                logger.warning(f"   nvidia-smi error: {e}")
            
            # Run compute workload
            start_time = time.time()
            end_time = start_time + duration_seconds
            iteration = 0
            
            logger.info(f"🔥 Starting {duration_seconds}s GPU workload on {hostname}...")
            
            while time.time() < end_time:
                # Intensive operations
                torch.bmm(tensor_a, tensor_b, out=result_tensor)
                result_tensor = torch.nn.functional.gelu(result_tensor)
                tensor_a = torch.bmm(result_tensor, tensor_a.transpose(-2, -1))
                torch.cuda.synchronize()
                
                iteration += 1
                
                # Check nvidia-smi every 500 iterations
                if iteration % 500 == 0:
                    try:
                        result = subprocess.run(nvidia_cmd, capture_output=True, text=True, timeout=5)
                        if result.returncode == 0:
                            logger.info(f"   Iter {iteration}: {result.stdout.strip()}")
                    except:
                        pass
                
                time.sleep(0.01)  # 10ms pause for monitoring visibility
            
            total_time = time.time() - start_time
            
            # Final nvidia-smi check
            try:
                result = subprocess.run(nvidia_cmd, capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    logger.info(f"   Final nvidia-smi: {result.stdout.strip()}")
            except Exception as e:
                logger.warning(f"   Final nvidia-smi error: {e}")
            
            return {
                "worker_id": self.worker_id,
                "hostname": hostname,
                "gpu_name": gpu_name,
                "duration": total_time,
                "iterations": iteration,
                "allocated_gb": allocated_gb,
                "ops_per_second": iteration / total_time
            }
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            logger.error(f"Worker {self.worker_id} failed: {error_details}")
            return {"worker_id": self.worker_id, "error": str(e)}

def main():
    duration = 60  # 1 minute test
    
    logger.info("🔍 GPU MONITORING TEST with nvidia-smi")
    logger.info("=" * 60)
    logger.info("This will show nvidia-smi output during GPU workload")
    logger.info("=" * 60)
    
    try:
        # Connect to Ray cluster
        if not ray.is_initialized():
            ray.init(address='192.168.1.10:6379')
            logger.info("✅ Connected to Ray cluster")
        
        # Create workers
        worker_3090 = MonitoredGPUWorker.options(
            resources={"node:192.168.1.10": 0.01}
        ).remote(0)
        
        worker_3070 = MonitoredGPUWorker.options(
            resources={"node:192.168.1.11": 0.01}
        ).remote(1)
        
        logger.info("🔥 Starting GPU workload with nvidia-smi monitoring...")
        logger.info("   Watch for GPU utilization % and memory usage changes")
        
        # Run workload
        futures = [
            worker_3090.run_gpu_with_monitoring.remote(duration),
            worker_3070.run_gpu_with_monitoring.remote(duration)
        ]
        
        results = ray.get(futures, timeout=duration + 30)
        
        logger.info("=" * 60)
        logger.info("🎯 GPU MONITORING RESULTS")
        logger.info("=" * 60)
        
        for result in results:
            if "error" in result:
                logger.error(f"❌ Worker {result['worker_id']}: {result['error']}")
            else:
                logger.info(f"✅ Worker {result['worker_id']} ({result['hostname']}):")
                logger.info(f"   GPU: {result['gpu_name']}")
                logger.info(f"   Duration: {result['duration']:.1f}s")
                logger.info(f"   Iterations: {result['iterations']}")
                logger.info(f"   VRAM: {result['allocated_gb']:.1f}GB")
                logger.info(f"   Ops/sec: {result['ops_per_second']:.1f}")
        
        logger.info("=" * 60)
        logger.info("✅ MONITORING TEST COMPLETED")
        logger.info("Check the nvidia-smi outputs above for GPU utilization")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        ray.shutdown()

if __name__ == "__main__":
    main()
