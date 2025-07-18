#!/usr/bin/env python3
"""
Simple Ray + CUDA test to verify both PCs can access CUDA through Ray
"""
import ray
import torch
import socket
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@ray.remote(num_cpus=1, num_gpus=0.5)
class CudaTestWorker:
    def __init__(self, worker_id):
        self.worker_id = worker_id
    
    def test_cuda(self):
        """Test CUDA availability on this worker"""
        try:
            import torch
            import socket
            
            hostname = socket.gethostname()
            
            # Basic info
            result = {
                "worker_id": self.worker_id,
                "hostname": hostname,
                "cuda_available": torch.cuda.is_available(),
                "gpu_count": 0,
                "gpu_name": "None",
                "torch_version": torch.__version__,
                "cuda_version": "None"
            }
            
            if torch.cuda.is_available():
                result["gpu_count"] = torch.cuda.device_count()
                result["gpu_name"] = torch.cuda.get_device_name(0)
                result["cuda_version"] = torch.version.cuda
                
                # Test simple GPU operation
                try:
                    device = torch.device("cuda:0")
                    test_tensor = torch.randn(100, 100, device=device)
                    test_result = torch.mm(test_tensor, test_tensor)
                    result["gpu_test"] = "SUCCESS"
                    result["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
                except Exception as e:
                    result["gpu_test"] = f"FAILED: {e}"
            else:
                result["gpu_test"] = "CUDA_NOT_AVAILABLE"
            
            return result
            
        except Exception as e:
            return {
                "worker_id": self.worker_id,
                "error": str(e),
                "hostname": socket.gethostname()
            }

def main():
    logger.info("🧪 RAY + CUDA DUAL-PC TEST")
    logger.info("=" * 50)
    
    try:
        # Connect to existing Ray cluster
        if not ray.is_initialized():
            ray.init(address='192.168.1.10:6379')
            logger.info("✅ Connected to Ray cluster")
        
        # Check cluster resources
        cluster_resources = ray.cluster_resources()
        logger.info(f"📊 Cluster: {cluster_resources.get('GPU', 0)} GPUs, {cluster_resources.get('CPU', 0)} CPUs")
        
        # Create workers with explicit node placement
        logger.info("🔧 Creating CUDA test workers...")
        
        # Worker 0: Force on Head PC 
        worker_0 = CudaTestWorker.options(
            resources={"node:192.168.1.10": 0.01}
        ).remote(0)
        
        # Worker 1: Force on Worker PC 2
        worker_1 = CudaTestWorker.options(
            resources={"node:192.168.1.11": 0.01}
        ).remote(1)
        
        # Test CUDA on both workers
        logger.info("🧪 Testing CUDA on both workers...")
        test_futures = [
            worker_0.test_cuda.remote(),
            worker_1.test_cuda.remote()
        ]
        
        results = ray.get(test_futures, timeout=30)
        
        logger.info("📊 CUDA TEST RESULTS:")
        logger.info("=" * 50)
        
        for result in results:
            if "error" in result:
                logger.error(f"❌ Worker {result['worker_id']} ({result['hostname']}): {result['error']}")
            else:
                status = "✅" if result["cuda_available"] else "❌"
                logger.info(f"{status} Worker {result['worker_id']} ({result['hostname']}):")
                logger.info(f"   CUDA Available: {result['cuda_available']}")
                logger.info(f"   GPU Count: {result['gpu_count']}")
                logger.info(f"   GPU Name: {result['gpu_name']}")
                logger.info(f"   PyTorch: {result['torch_version']}")
                logger.info(f"   CUDA Version: {result['cuda_version']}")
                logger.info(f"   GPU Test: {result['gpu_test']}")
                if "gpu_memory_gb" in result:
                    logger.info(f"   GPU Memory: {result['gpu_memory_gb']:.1f}GB")
        
        # Check if Worker PC 2 has CUDA issues
        worker_2_result = results[1] if len(results) > 1 else None
        if worker_2_result and not worker_2_result.get("cuda_available", False):
            logger.error("🚨 WORKER PC 2 CUDA ISSUE DETECTED!")
            logger.error("   This explains why GPU utilization is 0% on Worker PC 2")
            logger.error("   Possible solutions:")
            logger.error("   1. Check if PyTorch with CUDA is installed on Worker PC 2")
            logger.error("   2. Verify NVIDIA drivers on Worker PC 2")
            logger.error("   3. Check CUDA_VISIBLE_DEVICES environment variable")
            logger.error("   4. Restart Ray cluster with proper environment")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
    finally:
        ray.shutdown()

if __name__ == "__main__":
    main()
