#!/usr/bin/env python3
"""
🔧 DISTRIBUTED GPU DETECTION FIX
Properly handle CUDA device IDs in Ray distributed environment
"""

import torch
import ray
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@ray.remote(num_gpus=1)
class FixedDistributedWorker:
    def __init__(self):
        self.device_info = self._setup_local_gpu()
        
    def _setup_local_gpu(self):
        """Setup GPU using LOCAL device IDs only"""
        try:
            # Always use device 0 on the local node
            local_device_id = 0
            
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                logger.info(f"🎮 Local CUDA devices available: {device_count}")
                
                if device_count > 0:
                    # Set to first available local GPU
                    torch.cuda.set_device(local_device_id)
                    device_name = torch.cuda.get_device_name(local_device_id)
                    
                    # Get memory info using proper local device
                    memory_allocated = torch.cuda.memory_allocated(local_device_id) / 1e9
                    memory_reserved = torch.cuda.memory_reserved(local_device_id) / 1e9
                    memory_total = torch.cuda.get_device_properties(local_device_id).total_memory / 1e9
                    
                    device_info = {
                        'device_id': local_device_id,
                        'device_name': device_name,
                        'memory_allocated': memory_allocated,
                        'memory_reserved': memory_reserved, 
                        'memory_total': memory_total,
                        'memory_free': memory_total - memory_allocated,
                        'utilization': (memory_allocated / memory_total) * 100
                    }
                    
                    logger.info(f"✅ GPU setup successful: {device_name}")
                    logger.info(f"   Device ID: {local_device_id}")
                    logger.info(f"   Memory: {memory_allocated:.2f}GB/{memory_total:.2f}GB")
                    
                    return device_info
                else:
                    logger.error("❌ No CUDA devices found on this node")
                    return None
            else:
                logger.error("❌ CUDA not available on this node")
                return None
                
        except Exception as e:
            logger.error(f"❌ GPU setup failed: {e}")
            return None
    
    def get_device_info(self):
        """Get current device information"""
        return self.device_info
    
    def run_gpu_test(self, duration_seconds=10):
        """Run a simple GPU test using local device"""
        if not self.device_info:
            return {"error": "No GPU available"}
        
        try:
            device_id = self.device_info['device_id']
            device = f"cuda:{device_id}"
            
            logger.info(f"🧪 Testing GPU {device_id} ({self.device_info['device_name']})")
            
            # Simple GPU computation
            batch_size = 512
            iterations = 0
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()  # type: ignore
            
            import time
            end_test = time.time() + duration_seconds
            
            while time.time() < end_test:
                # Create tensors on the correct local device
                x = torch.randn(batch_size, 256, device=device)
                y = torch.nn.functional.relu(x)
                z = y.mean()
                
                iterations += 1
                
                # Cleanup every 10 iterations
                if iterations % 10 == 0:
                    torch.cuda.empty_cache()
            
            end_time.record()  # type: ignore
            torch.cuda.synchronize()
            
            # Get updated memory info
            memory_allocated = torch.cuda.memory_allocated(device_id) / 1e9
            memory_reserved = torch.cuda.memory_reserved(device_id) / 1e9
            utilization = (memory_allocated / self.device_info['memory_total']) * 100
            
            result = {
                'device_name': self.device_info['device_name'],
                'iterations': iterations,
                'operations': iterations * batch_size,
                'duration': duration_seconds,
                'memory_allocated': memory_allocated,
                'memory_utilization': utilization,
                'success': True
            }
            
            logger.info(f"✅ GPU test completed: {iterations} iterations, {result['operations']} operations")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ GPU test failed: {e}")
            return {"error": str(e), "success": False}

def test_distributed_gpu_fix():
    """Test the distributed GPU fix"""
    logger.info("🧪 TESTING DISTRIBUTED GPU FIX")
    logger.info("=" * 60)
    
    try:
        # Connect to Ray
        if not ray.is_initialized():
            ray.init(address='auto')
        
        # Get cluster info
        cluster_resources = ray.cluster_resources()
        available_resources = ray.available_resources()
        
        logger.info(f"📊 Cluster resources: {cluster_resources}")
        logger.info(f"💡 Available resources: {available_resources}")
        
        # Create workers on different nodes
        logger.info("\n🚀 Creating distributed workers...")
        
        # Create 2 workers - they should automatically distribute
        workers = [FixedDistributedWorker.remote() for _ in range(2)]
        
        logger.info(f"✅ Created {len(workers)} distributed workers")
        
        # Get device info from each worker
        logger.info("\n🔍 Getting device information from each worker...")
        device_info_futures = [worker.get_device_info.remote() for worker in workers]  # type: ignore
        device_infos = ray.get(device_info_futures)
        
        for i, info in enumerate(device_infos):
            if info:
                logger.info(f"Worker {i+1}: {info['device_name']} (Device {info['device_id']})")
                logger.info(f"   Memory: {info['memory_allocated']:.2f}GB/{info['memory_total']:.2f}GB")
            else:
                logger.error(f"Worker {i+1}: No GPU info available")
        
        # Run GPU tests
        logger.info("\n🧪 Running GPU tests on each worker...")
        test_futures = [worker.run_gpu_test.remote(10) for worker in workers]  # type: ignore
        test_results = ray.get(test_futures)
        
        logger.info("\n📊 Test Results:")
        total_operations = 0
        successful_workers = 0
        
        for i, result in enumerate(test_results):
            if result.get('success', False):
                logger.info(f"✅ Worker {i+1} ({result['device_name']}): {result['iterations']} iterations, {result['operations']} operations")
                total_operations += result['operations']
                successful_workers += 1
            else:
                logger.error(f"❌ Worker {i+1} failed: {result.get('error', 'Unknown error')}")
        
        logger.info(f"\n🎉 SUMMARY:")
        logger.info(f"✅ Successful workers: {successful_workers}/{len(workers)}")
        logger.info(f"🔥 Total operations: {total_operations:,}")
        
        if successful_workers == len(workers):
            logger.info("🎊 ALL WORKERS SUCCESSFUL - GPU distribution fixed!")
            return True
        else:
            logger.error("💥 Some workers failed - needs further investigation")
            return False
        
        # Cleanup
        for worker in workers:
            ray.kill(worker)
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False
    finally:
        if ray.is_initialized():
            ray.shutdown()

if __name__ == "__main__":
    success = test_distributed_gpu_fix()
    if success:
        print("\n🎉 GPU distribution fix successful!")
        print("💡 You can now use this pattern in your training code")
    else:
        print("\n💥 GPU distribution fix failed!")
        print("🔍 Check the logs for details") 