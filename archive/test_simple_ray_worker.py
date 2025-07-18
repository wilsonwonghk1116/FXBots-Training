#!/usr/bin/env python3
"""
Simple Ray worker test to isolate pickle issues
"""

import ray
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@ray.remote(num_cpus=1, num_gpus=0.1)
class SimpleTestWorker:
    """Minimal test worker"""
    
    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self.device = None
        self.initialized = False
    
    def initialize_cuda(self):
        """Initialize CUDA after Ray worker creation"""
        if self.initialized:
            return "Already initialized"
            
        try:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
                torch.cuda.set_device(0)
                temp_tensor = torch.randn(100, 100, device=self.device)
                result = torch.sum(temp_tensor).item()
                self.initialized = True
                return f"CUDA initialized: {result:.4f}"
            else:
                return "CUDA not available"
        except Exception as e:
            return f"CUDA initialization failed: {e}"
    
    def run_test(self) -> str:
        """Run simple test"""
        cuda_result = self.initialize_cuda()
        return f"Worker {self.worker_id}: {cuda_result}"

def main():
    """Test simple Ray worker"""
    try:
        # Initialize Ray
        ray.init(ignore_reinit_error=True)
        logger.info("✅ Ray initialized")
        
        # Test worker creation
        logger.info("Creating simple test worker...")
        worker = SimpleTestWorker.remote(0)
        logger.info("✅ Worker created successfully")
        
        # Test worker execution
        result = ray.get(worker.run_test.remote())
        logger.info(f"✅ Worker test result: {result}")
        
        # Cleanup
        ray.shutdown()
        logger.info("✅ Test completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False
        
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 