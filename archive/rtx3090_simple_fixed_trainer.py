#!/usr/bin/env python3
"""
RTX 3090 Simplified Fixed Trainer
=================================

Fixed version using proven Ray pattern that avoids pickle issues.
"""

import os
import sys
import time
import logging
import argparse
import ray
import torch
from datetime import datetime
from typing import Dict, Optional

# RTX 3090 Environment Setup
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@ray.remote(num_cpus=30, num_gpus=1.0)
class RTX3090SimplifiedWorker:
    """Simplified RTX 3090 worker that avoids pickle issues"""
    
    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self.device = None
        self.initialized = False
        self.iteration_count = 0
        
    def initialize_gpu(self):
        """Initialize GPU after Ray worker creation"""
        if self.initialized:
            return "Already initialized"
            
        try:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
                torch.cuda.set_device(0)
                
                # Enable optimizations
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                
                # Test GPU
                test_tensor = torch.randn(1024, 1024, device=self.device)
                result = torch.sum(test_tensor).item()
                
                self.initialized = True
                
                gpu_name = torch.cuda.get_device_name(0)
                vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                
                logger.info(f"🚀 Worker {self.worker_id}: {gpu_name} ({vram_gb:.1f}GB) initialized")
                return f"GPU initialized: {result:.4f}"
            else:
                return "CUDA not available"
        except Exception as e:
            return f"GPU initialization failed: {e}"
    
    def run_training_iteration(self):
        """Run one training iteration"""
        if not self.initialized:
            init_result = self.initialize_gpu()
            if "failed" in init_result or "not available" in init_result:
                return {"error": init_result}
        
        try:
            # Simulate heavy GPU workload
            matrix_size = 4096
            
            start_time = time.time()
            
            # Mixed precision training simulation
            with torch.cuda.amp.autocast():
                a = torch.randn(matrix_size, matrix_size, device=self.device, dtype=torch.float16)
                b = torch.randn(matrix_size, matrix_size, device=self.device, dtype=torch.float16)
                c = torch.matmul(a, b)
                torch.cuda.synchronize()
            
            operation_time = time.time() - start_time
            self.iteration_count += 1
            
            # Get memory info
            allocated_gb = torch.cuda.memory_allocated(0) / 1024**3
            
            return {
                "worker_id": self.worker_id,
                "iteration": self.iteration_count,
                "operation_time": operation_time,
                "allocated_gb": allocated_gb,
                "matrix_size": matrix_size,
                "status": "success"
            }
            
        except Exception as e:
            return {"error": f"Training iteration failed: {e}"}
    
    def run_training_session(self, duration_minutes: int) -> Dict:
        """Run training session for specified duration"""
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        results = {
            "worker_id": self.worker_id,
            "duration_minutes": duration_minutes,
            "total_iterations": 0,
            "total_time": 0,
            "avg_operation_time": 0,
            "status": "success"
        }
        
        iteration_times = []
        
        logger.info(f"🚀 Worker {self.worker_id}: Starting {duration_minutes}min training session")
        
        while time.time() < end_time:
            iteration_result = self.run_training_iteration()
            
            if "error" in iteration_result:
                results["status"] = "failed"
                results["error"] = iteration_result["error"]
                break
                
            iteration_times.append(iteration_result["operation_time"])
            results["total_iterations"] += 1
            
            if results["total_iterations"] % 100 == 0:
                logger.info(f"   Worker {self.worker_id}: {results['total_iterations']} iterations")
        
        results["total_time"] = time.time() - start_time
        if iteration_times:
            results["avg_operation_time"] = sum(iteration_times) / len(iteration_times)
        
        logger.info(f"✅ Worker {self.worker_id}: Session complete - {results['total_iterations']} iterations")
        return results

class RTX3090SimplifiedTrainer:
    """Simplified RTX 3090 trainer"""
    
    def __init__(self):
        self.num_workers = 1
        
    def run_training(self, duration_minutes: int):
        """Run simplified training session"""
        logger.info("🚀 RTX 3090 SIMPLIFIED TRAINER")
        logger.info("=" * 50)
        logger.info(f"Duration: {duration_minutes} minutes")
        logger.info(f"Workers: {self.num_workers}")
        
        # Connect to Ray
        try:
            ray.init(ignore_reinit_error=True)
            logger.info("✅ Connected to Ray cluster")
        except Exception as e:
            logger.error(f"❌ Ray connection failed: {e}")
            return
        
        # Create workers
        logger.info(f"🔥 Creating {self.num_workers} RTX 3090 workers...")
        workers = []
        
        try:
            for i in range(self.num_workers):
                worker = RTX3090SimplifiedWorker.remote(i)
                workers.append(worker)
                logger.info(f"✅ Worker {i} created")
            
            # Start training
            logger.info("🚀 Starting RTX 3090 training session...")
            start_time = time.time()
            
            # Run training sessions
            futures = []
            for worker in workers:
                future = worker.run_training_session.remote(duration_minutes)
                futures.append(future)
            
            # Wait for completion
            results = ray.get(futures)
            
            total_time = time.time() - start_time
            
            # Process results
            total_iterations = sum(r.get("total_iterations", 0) for r in results)
            successful_workers = sum(1 for r in results if r.get("status") == "success")
            
            logger.info("=" * 50)
            logger.info("🎯 TRAINING SESSION SUMMARY")
            logger.info("=" * 50)
            logger.info(f"Duration: {total_time:.1f} seconds")
            logger.info(f"Workers: {successful_workers}/{len(workers)} successful")
            logger.info(f"Total iterations: {total_iterations}")
            logger.info(f"Iterations/second: {total_iterations/total_time:.2f}")
            
            for i, result in enumerate(results):
                if result.get("status") == "success":
                    logger.info(f"Worker {i}: {result['total_iterations']} iterations, "
                              f"avg: {result['avg_operation_time']:.4f}s")
                else:
                    logger.error(f"Worker {i}: {result.get('error', 'Unknown error')}")
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"rtx3090_simplified_results_{timestamp}.json"
            
            import json
            with open(results_file, 'w') as f:
                json.dump({
                    "summary": {
                        "total_time": total_time,
                        "total_iterations": total_iterations,
                        "successful_workers": successful_workers,
                        "iterations_per_second": total_iterations/total_time
                    },
                    "worker_results": results
                }, f, indent=2)
            
            logger.info(f"📊 Results saved to: {results_file}")
            logger.info("=" * 50)
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
        
        finally:
            ray.shutdown()
            logger.info("🔗 Ray disconnected")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='RTX 3090 Simplified Trainer')
    parser.add_argument('--duration', type=int, default=1, 
                       help='Training duration in minutes (default: 1)')
    args = parser.parse_args()
    
    trainer = RTX3090SimplifiedTrainer()
    trainer.run_training(args.duration)

if __name__ == "__main__":
    main() 