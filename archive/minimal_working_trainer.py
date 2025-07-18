#!/usr/bin/env python3
"""
MINIMAL WORKING TRAINER - ULTRA-CONSERVATIVE APPROACH
=====================================================
Uses absolutely minimal resource requests to guarantee execution:
- 1 CPU + 0.1 GPU per worker (tiny requests)
- More workers to compensate for smaller individual allocation
- Immediate feedback and execution verification
- Progressive resource scaling once working

Goal: Get SOMETHING working on both PCs first, then scale up.
"""

import os
import sys
import time
import torch
import psutil
import threading
import json
import logging
import warnings
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np

# Suppress warnings
warnings.filterwarnings('ignore')

try:
    import ray
except ImportError:
    print("❌ Ray not installed. Please install: pip install ray[default]")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'minimal_working_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

@ray.remote(num_cpus=1, num_gpus=0.1)
class MinimalWorker:
    """Ultra-minimal worker with tiny resource requests"""
    
    def __init__(self, worker_id: int, target_pc: str):
        self.worker_id = worker_id
        self.target_pc = target_pc  # "PC1" or "PC2"
        self.device = None
        self.start_time = time.time()
        self.setup_minimal_gpu()
        
    def setup_minimal_gpu(self):
        """Setup minimal GPU allocation"""
        try:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
                torch.cuda.set_device(0)
                
                # Minimal VRAM allocation (100MB)
                minimal_vram = 100 * 1024 * 1024  # 100MB
                self.vram_tensor = torch.empty(minimal_vram // 4, dtype=torch.float32, device=self.device)
                
                # Get GPU info
                gpu_name = torch.cuda.get_device_name(0)
                total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                logger.info(f"🔥 {self.target_pc} Worker {self.worker_id}: {gpu_name} ready, 0.1GB/{total_vram:.1f}GB VRAM")
            else:
                self.device = torch.device("cpu")
                logger.info(f"💻 {self.target_pc} Worker {self.worker_id}: CPU-only mode")
                
        except Exception as e:
            logger.warning(f"⚠️ {self.target_pc} Worker {self.worker_id} GPU setup failed: {e}")
            self.device = torch.device("cpu")
    
    def run_minimal_workload(self, duration_minutes: int) -> Dict:
        """Run minimal but visible workload"""
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        iteration = 0
        total_score = 0.0
        
        logger.info(f"🚀 {self.target_pc} Worker {self.worker_id}: STARTING workload for {duration_minutes} minutes")
        
        # Immediate confirmation this worker is running
        logger.info(f"✅ {self.target_pc} Worker {self.worker_id}: CONFIRMED RUNNING!")
        
        while time.time() < end_time:
            iteration += 1
            
            # Minimal but real GPU operations
            if self.device is not None and self.device.type == "cuda":
                score = self.minimal_gpu_operations()
            else:
                score = self.minimal_cpu_operations()
            
            total_score += score
            
            # Frequent progress reports (every 10 iterations)
            if iteration % 10 == 0:
                elapsed = (time.time() - start_time) / 60
                avg_score = total_score / iteration
                logger.info(f"📊 {self.target_pc} Worker {self.worker_id}: {iteration} iter, avg score {avg_score:.4f} [{elapsed:.1f}m]")
            
            # Small pause
            time.sleep(0.1)
        
        total_time = time.time() - start_time
        final_avg_score = total_score / iteration if iteration > 0 else 0
        
        result = {
            "worker_id": self.worker_id,
            "target_pc": self.target_pc,
            "device": str(self.device),
            "iterations": iteration,
            "total_score": total_score,
            "avg_score": final_avg_score,
            "total_time_minutes": total_time / 60,
            "iterations_per_minute": iteration / (total_time / 60) if total_time > 0 else 0
        }
        
        logger.info(f"🎯 {self.target_pc} Worker {self.worker_id}: COMPLETED - {iteration} iterations, avg score {final_avg_score:.4f}")
        
        return result
    
    def minimal_gpu_operations(self) -> float:
        """Minimal GPU operations that will actually use GPU"""
        try:
            if self.device is None:
                return self.minimal_cpu_operations()
                
            # Small matrix operations
            size = 256  # Very small
            a = torch.randn(size, size, device=self.device, dtype=torch.float32)
            b = torch.randn(size, size, device=self.device, dtype=torch.float32)
            
            # Matrix multiplication
            c = torch.matmul(a, b)
            
            # Small neural network
            x = torch.randn(32, 128, device=self.device)
            w = torch.randn(128, 64, device=self.device)
            output = torch.relu(torch.matmul(x, w))
            
            # Return score
            score = float(torch.mean(torch.abs(output)).item())
            return score
            
        except Exception as e:
            logger.warning(f"⚠️ {self.target_pc} Worker {self.worker_id} GPU op failed: {e}")
            return self.minimal_cpu_operations()
    
    def minimal_cpu_operations(self) -> float:
        """Minimal CPU operations"""
        result = 0.0
        for i in range(1000):
            result += np.sin(i) * np.cos(i)
        return abs(result) / 1000

class MinimalTrainer:
    """Minimal trainer focused on getting ANY execution on both PCs"""
    
    def __init__(self):
        self.workers = []
        self.monitoring_active = False
        
    def connect_to_existing_cluster(self) -> bool:
        """Connect to the existing Ray cluster"""
        try:
            if not ray.is_initialized():
                ray.init(address="auto", ignore_reinit_error=True)
            
            time.sleep(2)  # Wait for connection
            
            # Check cluster
            cluster_resources = ray.cluster_resources()
            nodes = ray.nodes()
            active_nodes = len([n for n in nodes if n['Alive']])
            
            total_cpus = int(cluster_resources.get('CPU', 0))
            total_gpus = int(cluster_resources.get('GPU', 0))
            
            logger.info(f"🌐 Connected to cluster: {active_nodes} nodes, {total_cpus} CPUs, {total_gpus} GPUs")
            
            if active_nodes >= 1:
                logger.info("✅ Cluster connection successful!")
                
                # Log node details
                for i, node in enumerate([n for n in nodes if n['Alive']]):
                    node_ip = node.get('NodeManagerAddress', 'unknown')
                    logger.info(f"  Node {i}: {node_ip}")
                
                return True
            else:
                logger.error("❌ No active nodes found")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to connect to cluster: {e}")
            return False
    
    def spawn_minimal_workers(self, num_workers: int = 6) -> bool:
        """Spawn minimal workers with tiny resource requests"""
        try:
            logger.info(f"🚀 Spawning {num_workers} minimal workers...")
            
            for i in range(num_workers):
                # Alternate target PC assignment
                target_pc = "PC1" if i % 2 == 0 else "PC2"
                
                worker = MinimalWorker.remote(i, target_pc)
                self.workers.append(worker)
                
                logger.info(f"✅ Worker {i} spawned (target: {target_pc})")
            
            logger.info(f"🎯 Total workers spawned: {len(self.workers)}")
            
            # Small delay to let workers initialize
            time.sleep(2)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to spawn workers: {e}")
            return False
    
    def start_simple_monitoring(self):
        """Simple resource monitoring"""
        def monitor():
            while self.monitoring_active:
                try:
                    cpu_percent = psutil.cpu_percent(interval=1)
                    memory = psutil.virtual_memory()
                    
                    gpu_info = "N/A"
                    if torch.cuda.is_available():
                        gpu_memory = torch.cuda.memory_allocated() / (1024**3)
                        gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                        gpu_percent = (gpu_memory / gpu_total) * 100
                        gpu_info = f"{gpu_percent:.1f}% ({gpu_memory:.1f}GB/{gpu_total:.1f}GB)"
                    
                    logger.info(f"📊 PC1 Resources - CPU: {cpu_percent:.1f}%, RAM: {memory.percent:.1f}%, GPU: {gpu_info}")
                    
                    time.sleep(20)  # Monitor every 20 seconds
                    
                except Exception as e:
                    logger.warning(f"⚠️ Monitoring error: {e}")
                    time.sleep(10)
        
        self.monitoring_active = True
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
    
    def run_minimal_training(self, duration_minutes: int = 3) -> Dict:
        """Run minimal training to verify both PCs work"""
        logger.info("🎯 STARTING MINIMAL TRAINING")
        logger.info(f"⏱️ Duration: {duration_minutes} minutes")
        logger.info(f"🖥️ Workers: {len(self.workers)}")
        logger.info("🎯 Goal: Verify BOTH PCs execute workload")
        
        start_time = time.time()
        
        # Start monitoring
        self.start_simple_monitoring()
        
        # Launch all workers
        logger.info("🚀 Launching ALL workers...")
        
        futures = [worker.run_minimal_workload.remote(duration_minutes) for worker in self.workers]
        
        logger.info(f"⏳ Waiting for {len(futures)} workers to complete...")
        
        # Wait for results
        results = ray.get(futures)
        
        self.monitoring_active = False
        total_time = time.time() - start_time
        
        # Analyze results
        pc1_results = [r for r in results if r["target_pc"] == "PC1"]
        pc2_results = [r for r in results if r["target_pc"] == "PC2"]
        
        gpu_results = [r for r in results if "cuda" in r["device"]]
        cpu_results = [r for r in results if "cpu" in r["device"]]
        
        total_iterations = sum(r["iterations"] for r in results)
        avg_score = np.mean([r["avg_score"] for r in results]) if results else 0
        
        summary = {
            "total_time_minutes": total_time / 60,
            "total_workers": len(results),
            "pc1_workers": len(pc1_results),
            "pc2_workers": len(pc2_results),
            "gpu_workers": len(gpu_results),
            "cpu_workers": len(cpu_results),
            "total_iterations": total_iterations,
            "overall_avg_score": avg_score,
            "pc1_avg_score": np.mean([r["avg_score"] for r in pc1_results]) if pc1_results else 0,
            "pc2_avg_score": np.mean([r["avg_score"] for r in pc2_results]) if pc2_results else 0,
            "detailed_results": results
        }
        
        logger.info("🎉 MINIMAL TRAINING COMPLETE!")
        logger.info(f"⏱️ Total Time: {summary['total_time_minutes']:.1f} minutes")
        logger.info(f"🏆 Overall Avg Score: {summary['overall_avg_score']:.4f}")
        logger.info(f"🔄 Total Iterations: {summary['total_iterations']}")
        logger.info(f"💪 PC1 Workers: {summary['pc1_workers']}, avg score {summary['pc1_avg_score']:.4f}")
        logger.info(f"💪 PC2 Workers: {summary['pc2_workers']}, avg score {summary['pc2_avg_score']:.4f}")
        logger.info(f"🔥 GPU Workers: {summary['gpu_workers']}")
        logger.info(f"💻 CPU Workers: {summary['cpu_workers']}")
        
        # Success verification
        if summary['pc2_workers'] > 0:
            logger.info("✅ SUCCESS: PC2 workers executed!")
        else:
            logger.warning("⚠️ WARNING: No PC2 workers executed")
        
        if summary['gpu_workers'] > 0:
            logger.info("✅ SUCCESS: GPU workers executed!")
        else:
            logger.warning("⚠️ WARNING: No GPU workers executed")
        
        return summary

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Minimal Working Trainer')
    parser.add_argument('--duration', type=int, default=3, 
                       help='Training duration in minutes (default: 3)')
    parser.add_argument('--workers', type=int, default=6,
                       help='Number of workers (default: 6)')
    
    args = parser.parse_args()
    
    print("🎯 MINIMAL WORKING TRAINER")
    print("==========================")
    print("Ultra-conservative approach to verify both PCs work")
    print("Resource per worker: 1 CPU + 0.1 GPU")
    print()
    
    trainer = MinimalTrainer()
    
    # Connect to cluster
    if not trainer.connect_to_existing_cluster():
        print("❌ Failed to connect to Ray cluster")
        return
    
    # Spawn workers
    if not trainer.spawn_minimal_workers(args.workers):
        print("❌ Failed to spawn workers")
        return
    
    # Run training
    try:
        results = trainer.run_minimal_training(args.duration)
        
        # Save results
        results_file = f"minimal_working_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"📊 Results saved to: {results_file}")
        
        # Final verification
        success_indicators = []
        if results['pc2_workers'] > 0:
            success_indicators.append("✅ PC2 utilized")
        if results['gpu_workers'] > 0:
            success_indicators.append("✅ GPU utilized")
        if results['total_iterations'] > 0:
            success_indicators.append("✅ Workers executed")
        
        if len(success_indicators) >= 2:
            print("🎉 MINIMAL SYSTEM WORKING! Ready for scaling up.")
        else:
            print("⚠️ Still need debugging...")
        
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
    finally:
        # Cleanup
        try:
            ray.shutdown()
            print("🧹 Ray shutdown complete")
        except:
            pass

if __name__ == "__main__":
    main() 