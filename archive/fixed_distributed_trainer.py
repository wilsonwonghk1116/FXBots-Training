#!/usr/bin/env python3
"""
🚀 FIXED DISTRIBUTED TRAINER
Properly handles GPU device IDs in Ray distributed environment
"""

import ray
import torch
import time
import logging
from ray_conservative_config import get_actor_config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@ray.remote(num_cpus=15, num_gpus=0.8)
class FixedDistributedForexWorker:
    def __init__(self):
        self.device_info = self._setup_local_gpu()
        self.device = self._get_device()
        
    def _setup_local_gpu(self):
        """Setup GPU using LOCAL device IDs only"""
        try:
            # Always use device 0 on the local node
            local_device_id = 0
            
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                
                if device_count > 0:
                    # Set to first available local GPU
                    torch.cuda.set_device(local_device_id)
                    device_name = torch.cuda.get_device_name(local_device_id)
                    memory_total = torch.cuda.get_device_properties(local_device_id).total_memory / 1e9
                    
                    device_info = {
                        'device_id': local_device_id,
                        'device_name': device_name,
                        'memory_total': memory_total,
                        'success': True
                    }
                    
                    logger.info(f"✅ Worker GPU setup: {device_name} (Local Device {local_device_id})")
                    return device_info
                else:
                    logger.error("❌ No CUDA devices found on this node")
                    return {'success': False, 'error': 'No CUDA devices'}
            else:
                logger.error("❌ CUDA not available on this node")
                return {'success': False, 'error': 'CUDA not available'}
                
        except Exception as e:
            logger.error(f"❌ GPU setup failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _get_device(self):
        """Get the proper device string for this worker"""
        if self.device_info.get('success', False):
            return f"cuda:{self.device_info['device_id']}"
        else:
            return "cpu"
    
    def get_worker_info(self):
        """Get worker information including GPU details"""
        return {
            'device_info': self.device_info,
            'device': self.device,
            'node_ip': ray.util.get_node_ip_address()
        }
    
    def train_forex_bot(self, duration_minutes=1):
        """Train forex bot with proper GPU handling"""
        try:
            if not self.device_info.get('success', False):
                logger.warning(f"⚠️  Worker using CPU fallback: {self.device_info.get('error', 'Unknown error')}")
                device = "cpu"
            else:
                device = self.device
                logger.info(f"🎮 Worker using GPU: {self.device_info['device_name']} on {device}")
            
            # Forex bot training simulation
            batch_size = 256
            feature_size = 338  # Your forex features
            hidden_size = 128
            
            iterations = 0
            start_time = time.time()
            end_time = start_time + duration_minutes * 60
            
            # Create model components on the correct device
            while time.time() < end_time:
                try:
                    # Simulate forex data processing
                    market_data = torch.randn(batch_size, feature_size, device=device)
                    
                    # Simple neural network operations
                    weights1 = torch.randn(feature_size, hidden_size, device=device)
                    bias1 = torch.randn(hidden_size, device=device)
                    
                    # Forward pass
                    hidden = torch.relu(torch.mm(market_data, weights1) + bias1)
                    
                    # Output layer (buy/sell/hold decisions)
                    weights2 = torch.randn(hidden_size, 3, device=device)
                    output = torch.softmax(torch.mm(hidden, weights2), dim=1)
                    
                    # Simulate profit calculation
                    profit = output.mean()
                    
                    iterations += 1
                    
                    # Memory cleanup every 10 iterations
                    if iterations % 10 == 0 and device.startswith('cuda'):
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    logger.error(f"❌ Training iteration failed: {e}")
                    # Continue with next iteration
                    continue
            
            actual_duration = time.time() - start_time
            operations = iterations * batch_size
            
            # Get final memory info if using GPU
            memory_info = {}
            if device.startswith('cuda') and self.device_info.get('success', False):
                try:
                    device_id = self.device_info['device_id']
                    memory_allocated = torch.cuda.memory_allocated(device_id) / 1e9
                    memory_total = self.device_info['memory_total']
                    memory_utilization = (memory_allocated / memory_total) * 100
                    
                    memory_info = {
                        'memory_allocated': memory_allocated,
                        'memory_total': memory_total,
                        'memory_utilization': memory_utilization
                    }
                except Exception as e:
                    logger.warning(f"⚠️  Could not get memory info: {e}")
                    memory_info = {'error': str(e)}
            
            result = {
                'worker_info': self.get_worker_info(),
                'iterations': iterations,
                'operations': operations,
                'duration': actual_duration,
                'device_used': device,
                'memory_info': memory_info,
                'success': True
            }
            
            logger.info(f"✅ Training completed: {iterations} iterations, {operations} operations on {device}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            return {
                'worker_info': self.get_worker_info(),
                'error': str(e),
                'success': False
            }

class FixedDistributedTrainer:
    def __init__(self):
        self.config = get_actor_config()
        
    def start_training(self, duration_minutes=1, num_workers=2):
        """Start distributed training with fixed GPU handling"""
        logger.info("🚀 Starting FIXED DISTRIBUTED TRAINER")
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
            
            # Create workers
            logger.info(f"\n🏭 Creating {num_workers} distributed workers...")
            workers = [FixedDistributedForexWorker.remote() for _ in range(num_workers)]
            
            # Get worker info
            logger.info("\n🔍 Getting worker information...")
            worker_info_futures = [worker.get_worker_info.remote() for worker in workers]  # type: ignore
            worker_infos = ray.get(worker_info_futures)
            
            for i, info in enumerate(worker_infos):
                device_info = info['device_info']
                if device_info.get('success', False):
                    logger.info(f"Worker {i+1}: {device_info['device_name']} on {info['node_ip']}")
                else:
                    logger.warning(f"Worker {i+1}: CPU fallback on {info['node_ip']} - {device_info.get('error', 'Unknown')}")
            
            # Start training
            logger.info(f"\n🎯 Starting {duration_minutes}-minute training session...")
            training_futures = [worker.train_forex_bot.remote(duration_minutes) for worker in workers]  # type: ignore
            
            # Monitor progress
            start_time = time.time()
            while training_futures:
                ready, training_futures = ray.wait(training_futures, timeout=10.0)
                
                if ready:
                    elapsed = time.time() - start_time
                    completed = num_workers - len(training_futures)
                    logger.info(f"📊 Progress: {completed}/{num_workers} workers completed ({elapsed:.1f}s elapsed)")
                
                if time.time() - start_time > duration_minutes * 60 + 30:  # 30s buffer
                    logger.warning("⚠️  Training timeout, collecting available results...")
                    break
            
            # Collect all results
            logger.info("\n📥 Collecting final results...")
            all_futures = [worker.train_forex_bot.remote(0) for worker in workers]  # type: ignore
            results = ray.get(all_futures, timeout=30.0)
            
            # Process results
            logger.info("\n📊 TRAINING RESULTS:")
            total_operations = 0
            successful_workers = 0
            gpu_workers = 0
            cpu_workers = 0
            
            for i, result in enumerate(results):
                if result.get('success', False):
                    device_used = result['device_used']
                    device_name = result['worker_info']['device_info'].get('device_name', 'Unknown')
                    node_ip = result['worker_info']['node_ip']
                    
                    logger.info(f"✅ Worker {i+1} ({device_name} on {node_ip}):")
                    logger.info(f"   Device: {device_used}")
                    logger.info(f"   Operations: {result['operations']:,}")
                    logger.info(f"   Duration: {result['duration']:.1f}s")
                    
                    if 'memory_info' in result and 'memory_utilization' in result['memory_info']:
                        logger.info(f"   VRAM: {result['memory_info']['memory_utilization']:.1f}%")
                    
                    total_operations += result['operations']
                    successful_workers += 1
                    
                    if device_used.startswith('cuda'):
                        gpu_workers += 1
                    else:
                        cpu_workers += 1
                else:
                    logger.error(f"❌ Worker {i+1} failed: {result.get('error', 'Unknown error')}")
            
            logger.info(f"\n🎉 SUMMARY:")
            logger.info(f"✅ Successful workers: {successful_workers}/{num_workers}")
            logger.info(f"🎮 GPU workers: {gpu_workers}")
            logger.info(f"💻 CPU workers: {cpu_workers}")
            logger.info(f"🔥 Total operations: {total_operations:,}")
            logger.info(f"⚡ Operations per second: {total_operations/(duration_minutes*60):,.0f}")
            
            # Cleanup
            for worker in workers:
                ray.kill(worker)
            
            return {
                'successful_workers': successful_workers,
                'total_workers': num_workers,
                'gpu_workers': gpu_workers,
                'cpu_workers': cpu_workers,
                'total_operations': total_operations,
                'operations_per_second': total_operations/(duration_minutes*60),
                'success': successful_workers > 0
            }
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            return {'success': False, 'error': str(e)}

if __name__ == "__main__":
    trainer = FixedDistributedTrainer()
    result = trainer.start_training(duration_minutes=1, num_workers=2)
    
    if result.get('success', False):
        print(f"\n🎊 Training successful!")
        print(f"📊 {result['total_operations']:,} operations across {result['successful_workers']} workers")
        print(f"🎮 GPU: {result['gpu_workers']}, CPU: {result['cpu_workers']}")
    else:
        print(f"\n💥 Training failed: {result.get('error', 'Unknown error')}") 