#!/usr/bin/env python3
"""
🔧 FIXED RTX 3070 TRAINER CONFIGURATION
Conservative resource allocation to prevent Ray conflicts
"""

import ray
from ray_conservative_config import get_conservative_resources, get_actor_config

# Import the original trainer but with fixed configuration
class FixedRTX3070Trainer:
    def __init__(self):
        self.config = get_actor_config()
        
    def start_training(self, duration_minutes=1):
        """Start training with fixed resource allocation"""
        print("🚀 Starting FIXED RTX 3070 Trainer")
        print("=" * 50)
        
        # Conservative resource allocation
        actor_config = self.config
        
        try:
            # Connect to Ray
            if not ray.is_initialized():
                ray.init(address='auto')
            
            print(f"📊 Using conservative config: {actor_config}")
            
            # Create exactly 2 actors with reduced resources
            @ray.remote(
                num_cpus=actor_config['cpu_per_actor'],
                num_gpus=actor_config['gpu_per_actor']
            )
            class ConservativeWorker:
                def train(self, duration):
                    import time
                    import torch
                    
                    # Smaller batch size for reduced memory
                    batch_size = 256  # Reduced from 512
                    iterations = 0
                    
                    start_time = time.time()
                    end_time = start_time + duration * 60
                    
                    while time.time() < end_time:
                        # Simulate smaller training step
                        x = torch.randn(batch_size, 128).cuda()
                        y = torch.nn.functional.relu(x)
                        loss = y.mean()
                        
                        iterations += 1
                        
                        # Memory cleanup every 10 iterations
                        if iterations % 10 == 0:
                            torch.cuda.empty_cache()
                    
                    return {
                        "iterations": iterations,
                        "duration": time.time() - start_time,
                        "operations": iterations * batch_size
                    }
            
            # Create only 2 workers to prevent resource conflicts
            workers = [ConservativeWorker.remote() for _ in range(2)]
            print(f"✅ Created {len(workers)} conservative workers")
            
            # Start training
            futures = [worker.train.remote(duration_minutes) for worker in workers]  # type: ignore
            
            # Monitor with simple progress
            print(f"🎯 Training for {duration_minutes} minute(s)...")
            
            # Use ray.wait for non-blocking monitoring
            remaining = futures[:]
            completed_count = 0
            
            while remaining:
                ready, remaining = ray.wait(remaining, timeout=10.0)
                
                for completed_future in ready:
                    completed_count += 1
                    print(f"✅ Worker {completed_count}/2 completed")
            
            # Collect results
            results = ray.get(futures)
            
            print("🎉 TRAINING COMPLETE!")
            for i, result in enumerate(results):
                print(f"📊 Worker {i+1}: {result['iterations']} iterations, {result['operations']} operations")
            
            return results
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return None
        finally:
            # Clean up workers
            try:
                for worker in workers:
                    ray.kill(worker)
            except:
                pass

if __name__ == "__main__":
    trainer = FixedRTX3070Trainer()
    trainer.start_training(1)  # 1 minute test 