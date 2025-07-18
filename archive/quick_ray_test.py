#!/usr/bin/env python3
"""
Quick Ray Cluster Test
Test Ray cluster connectivity and basic functionality before running the full simulation

Usage: python quick_ray_test.py
"""

import ray
import torch
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@ray.remote(num_cpus=1, num_gpus=0.1)
class TestActor:
    def __init__(self, actor_id):
        self.actor_id = actor_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def test_computation(self):
        """Test basic computation with GPU if available"""
        start_time = time.time()
        
        # Test basic computation
        if self.device.type == 'cuda':
            # GPU test
            x = torch.randn(1000, 1000, device=self.device)
            y = torch.matmul(x, x)
            result = y.sum().item()
        else:
            # CPU test
            import numpy as np
            x = np.random.randn(1000, 1000)
            y = np.matmul(x, x)
            result = y.sum()
        
        compute_time = time.time() - start_time
        
        return {
            'actor_id': self.actor_id,
            'device': str(self.device),
            'result': float(result),
            'compute_time': compute_time
        }

def main():
    print("RAY CLUSTER TEST - Connectivity and Basic Functionality")
    print("="*60)
    
    try:
        # Connect to Ray cluster
        if not ray.is_initialized():
            ray.init(address='auto')
        
        # Display cluster info
        resources = ray.cluster_resources()
        nodes = ray.nodes()
        
        print(f"✓ Connected to Ray Cluster")
        print(f"  - CPUs: {resources.get('CPU', 0)}")
        print(f"  - GPUs: {resources.get('GPU', 0)}")
        print(f"  - Memory: {resources.get('memory', 0) / 1e9:.1f} GB")
        print(f"  - Nodes: {len(nodes)}")
        print()
        
        # Display node details
        for i, node in enumerate(nodes):
            if node['Alive']:
                print(f"  Node {i+1}: {node['NodeManagerAddress']}:{node['NodeManagerPort']}")
                print(f"    - CPU: {node['Resources'].get('CPU', 0)}")
                print(f"    - GPU: {node['Resources'].get('GPU', 0)}")
                print(f"    - Memory: {node['Resources'].get('memory', 0) / 1e9:.1f} GB")
        print()
        
        # Test actor creation and execution
        print("Testing actor creation and execution...")
        
        # Create test actors
        n_actors = 10
        actors = []
        for i in range(n_actors):
            actor = TestActor.remote(actor_id=i)
            actors.append(actor)
        
        print(f"✓ Created {n_actors} test actors")
        
        # Run parallel computation test
        print("Running parallel computation test...")
        futures = [actor.test_computation.remote() for actor in actors]
        
        start_time = time.time()
        results = ray.get(futures)
        total_time = time.time() - start_time
        
        print(f"✓ Parallel computation completed in {total_time:.2f} seconds")
        
        # Analyze results
        cpu_actors = [r for r in results if 'cpu' in r['device'].lower()]
        gpu_actors = [r for r in results if 'cuda' in r['device'].lower()]
        
        print(f"  - CPU actors: {len(cpu_actors)}")
        print(f"  - GPU actors: {len(gpu_actors)}")
        
        if cpu_actors:
            avg_cpu_time = sum(r['compute_time'] for r in cpu_actors) / len(cpu_actors)
            print(f"  - Average CPU compute time: {avg_cpu_time:.3f}s")
        
        if gpu_actors:
            avg_gpu_time = sum(r['compute_time'] for r in gpu_actors) / len(gpu_actors)
            print(f"  - Average GPU compute time: {avg_gpu_time:.3f}s")
        
        print()
        print("✓ Ray Cluster Test PASSED")
        print("  Ready for full Kelly Monte Carlo simulation!")
        
        return True
        
    except Exception as e:
        print(f"✗ Ray Cluster Test FAILED: {e}")
        return False
        
    finally:
        if ray.is_initialized():
            ray.shutdown()

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
