#!/usr/bin/env python3
"""
Test GPU distribution across Ray cluster to ensure both PC1 and PC2 GPUs are utilized
"""

import ray
import torch
import time

def test_gpu_distribution():
    """Test that Ray tasks are properly distributed across both GPUs"""
    print("Testing GPU distribution across Ray cluster...")
    
    # Connect to Ray cluster
    ray.init(address='192.168.1.10:6379')
    
    # Check cluster resources
    cluster_resources = ray.cluster_resources()
    print(f"Cluster resources: {cluster_resources}")
    
    nodes = ray.nodes()
    print(f"Active nodes: {len([n for n in nodes if n['Alive']])}")
    
    @ray.remote(num_gpus=0.5)
    def test_gpu_task(worker_id):
        """Simple GPU test task"""
        import torch
        
        result = {
            'worker_id': worker_id,
            'cuda_available': torch.cuda.is_available(),
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'current_device': None,
            'device_name': None,
            'memory_total': 0,
            'memory_allocated': 0
        }
        
        if torch.cuda.is_available():
            device_id = torch.cuda.current_device()
            result['current_device'] = device_id
            result['device_name'] = torch.cuda.get_device_name(device_id)
            result['memory_total'] = torch.cuda.get_device_properties(device_id).total_memory / 1024**3
            
            # Test actual GPU computation
            try:
                # Set memory fraction to prevent OOM
                torch.cuda.set_per_process_memory_fraction(0.3, device=device_id)
                
                # Create small tensor for testing
                test_tensor = torch.randn(500, 500, device=f'cuda:{device_id}')
                test_result = torch.matmul(test_tensor, test_tensor.t())
                
                result['memory_allocated'] = torch.cuda.memory_allocated(device_id) / 1024**3
                result['computation_success'] = True
                result['test_result_sum'] = float(test_result.sum())
                
                # Cleanup
                del test_tensor, test_result
                torch.cuda.empty_cache()
                
            except Exception as e:
                result['computation_success'] = False
                result['error'] = str(e)
        
        return result
    
    # Launch 4 tasks to test distribution
    print("\nLaunching 4 GPU test tasks...")
    futures = []
    for i in range(4):
        future = test_gpu_task.remote(i)
        futures.append(future)
    
    # Get results
    results = ray.get(futures)
    
    print("\n=== GPU Test Results ===")
    pc1_count = 0
    pc2_count = 0
    
    for result in results:
        print(f"\nWorker {result['worker_id']}:")
        print(f"  CUDA Available: {result['cuda_available']}")
        print(f"  Device Count: {result['device_count']}")
        print(f"  Current Device: {result['current_device']}")
        print(f"  Device Name: {result['device_name']}")
        print(f"  Memory Total: {result['memory_total']:.2f} GB")
        print(f"  Memory Allocated: {result['memory_allocated']:.2f} GB")
        print(f"  Computation Success: {result.get('computation_success', 'N/A')}")
        
        if result.get('error'):
            print(f"  Error: {result['error']}")
        
        # Count GPUs by type
        if result['device_name']:
            if 'RTX 3070' in result['device_name']:
                pc2_count += 1
            else:
                pc1_count += 1
    
    print(f"\n=== Summary ===")
    print(f"Tasks on PC1 GPU: {pc1_count}")
    print(f"Tasks on PC2 GPU (RTX 3070): {pc2_count}")
    
    if pc2_count > 0:
        print("✅ SUCCESS: PC2 RTX 3070 is being utilized!")
    else:
        print("❌ PROBLEM: PC2 RTX 3070 is NOT being utilized!")
    
    if pc1_count > 0 and pc2_count > 0:
        print("🎉 PERFECT: Both GPUs are being utilized for distributed training!")
    
    ray.shutdown()

if __name__ == "__main__":
    test_gpu_distribution()
