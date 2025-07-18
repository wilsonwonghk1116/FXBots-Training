#!/usr/bin/env python3
"""
Comprehensive test of VRAM distribution across both PC1 and PC2 GPUs
This simulates the actual workload that was causing the OOM error
"""

import ray
import torch
import time
import numpy as np

def comprehensive_vram_test():
    """Test VRAM usage distribution to prevent OOM errors"""
    print("🔍 Testing VRAM distribution across PC1 and PC2...")
    
    # Connect to Ray cluster
    ray.init(address='192.168.1.10:6379')
    
    @ray.remote(num_gpus=0.5)  # Use same allocation as fixed training
    def vram_intensive_task(worker_id, iterations=100):
        """Simulate the actual training workload that was causing OOM"""
        import torch
        import gc
        import time
        
        results = []
        print(f"Worker {worker_id}: Starting VRAM intensive test...")
        
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_id = torch.cuda.current_device()
            device = f'cuda:{device_id}'
            
            # Set conservative memory fraction (40% like in the fix)
            torch.cuda.set_per_process_memory_fraction(0.4, device=device_id)
            
            gpu_name = torch.cuda.get_device_name(device_id)
            total_memory = torch.cuda.get_device_properties(device_id).total_memory / 1024**3
            
            print(f"Worker {worker_id}: Using {gpu_name} ({total_memory:.1f} GB total)")
            
            for i in range(iterations):
                try:
                    # Create tensors similar to training workload (but smaller for testing)
                    tensor_size = 1200  # Same size as in the fixed code
                    
                    # Create GPU tensors
                    gpu_data = torch.randn(tensor_size, tensor_size, device=device, dtype=torch.float32)
                    gpu_result = torch.matmul(gpu_data, gpu_data.t())
                    trading_signal = torch.sigmoid(gpu_result.mean())
                    
                    # Monitor memory usage
                    memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
                    memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
                    
                    results.append({
                        'worker_id': worker_id,
                        'iteration': i,
                        'gpu_name': gpu_name,
                        'memory_allocated_gb': memory_allocated,
                        'memory_reserved_gb': memory_reserved,
                        'signal': float(trading_signal.cpu()),
                        'success': True
                    })
                    
                    # Immediate cleanup
                    del gpu_data, gpu_result, trading_signal
                    torch.cuda.empty_cache()
                    
                    # Report every 20 iterations
                    if i % 20 == 0:
                        print(f"Worker {worker_id}: Iteration {i}, Memory: {memory_allocated:.2f} GB allocated, {memory_reserved:.2f} GB reserved")
                    
                    time.sleep(0.01)  # Brief pause
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"Worker {worker_id}: OOM at iteration {i} - This indicates a problem!")
                        results.append({
                            'worker_id': worker_id,
                            'iteration': i,
                            'gpu_name': gpu_name,
                            'error': 'OOM',
                            'success': False
                        })
                        torch.cuda.empty_cache()
                        gc.collect()
                        break
                    else:
                        raise e
        else:
            print(f"Worker {worker_id}: No CUDA available")
            return [{'worker_id': worker_id, 'error': 'No CUDA', 'success': False}]
        
        print(f"Worker {worker_id}: Completed {len(results)} iterations successfully")
        return results
    
    # Launch 4 workers to test both GPUs (2 per GPU)
    print("🚀 Launching 4 workers to test VRAM distribution...")
    futures = []
    for i in range(4):
        future = vram_intensive_task.remote(i, 50)  # 50 iterations per worker
        futures.append(future)
    
    # Get results
    all_results = ray.get(futures)
    
    # Analyze results
    print("\n=== VRAM Test Results ===")
    pc1_workers = []
    pc2_workers = []
    total_success = 0
    total_oom = 0
    
    for worker_results in all_results:
        for result in worker_results:
            if result.get('success'):
                total_success += 1
            elif result.get('error') == 'OOM':
                total_oom += 1
                print(f"❌ Worker {result['worker_id']}: OOM ERROR detected!")
            
            # Categorize by GPU
            if 'gpu_name' in result:
                if 'RTX 3070' in result['gpu_name']:
                    pc2_workers.append(result['worker_id'])
                else:
                    pc1_workers.append(result['worker_id'])
    
    # Remove duplicates
    pc1_workers = list(set(pc1_workers))
    pc2_workers = list(set(pc2_workers))
    
    print(f"\n📊 Distribution Summary:")
    print(f"PC1 workers: {pc1_workers} ({len(pc1_workers)} workers)")
    print(f"PC2 RTX 3070 workers: {pc2_workers} ({len(pc2_workers)} workers)")
    print(f"Total successful operations: {total_success}")
    print(f"Total OOM errors: {total_oom}")
    
    if total_oom == 0:
        print("✅ SUCCESS: No OOM errors detected!")
    else:
        print(f"❌ PROBLEM: {total_oom} OOM errors detected!")
    
    if len(pc2_workers) > 0:
        print("✅ SUCCESS: PC2 RTX 3070 is being utilized!")
    else:
        print("❌ PROBLEM: PC2 RTX 3070 is NOT being utilized!")
    
    if len(pc1_workers) > 0 and len(pc2_workers) > 0 and total_oom == 0:
        print("🎉 PERFECT: VRAM distribution working correctly across both GPUs!")
        print("🚀 Ready for full training system!")
    else:
        print("⚠️ Issues detected - may need further tuning")
    
    ray.shutdown()

if __name__ == "__main__":
    comprehensive_vram_test()
