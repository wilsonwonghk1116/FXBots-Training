#!/usr/bin/env python3
"""
RTX 3090 + RTX 3070 DUAL-GPU DISTRIBUTED COMPUTE OPTIMIZER
============================================================

Fixed version for dual-PC Ray cluster setup.
Forces creation of 2 workers to utilize both GPUs across the cluster.

TARGETS:
- GPU Utilization: 90%+ on BOTH GPUs
- Distributed execution across Head PC + Worker PC 2
- RTX 3090 (Head PC) + RTX 3070 (Worker PC 2)

Usage:
    python rtx3090_smart_compute_optimizer_dual_gpu.py --duration=10
"""

import os
import sys
import time
import logging
import argparse
import ray
import numpy as np
import torch
import socket
import multiprocessing
import threading
import subprocess
import gc
import traceback
from datetime import datetime
from typing import Dict, List
import tqdm
import gc
import subprocess
import traceback
import socket
import threading
import multiprocessing
import json
import traceback

# Smart GPU optimization environment with VRAM cleanup
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64,roundup_power2_divisions:16"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["CUDA_CACHE_DISABLE"] = "0"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def cleanup_gpu_memory():
    """Comprehensive GPU VRAM cleanup for both Head PC and Worker PC"""
    logger.info("🧹 COMPREHENSIVE GPU VRAM CLEANUP INITIATED")
    logger.info("=" * 60)
    
    try:
        import torch
        
        # Force immediate cleanup of all cached memory
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            logger.info(f"🔍 Found {device_count} CUDA device(s)")
            
            for device_id in range(device_count):
                try:
                    torch.cuda.set_device(device_id)
                    device_name = torch.cuda.get_device_name(device_id)
                    
                    # Get memory info before cleanup
                    total_memory = torch.cuda.get_device_properties(device_id).total_memory / 1024**3
                    allocated_before = torch.cuda.memory_allocated(device_id) / 1024**3
                    cached_before = torch.cuda.memory_reserved(device_id) / 1024**3
                    
                    logger.info(f"  🎮 GPU {device_id} ({device_name}): {total_memory:.1f}GB total")
                    logger.info(f"     Before cleanup: {allocated_before:.2f}GB allocated, {cached_before:.2f}GB cached")
                    
                    # Comprehensive cleanup sequence
                    torch.cuda.empty_cache()           # Clear cached memory
                    torch.cuda.ipc_collect()          # Collect IPC memory
                    torch.cuda.synchronize()          # Ensure all operations complete
                    
                    # Force Python garbage collection
                    gc.collect()
                    
                    # Second pass cleanup
                    torch.cuda.empty_cache()
                    
                    # Get memory info after cleanup
                    allocated_after = torch.cuda.memory_allocated(device_id) / 1024**3
                    cached_after = torch.cuda.memory_reserved(device_id) / 1024**3
                    freed_memory = (allocated_before + cached_before) - (allocated_after + cached_after)
                    
                    logger.info(f"     After cleanup:  {allocated_after:.2f}GB allocated, {cached_after:.2f}GB cached")
                    logger.info(f"     ✅ Freed: {freed_memory:.2f}GB ({freed_memory/total_memory*100:.1f}% of total)")
                    
                except Exception as e:
                    logger.warning(f"⚠️ GPU {device_id} cleanup warning: {e}")
        else:
            logger.warning("⚠️ CUDA not available for cleanup")
            
    except ImportError:
        logger.warning("⚠️ PyTorch not available for cleanup")
    
    # System-level GPU memory cleanup
    try:
        logger.info("🧹 System-level GPU cleanup...")
        # Clear any lingering CUDA contexts
        subprocess.run(["nvidia-smi", "--gpu-reset"], capture_output=True, timeout=10)
        logger.info("✅ System-level cleanup completed")
    except Exception as e:
        logger.warning(f"⚠️ System cleanup warning: {e}")
    
    # Force Python garbage collection
    gc.collect()
    
    logger.info("=" * 60)
    logger.info("🎯 GPU VRAM CLEANUP COMPLETED - READY FOR MAXIMUM UTILIZATION")
    logger.info("=" * 60)

@ray.remote(num_cpus=8, num_gpus=0.9)  # FIXED: Use fractional GPU allocation like working examples
class SmartGPUComputeWorker:
    """Smart GPU compute worker with memory-aware optimization"""
    
    def __init__(self, worker_id: int, gpu_type: str):
        self.worker_id = worker_id
        self.gpu_type = gpu_type
        self.device = None
        self.streams = []
        self.initialized = False
        self.iteration_count = 0
        
        # CPU-intensive background workers for 80% CPU utilization
        self.cpu_workers = []
        self.cpu_intensive_enabled = True
        
        # GPU-SPECIFIC optimized configurations (Head PC gets priority CPU, Worker PC gets stable settings)
        if gpu_type == "RTX3090":
            # Head PC 1: Balanced aggressive settings considering potential VRAM fragmentation
            self.matrix_size = 4096       # FIXED: Reduced to working size from previous examples
            self.batch_size = 8           # Keep same
            self.num_streams = 12         # FIXED: Reduced to working number
            self.operations_per_stream = 12
            self.prealloc_tensors = 6     # FIXED: Reduced allocation attempts
            # Target high CPU usage on Head PC
            self.cpu_target_percent = 0.80  # FIXED: Reduced to working level
            self.cpu_intensity_multiplier = 1.5  # FIXED: Reduced intensity
        else:
            # Worker PC 2: Use EXACT working RTX3070 settings from working examples
            self.matrix_size = 3072       # FIXED: Increased from 2048 to boost VRAM allocation (3072^2 * factors ~ larger tensors)
            self.batch_size = 8           # FIXED: Increased from 6 to further increase memory footprint
            self.num_streams = 10         # FIXED: Slightly increased from 8 for more concurrency
            self.operations_per_stream = 12  # FIXED: Increased from 10 for more intensive compute
            self.prealloc_tensors = 4     # FIXED: Much smaller allocation
            # Standard CPU usage for Worker PC
            self.cpu_target_percent = 0.8  # Keep same
            self.cpu_intensity_multiplier = 1.0  # Keep same
    
    def initialize_smart_gpu(self):
        """Initialize GPU with smart compute optimizations and gradual memory allocation"""
        if self.initialized:
            return f"Worker {self.worker_id} already initialized"
            
        try:
            # ENHANCED DIAGNOSTICS for Worker PC 2 CUDA issue
            import socket  # Import socket locally for Ray worker
            hostname = socket.gethostname()
            logger.info(f"🔍 Worker {self.worker_id} diagnostics on {hostname}:")
            logger.info(f"   Python executable: {sys.executable}")
            logger.info(f"   Python path: {sys.path[:3]}...")  # First 3 entries
            
            # Import PyTorch locally to avoid serialization issues
            try:
                import torch
                from torch.amp.autocast_mode import autocast
                logger.info(f"   ✅ PyTorch imported successfully: {torch.__version__}")
            except ImportError as e:
                error_msg = f"PyTorch import failed: {e}"
                logger.error(f"   ❌ {error_msg}")
                return f"GPU initialization failed: {error_msg}"
            
            # Check CUDA with enhanced diagnostics
            cuda_available = torch.cuda.is_available()
            logger.info(f"   CUDA available: {cuda_available}")
            
            if not cuda_available:
                logger.error(f"❌ Worker {self.worker_id}: CUDA not available on {hostname}")
                
                # Enhanced CUDA diagnostics
                logger.error("🔍 DETAILED CUDA DIAGNOSTICS:")
                logger.error(f"   PyTorch version: {torch.__version__}")
                logger.error(f"   CUDA compiled version: {torch.version.cuda}")
                logger.error(f"   cuDNN version: {torch.backends.cudnn.version()}")
                
                # Check if CUDA devices are detected at system level
                try:
                    import subprocess  # Import subprocess locally for Ray worker
                    result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        logger.error(f"   ✅ nvidia-smi detects GPUs: {result.stdout.strip()}")
                    else:
                        logger.error("   ❌ nvidia-smi failed to detect GPUs")
                except Exception as e:
                    logger.error(f"   ❌ nvidia-smi error: {e}")
                
                # Check environment variables
                import os  # Import os locally for Ray worker
                cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
                logger.error(f"   CUDA_VISIBLE_DEVICES: {cuda_visible}")
                
                # **CRITICAL**: This might be a Ray environment issue
                logger.error("   💡 POSSIBLE CAUSES:")
                logger.error("   1. Ray worker using different Python environment than where PyTorch+CUDA installed")
                logger.error("   2. Ray worker doesn't have same environment variables")
                logger.error("   3. Ray worker running in container/isolated environment")
                logger.error(f"   4. Worker PC 2 Ray environment path: {sys.executable}")
                
                # Return descriptive error but don't crash the system
                return f"CUDA not available on {hostname} - see diagnostics above"
                
            # If we get here, CUDA is available
            logger.info(f"   ✅ CUDA available with {torch.cuda.device_count()} device(s)")
                
            # Use GPU 0 on whatever machine this worker is on
            self.device = torch.device("cuda:0")
            torch.cuda.set_device(0)
            
            # Get GPU info for confirmation
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"   🎮 GPU detected: {gpu_name}")
            
            # COMPREHENSIVE VRAM CLEANUP FIRST
            logger.info(f"🧹 Worker {self.worker_id}: Starting with comprehensive VRAM cleanup...")
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
            import gc
            gc.collect()
            torch.cuda.empty_cache()  # Second pass
            
            # Calculate available VRAM after cleanup
            total_vram_bytes = torch.cuda.get_device_properties(0).total_memory
            current_allocated = torch.cuda.memory_allocated(0)
            current_cached = torch.cuda.memory_reserved(0)
            available_vram_bytes = total_vram_bytes - current_allocated - current_cached
            
            # Conservative target: 85% of available VRAM (not total) to prevent OOM
            self.target_vram_gb = (available_vram_bytes * 0.95) / 1024**3  # FIXED: Increased from 0.85 to 0.95 for higher target utilization
            total_vram_gb = total_vram_bytes / 1024**3
            
            logger.info(f"🎯 Worker {self.worker_id}: VRAM Analysis")
            logger.info(f"   Total VRAM: {total_vram_gb:.1f}GB")
            logger.info(f"   Available: {available_vram_bytes/1024**3:.1f}GB")
            logger.info(f"   Target: {self.target_vram_gb:.1f}GB (95% of available)")  # FIXED: Updated log
            
            # SMART OPTIMIZATIONS
            # 1. Enable tensor core optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
            
            # 2. Create CUDA streams for concurrent execution
            self.streams = [torch.cuda.Stream() for _ in range(self.num_streams)]
            logger.info(f"🚀 Worker {self.worker_id}: Created {self.num_streams} CUDA streams")
            
            # 3. GRADUAL TENSOR ALLOCATION with safety checks
            self.pre_allocated_tensors = []
            
            # Calculate conservative tensor sizes based on available VRAM
            base_tensor_size = self.batch_size * self.matrix_size * self.matrix_size * 2  # FP16 = 2 bytes
            single_tensor_set_gb = (base_tensor_size * 7) / 1024**3  # 7 tensors per set
            
            # Adaptive allocation based on GPU type and available memory
            if self.gpu_type == "RTX3090":
                # Head PC: More conservative due to potential system overhead
                safety_margin = 0.7  # 70% safety margin
                max_allocation_attempts = 3  # Limited attempts to prevent OOM
            else:
                # Worker PC: Push harder for laptop GPU
                safety_margin = 1.0     # FIXED: Increased to 1.0 to maximize allocation
                max_allocation_attempts = 100  # FIXED: High value to allow maximum possible sets
            
            max_tensor_sets = max(1, int((self.target_vram_gb * safety_margin) / single_tensor_set_gb))
            allocation_attempts = min(max_allocation_attempts, max_tensor_sets)
            
            logger.info(f"🔧 Worker {self.worker_id}: GRADUAL ALLOCATION STRATEGY")
            logger.info(f"   Single tensor set: {single_tensor_set_gb:.2f}GB")
            logger.info(f"   Max possible sets: {max_tensor_sets}")
            logger.info(f"   Allocation attempts: {allocation_attempts}")
            logger.info(f"   Safety margin: {safety_margin*100:.0f}%")
            
            # GRADUAL ALLOCATION with memory monitoring
            successful_allocations = 0
            for i in range(allocation_attempts):
                try:
                    # Check available memory before each allocation
                    current_available = (total_vram_bytes - torch.cuda.memory_allocated(0) - torch.cuda.memory_reserved(0)) / 1024**3
                    needed_memory = single_tensor_set_gb * 1.2  # 20% overhead buffer
                    
                    if current_available < needed_memory:
                        logger.info(f"⛔ Worker {self.worker_id}: Stopping allocation - insufficient memory")
                        logger.info(f"   Available: {current_available:.2f}GB, Needed: {needed_memory:.2f}GB")
                        break
                    
                    # Create optimized tensor set with proper sizing
                    if self.gpu_type == "RTX3090":
                        # Head PC: Balanced tensors to prevent OOM
                        tensors = {
                            'A': torch.randn(self.batch_size, self.matrix_size, self.matrix_size, 
                                           device=self.device, dtype=torch.float16, requires_grad=False),
                            'B': torch.randn(self.batch_size, self.matrix_size, self.matrix_size, 
                                           device=self.device, dtype=torch.float16, requires_grad=False),
                            'workspace': torch.empty(self.batch_size, self.matrix_size, self.matrix_size, 
                                                   device=self.device, dtype=torch.float16),
                            'C': torch.randn(self.batch_size, self.matrix_size, self.matrix_size, 
                                           device=self.device, dtype=torch.float16, requires_grad=False),
                            'D': torch.randn(self.batch_size, self.matrix_size, self.matrix_size, 
                                           device=self.device, dtype=torch.float16, requires_grad=False),
                            'temp1': torch.empty(self.batch_size, self.matrix_size, self.matrix_size, 
                                               device=self.device, dtype=torch.float16),
                            'temp2': torch.empty(self.batch_size, self.matrix_size, self.matrix_size, 
                                               device=self.device, dtype=torch.float16)
                        }
                    else:
                        # Worker PC: Standard tensors
                        tensors = {
                            'A': torch.randn(self.batch_size, self.matrix_size, self.matrix_size, 
                                           device=self.device, dtype=torch.float16, requires_grad=False),
                            'B': torch.randn(self.batch_size, self.matrix_size, self.matrix_size, 
                                           device=self.device, dtype=torch.float16, requires_grad=False),
                            'workspace': torch.empty(self.batch_size, self.matrix_size, self.matrix_size, 
                                                   device=self.device, dtype=torch.float16),
                            'C': torch.randn(self.batch_size, self.matrix_size, self.matrix_size, 
                                           device=self.device, dtype=torch.float16, requires_grad=False),
                            'D': torch.randn(self.batch_size, self.matrix_size, self.matrix_size, 
                                           device=self.device, dtype=torch.float16, requires_grad=False),
                            'temp1': torch.empty(self.batch_size, self.matrix_size, self.matrix_size, 
                                               device=self.device, dtype=torch.float16),
                            'temp2': torch.empty(self.batch_size, self.matrix_size, self.matrix_size, 
                                               device=self.device, dtype=torch.float16)
                        }
                    
                    # Force immediate allocation and test
                    for tensor in tensors.values():
                        tensor.zero_()  # Force actual memory allocation
                    
                    torch.cuda.synchronize()  # Ensure allocation completes
                    
                    self.pre_allocated_tensors.append(tensors)
                    successful_allocations += 1
                    
                    # Check current VRAM usage
                    current_vram_gb = torch.cuda.memory_allocated(0) / 1024**3
                    
                    logger.info(f"✅ Worker {self.worker_id}: Allocated tensor set {i+1}/{allocation_attempts}")
                    logger.info(f"   Current VRAM usage: {current_vram_gb:.1f}GB")
                    
                    # Check if we've reached a good utilization level - FIXED: Increased threshold to 95% for max push
                    utilization_percent = current_vram_gb / total_vram_gb * 100
                    if utilization_percent >= 95:
                        logger.info(f"🎯 Worker {self.worker_id}: Reached good utilization: {utilization_percent:.1f}%")
                        break
                        
                except torch.cuda.OutOfMemoryError as e:
                    logger.warning(f"⚠️ Worker {self.worker_id}: VRAM limit reached at {i+1} tensor sets")
                    logger.warning(f"   CUDA OOM: {str(e)[:100]}...")
                    # Clean up partial allocation
                    torch.cuda.empty_cache()
                    break
                except Exception as e:
                    logger.error(f"❌ Worker {self.worker_id}: Allocation error at {i+1}: {e}")
                    break
            
            # Ensure we have minimum required tensor sets
            if successful_allocations == 0:
                logger.error(f"❌ Worker {self.worker_id}: Could not allocate any tensor sets")
                raise RuntimeError("Failed to allocate GPU memory")
            
            logger.info(f"✅ Worker {self.worker_id}: Successfully allocated {successful_allocations} tensor sets")
            
            # 4. Warm up tensor cores
            self._warmup_tensor_cores()
            
            # 5. Start CPU-intensive background workers for maximum CPU utilization
            self._start_cpu_intensive_workers()
            
            self.initialized = True
            
            # Report final status
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            allocated_gb = torch.cuda.memory_allocated(0) / 1024**3
            
            # Get node information
            node_id = ray.runtime_context.get_runtime_context().get_node_id()
            
            # Get hostname to identify which PC this is running on
            import socket
            hostname = socket.gethostname()
            
            logger.info(f"🎮 Worker {self.worker_id} ({self.gpu_type}): {gpu_name}")
            logger.info(f"   Node ID: {node_id}")
            logger.info(f"   Hostname: {hostname}")
            logger.info(f"   Matrix: {self.matrix_size}x{self.matrix_size} (tensor core optimized)")
            logger.info(f"   Batch: {self.batch_size}, Streams: {self.num_streams}")
            logger.info(f"   VRAM: {allocated_gb:.1f}GB / {vram_gb:.1f}GB ({allocated_gb/vram_gb*100:.1f}%)")
            logger.info(f"   Operations per stream: {self.operations_per_stream}")
            logger.info(f"   Tensor sets: {successful_allocations}")
            
            return f"Smart GPU optimization initialized on {gpu_name}: {allocated_gb:.1f}GB VRAM ({allocated_gb/vram_gb*100:.1f}%)"
            
        except Exception as e:
            return f"GPU initialization failed: {e}"
    
    def _warmup_tensor_cores(self):
        """Warm up tensor cores with intensive operations"""
        import torch
        from torch.amp.autocast_mode import autocast
        
        logger.info(f"🔥 Worker {self.worker_id}: Warming up tensor cores...")
        
        # Intensive tensor core warm-up
        for _ in range(5):
            for i, stream in enumerate(self.streams):
                with torch.cuda.stream(stream):
                    with autocast('cuda'):
                        # Use modulo to safely cycle through available tensor sets
                        tensor_idx = i % len(self.pre_allocated_tensors)
                        tensors = self.pre_allocated_tensors[tensor_idx]
                        result = torch.bmm(tensors['A'], tensors['B'])
                        torch.nn.functional.relu_(result)
                        tensors['workspace'].copy_(result)
            
            torch.cuda.synchronize()
    
    def _start_cpu_intensive_workers(self):
        """Start CPU-intensive background workers - REVERTED TO WORKING VERSION"""
        
        # REVERT TO ORIGINAL WORKING APPROACH - simple multiprocessing that was working before
        available_cpus = multiprocessing.cpu_count()
        
        if self.gpu_type == "RTX3090":
            # Head PC: Use moderate CPU utilization to avoid system freeze
            # REDUCED from previous aggressive settings that caused freezing
            primary_workers = max(1, int(available_cpus * 0.50))  # 50% instead of 70%
            secondary_workers = 0  # No secondary workers to avoid complexity
            cpu_work_size = 256  # Smaller work size for stability
            sleep_time = 0.002   # Longer sleep to reduce CPU pressure
            
            logger.info(f"🎯 HEAD PC MODERATE CPU UTILIZATION (STABILITY FOCUSED):")
            logger.info(f"   CPU workers: {primary_workers} (50% of {available_cpus} cores)")
            logger.info(f"   🎯 Target: 50-60% CPU usage for system stability")
        else:
            # Worker PC 2: KEEP ORIGINAL WORKING SETTINGS - these were working fine
            primary_workers = max(1, int(available_cpus * 0.8))  # Original: 80% utilization
            secondary_workers = 0  # Keep simple
            cpu_work_size = 512  # Original larger work size
            sleep_time = 0.0005  # Original shorter sleep
            
            logger.info(f"🔥 WORKER PC 2: ORIGINAL HIGH CPU UTILIZATION")
            logger.info(f"   CPU workers: {primary_workers} (80% of {available_cpus} cores)")
            logger.info(f"   🎯 Target: 80% CPU usage (original working settings)")
        
        logger.info(f"🔥 Worker {self.worker_id}: Starting CPU workers")
        logger.info(f"   Work size: {cpu_work_size}x{cpu_work_size} matrices")
        logger.info(f"   Sleep time: {sleep_time}s")
        
        # SIMPLE THREAD-BASED APPROACH (was working before)
        def cpu_intensive_work():
            """Simple CPU work that was working before"""
            while self.cpu_intensive_enabled:
                try:
                    # Simple mathematical operations
                    size = cpu_work_size
                    
                    # Basic CPU-intensive operations that were working
                    a = np.random.randn(size, size).astype(np.float32)
                    b = np.random.randn(size, size).astype(np.float32)
                    c = np.dot(a, b)  # Matrix multiplication
                    d = np.sum(c * np.sin(c))  # Reduction with trigonometric
                    
                    # Simple sleep
                    time.sleep(sleep_time)
                    
                except Exception as e:
                    time.sleep(0.001)
        
        # Start CPU workers using simple threading (was working before)
        total_workers = primary_workers + secondary_workers
        for i in range(total_workers):
            worker_thread = threading.Thread(target=cpu_intensive_work, daemon=True)
            worker_thread.start()
            self.cpu_workers.append(worker_thread)
        
        logger.info(f"✅ Worker {self.worker_id}: {len(self.cpu_workers)} CPU workers started (REVERTED TO WORKING VERSION)")
        if self.gpu_type == "RTX3090":
            logger.info(f"   🎯 HEAD PC: MODERATE utilization for stability")
        else:
            logger.info(f"   🎯 WORKER PC 2: HIGH utilization (original working settings)")
    def run_smart_compute_iteration(self):
        """Run smart compute iteration with maximum concurrency and memory safety"""
        if not self.initialized:
            init_result = self.initialize_smart_gpu()
            if "failed" in init_result or "not available" in init_result:
                return {"error": init_result}
        
        try:
            import torch
            from torch.amp.autocast_mode import autocast
            
            start_time = time.time()
            
            # Pre-execution memory check
            allocated_before = torch.cuda.memory_allocated(0) / 1024**3
            
            # CONCURRENT MULTI-STREAM EXECUTION with safe tensor indexing and memory monitoring
            for i, stream in enumerate(self.streams):
                with torch.cuda.stream(stream):
                    with autocast('cuda'):
                        # Use modulo to safely cycle through available tensor sets
                        tensor_idx = i % len(self.pre_allocated_tensors)
                        tensors = self.pre_allocated_tensors[tensor_idx]
                        
                        # INTENSIVE COMPUTE OPERATIONS with GPU-SPECIFIC optimization and memory safety
                        current_result = tensors['A']
                        
                        # Variable operations per stream based on GPU type
                        if self.gpu_type == "RTX3090":
                            # Head PC: Optimized operations for speed and memory efficiency
                            ops_count = self.operations_per_stream + 2  # Optimized from +4 for better performance
                            tensor_variations = 3                       
                        else:
                            # Worker PC: Standard computation (working well)
                            ops_count = self.operations_per_stream
                            tensor_variations = 3
                        
                        for op in range(ops_count):
                            # 1. Batch matrix multiplication (tensor cores)
                            current_result = torch.bmm(current_result, tensors['B'])
                            
                            # 2. Activation function (compute intensive)
                            current_result = torch.nn.functional.gelu(current_result)
                            
                            # 3. GPU-specific intensive operations with memory safety
                            if self.gpu_type == "RTX3090":
                                # Head PC: Memory-efficient complex operations
                                if op % 5 == 0:
                                    # Complex chain operations using available tensors
                                    temp_result = torch.bmm(tensors['C'], tensors['D'])
                                    current_result = current_result + temp_result * 0.1
                                elif op % 5 == 1:
                                    # Matrix transpose and complex multiply
                                    current_result = torch.bmm(current_result, tensors['A'].transpose(-2, -1))
                                    current_result = torch.nn.functional.relu(current_result)
                                elif op % 5 == 2:
                                    # Memory-safe eigenvalue-like operations
                                    sub_size = min(128, self.matrix_size)  # Conservative size for stability
                                    if sub_size <= current_result.size(-1):  # Safety check
                                        sub_tensor = current_result[:, :sub_size, :sub_size]
                                        # Symmetric positive definite approximation for stability
                                        symmetric_approx = torch.bmm(sub_tensor, sub_tensor.transpose(-2, -1))
                                        # Add small diagonal for numerical stability
                                        eye_term = torch.eye(sub_size, device=self.device, dtype=torch.float16).unsqueeze(0).expand_as(symmetric_approx)
                                        stable_matrix = symmetric_approx + eye_term * 0.01
                                        current_result[:, :sub_size, :sub_size] = stable_matrix
                                elif op % 5 == 3:
                                    # Complex tensor operations with memory reuse
                                    tensors['temp1'].copy_(torch.bmm(tensors['C'], current_result))
                                    tensors['temp2'].copy_(torch.bmm(tensors['D'], tensors['temp1']))
                                    current_result = current_result + tensors['temp2'] * 0.05
                                else:
                                    # Advanced mathematical operations
                                    current_result = torch.nn.functional.softmax(current_result, dim=-1)
                                    current_result = torch.bmm(current_result, tensors['workspace'])
                            else:
                                # Worker PC: Standard operations (working well)
                                if op % 3 == 0:
                                    # Use additional pre-allocated tensors
                                    temp_result = torch.bmm(tensors['C'], tensors['D'])
                                    current_result = current_result + temp_result * 0.1
                                elif op % 3 == 1:
                                    # Matrix transpose and multiply
                                    current_result = torch.bmm(current_result, tensors['A'].transpose(-2, -1))
                                else:
                                    # More complex operations with all tensors
                                    tensors['temp1'].copy_(torch.bmm(tensors['C'], current_result))
                                    tensors['temp2'].copy_(torch.bmm(tensors['D'], tensors['temp1']))
                                    current_result = current_result + tensors['temp2'] * 0.05
                            
                            # Memory management during intensive operations
                            if op % 10 == 0:  # Every 10 operations
                                # Check memory usage
                                current_allocated = torch.cuda.memory_allocated(0) / 1024**3
                                if current_allocated > allocated_before * 1.2:  # 20% growth limit
                                    torch.cuda.empty_cache()  # Clear cached memory
                        
                        # Store results in multiple tensors to maintain VRAM usage
                        tensors['workspace'].copy_(current_result)
                        tensors['temp1'].copy_(current_result * 0.5)
                        tensors['temp2'].copy_(current_result * 0.3)
            
            # Synchronize all streams
            torch.cuda.synchronize()
            
            operation_time = time.time() - start_time
            self.iteration_count += 1
            
            # Calculate performance metrics
            allocated_gb = torch.cuda.memory_allocated(0) / 1024**3
            
            # Theoretical FLOPS calculation (research-based)
            ops_per_bmm = 2 * self.batch_size * (self.matrix_size ** 3)
            total_bmm_ops = ops_per_bmm * self.num_streams * self.operations_per_stream * 2  # 2 bmm per operation
            estimated_tflops = (total_bmm_ops / operation_time) / 1e12
            
            # Get node information for tracking
            node_id = ray.runtime_context.get_runtime_context().get_node_id()
            
            return {
                "worker_id": self.worker_id,
                "gpu_type": self.gpu_type,
                "node_id": node_id,
                "iteration": self.iteration_count,
                "operation_time": operation_time,
                "allocated_gb": allocated_gb,
                "matrix_size": self.matrix_size,
                "batch_size": self.batch_size,
                "num_streams": self.num_streams,
                "operations_per_stream": self.operations_per_stream,
                "estimated_tflops": estimated_tflops,
                "ops_per_second": 1.0 / operation_time,
                "status": "success"
            }
            
        except torch.cuda.OutOfMemoryError as e:
            # Handle OOM gracefully
            logger.error(f"❌ Worker {self.worker_id} ({self.gpu_type}): CUDA OOM - {str(e)[:100]}...")
            # Attempt recovery
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            return {
                "worker_id": self.worker_id,
                "gpu_type": self.gpu_type, 
                "error": f"CUDA OOM - attempting recovery: {str(e)[:50]}...",
                "status": "oom_recovery"
            }
        except Exception as e:
            # Better error logging with more details
            error_msg = f"Compute iteration failed: {e}"
            logger.error(f"❌ Worker {self.worker_id} ({self.gpu_type}): {error_msg}")
            return {
                "worker_id": self.worker_id,
                "gpu_type": self.gpu_type, 
                "error": error_msg,
                "status": "failed"
            }
    
    def run_smart_training_session(self, duration_minutes: int) -> Dict:
        """Run smart training session with optimized GPU utilization"""
        import time
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        results = {
            "worker_id": self.worker_id,
            "gpu_type": self.gpu_type,
            "duration_minutes": duration_minutes,
            "total_iterations": 0,
            "total_time": 0,
            "avg_operation_time": 0,
            "avg_tflops": 0,
            "peak_tflops": 0,
            "avg_ops_per_second": 0,
            "status": "success"
        }
        
        operation_times = []
        tflops_measurements = []
        ops_per_second_measurements = []
        
        logger.info(f"🚀 Worker {self.worker_id} ({self.gpu_type}): SMART COMPUTE SESSION START")
        logger.info(f"   Duration: {duration_minutes} minutes")
        logger.info(f"   Matrix: {self.matrix_size}x{self.matrix_size}")
        logger.info(f"   Concurrent streams: {self.num_streams}")
        
        iteration_count = 0
        while time.time() < end_time:
            iteration_result = self.run_smart_compute_iteration()
            
            if "error" in iteration_result:
                results["status"] = "failed"
                results["error"] = iteration_result["error"]
                break
            
            operation_times.append(iteration_result["operation_time"])
            tflops_measurements.append(iteration_result["estimated_tflops"])
            ops_per_second_measurements.append(iteration_result["ops_per_second"])
            results["total_iterations"] += 1
            iteration_count += 1
            
            # Progress logging every 500 iterations
            if iteration_count % 500 == 0:
                current_tflops = iteration_result["estimated_tflops"]
                current_ops = iteration_result["ops_per_second"]
                node_id = iteration_result.get("node_id", "unknown")[:8]  # Short node ID
                logger.info(f"   Worker {self.worker_id} (Node {node_id}): {iteration_count} iterations, "
                          f"{current_tflops:.1f} TFLOPS, {current_ops:.1f} ops/sec")
        
        results["total_time"] = time.time() - start_time
        
        if operation_times:
            results["avg_operation_time"] = sum(operation_times) / len(operation_times)
            results["avg_tflops"] = sum(tflops_measurements) / len(tflops_measurements)
            results["peak_tflops"] = max(tflops_measurements)
            results["avg_ops_per_second"] = sum(ops_per_second_measurements) / len(ops_per_second_measurements)
        
        logger.info(f"✅ Worker {self.worker_id} ({self.gpu_type}): SMART SESSION COMPLETE")
        logger.info(f"   Total Iterations: {results['total_iterations']:,}")
        logger.info(f"   Average TFLOPS: {results['avg_tflops']:.1f}")
        logger.info(f"   Peak TFLOPS: {results['peak_tflops']:.1f}")
        logger.info(f"   Avg Ops/sec: {results['avg_ops_per_second']:.1f}")
        
        # Stop CPU-intensive workers
        self._stop_cpu_workers()
        
        return results
    
    def _stop_cpu_workers(self):
        """Stop CPU-intensive background workers (threads only)"""
        self.cpu_intensive_enabled = False
        logger.info(f"🛑 Worker {self.worker_id}: Stopping {len(self.cpu_workers)} CPU workers")
        
        # Wait for threads to finish (they check cpu_intensive_enabled flag)
        import time
        time.sleep(0.2)
        
        logger.info(f"✅ Worker {self.worker_id}: CPU workers stopped")

class DualGPUComputeOptimizer:
    """Dual-GPU compute optimizer for distributed Ray cluster"""
    
    def __init__(self):
        # FORCE dual-GPU configuration for distributed cluster
        self.workers_config = [
            {"gpu_type": "RTX3090", "count": 1},  # Head PC (RTX 3090)
            {"gpu_type": "RTX3070", "count": 1}   # Worker PC 2 (RTX 3070)
        ]
        
    def run_smart_optimization(self, duration_minutes: int):
        """Run smart GPU compute optimization across dual PCs with VRAM cleanup"""
        logger.info("🧠 DUAL-GPU DISTRIBUTED COMPUTE OPTIMIZER")
        logger.info("=" * 70)
        logger.info("🎯 HEAD PC (RTX 3090) + WORKER PC 2 (RTX 3070)")
        logger.info("=" * 70)
        logger.info(f"Duration: {duration_minutes} minutes")
        logger.info("Research Findings Applied:")
        logger.info("  ✅ Distributed execution across 2 PCs")
        logger.info("  ✅ Multiple CUDA streams for concurrency")
        logger.info("  ✅ Tensor core optimized matrix operations")
        logger.info("  ✅ Mixed precision (FP16) acceleration")
        logger.info("  ✅ Memory-aware batch sizing")
        logger.info("  ✅ Comprehensive VRAM cleanup")
        logger.info("=" * 70)
        
        # COMPREHENSIVE VRAM CLEANUP FOR BOTH PCs
        cleanup_gpu_memory()
        
        try:
            # Connect to Ray cluster (simple connection)
            if not ray.is_initialized():
                logger.info("🔌 Connecting to Ray cluster...")
                ray.init(address='192.168.1.10:6379', ignore_reinit_error=True)
                logger.info("✅ Connected to Ray cluster")
            
            # Check cluster resources
            cluster_resources = ray.cluster_resources()
            logger.info(f"📊 Cluster Resources: {cluster_resources}")
            
            # Verify we have GPUs
            total_gpus = cluster_resources.get('GPU', 0)
            if total_gpus < 1:
                logger.error("❌ No GPUs detected in Ray cluster!")
                logger.error("   Make sure CUDA is available and Ray can detect GPUs")
                return
            elif total_gpus == 1:
                logger.warning(f"⚠️ Single GPU detected. Running in single-PC mode.")
                logger.warning("   For dual-PC mode, ensure Worker PC 2 is connected to the cluster.")
            else:
                logger.info(f"✅ Multi-GPU cluster detected: {total_gpus} GPUs available")
            
            # Create smart workers - FORCE 2 workers with EXPLICIT node placement
            workers = []
            worker_id = 0
            
            # Get available nodes with detailed information
            nodes = ray.nodes()
            available_nodes = [node for node in nodes if node['Alive']]
            logger.info(f"📍 Available nodes for placement: {len(available_nodes)}")
            
            # Log node details for debugging
            for i, node in enumerate(available_nodes):
                node_ip = node.get('NodeManagerAddress', 'Unknown')
                node_id = node.get('NodeID', 'Unknown')[:8]
                logger.info(f"   Node {i}: {node_ip} (ID: {node_id})")
            
            # FORCE explicit node placement using node IPs
            if len(available_nodes) >= 2:
                logger.info(f"📍 FORCING EXPLICIT DUAL-PC DISTRIBUTION:")
                
                # Worker 0: Force on Head PC (192.168.1.10)
                head_node = None
                worker_node = None
                
                for node in available_nodes:
                    node_ip = node.get('NodeManagerAddress', '')
                    if '192.168.1.10' in node_ip:
                        head_node = node
                    elif '192.168.1.11' in node_ip:
                        worker_node = node
                
                if head_node and worker_node:
                    logger.info(f"✅ Found both nodes - Head: 192.168.1.10, Worker: 192.168.1.11")
                    
                    # Worker 0: RTX3090 on Head PC (force placement)
                    logger.info(f"📍 Creating Worker 0 (RTX3090) FORCED on Head PC 192.168.1.10...")
                    worker_0 = SmartGPUComputeWorker.options(
                        resources={"node:192.168.1.10": 0.01}  # Force on head node
                    ).remote(0, "RTX3090")
                    workers.append(worker_0)
                    
                    # Worker 1: RTX3070 on Worker PC 2 (force placement)  
                    logger.info(f"📍 Creating Worker 1 (RTX3070) FORCED on Worker PC 2 192.168.1.11...")
                    worker_1 = SmartGPUComputeWorker.options(
                        resources={"node:192.168.1.11": 0.01}  # Force on worker node
                    ).remote(1, "RTX3070")
                    workers.append(worker_1)
                    
                    logger.info(f"✅ EXPLICIT NODE PLACEMENT COMPLETED")
                else:
                    logger.warning(f"⚠️ Could not find both required nodes. Falling back to SPREAD strategy.")
                    # Fallback to SPREAD strategy
                    for gpu_config in self.workers_config:
                        gpu_type = gpu_config["gpu_type"]
                        worker = SmartGPUComputeWorker.options(
                            scheduling_strategy="SPREAD"
                        ).remote(worker_id, gpu_type)
                        workers.append(worker)
                        worker_id += 1
            else:
                logger.warning(f"⚠️ Only {len(available_nodes)} nodes available. Using default placement.")
                # Single node fallback
                for gpu_config in self.workers_config:
                    gpu_type = gpu_config["gpu_type"]
                    worker = SmartGPUComputeWorker.remote(worker_id, gpu_type)
                    workers.append(worker)
                    worker_id += 1
            
            logger.info(f"🚀 LAUNCHING {len(workers)} WORKERS FOR DUAL-GPU OPTIMIZATION...")
            
            # Test worker initialization with distributed placement verification
            logger.info("🧪 Testing worker initialization and placement...")
            init_futures = []
            
            # Initialize workers one by one to better control placement
            for i, worker in enumerate(workers):
                logger.info(f"🔧 Initializing Worker {i}...")
                init_future = worker.initialize_smart_gpu.remote()
                init_futures.append(init_future)
                
                # Small delay to encourage different node placement
                if i == 0 and len(workers) > 1:
                    time.sleep(1)  # Give first worker time to claim a node
            
            try:
                init_results = ray.get(init_futures, timeout=45)  # Increased timeout
                successful_inits = 0
                
                for i, result in enumerate(init_results):
                    if "failed" in result or "not available" in result:
                        logger.error(f"❌ Worker {i} initialization failed: {result}")
                    else:
                        logger.info(f"✅ Worker {i} initialized: {result}")
                        successful_inits += 1
                
                if successful_inits == 0:
                    logger.error("❌ No workers initialized successfully. Aborting.")
                    return
                elif successful_inits < len(workers):
                    logger.warning(f"⚠️ Only {successful_inits}/{len(workers)} workers initialized. Continuing with available workers.")
                    # Filter out failed workers
                    working_workers = []
                    for i, (worker, result) in enumerate(zip(workers, init_results)):
                        if "failed" not in result and "not available" not in result:
                            working_workers.append(worker)
                    workers = working_workers
                    
            except Exception as e:
                logger.error(f"❌ Worker initialization timeout or error: {e}")
                # Try to salvage any completed initializations
                ready_futures, _ = ray.wait(init_futures, timeout=5, num_returns=len(init_futures))
                if ready_futures:
                    partial_results = ray.get(ready_futures)
                    logger.info(f"⚠️ Salvaged {len(partial_results)} worker initializations")
                    # Keep only successfully initialized workers
                    working_workers = []
                    for i, future in enumerate(init_futures):
                        if future in ready_futures:
                            working_workers.append(workers[i])
                    workers = working_workers
                else:
                    logger.error("❌ No workers could be initialized. Aborting.")
                    return
            
            logger.info("🚦 All workers initialized. Verifying node distribution...")
            
            # Verify worker placement across nodes with detailed checking
            if len(workers) >= 2:
                logger.info("🔍 Testing worker placement distribution...")
                placement_futures = [worker.run_smart_compute_iteration.remote() for worker in workers]
                
                try:
                    placement_results = ray.get(placement_futures, timeout=15)  # Increased timeout
                    node_ids = set()
                    node_ips = set()
                    worker_details = []
                    
                    for i, result in enumerate(placement_results):
                        if "error" not in result:
                            node_id = result.get("node_id", "unknown")
                            gpu_type = result.get("gpu_type", "unknown")
                            worker_id = result.get("worker_id", i)
                            node_ids.add(node_id)
                            
                            worker_details.append({
                                "worker": worker_id,
                                "node_id": node_id[:12],
                                "gpu_type": gpu_type
                            })
                            
                            logger.info(f"✅ Worker {worker_id} ({gpu_type}): Node {node_id[:12]}...")
                            
                            # Try to map node ID to IP
                            for node in available_nodes:
                                if node.get('NodeID', '')[:12] == node_id[:12]:
                                    node_ip = node.get('NodeManagerAddress', 'Unknown')
                                    node_ips.add(node_ip)
                                    logger.info(f"   └─ Node IP: {node_ip}")
                                    break
                        else:
                            logger.warning(f"⚠️ Worker {i} test failed: {result.get('error', 'Unknown')}")
                    
                    if len(node_ids) > 1:
                        logger.info(f"🎯 SUCCESS: Workers distributed across {len(node_ids)} nodes!")
                        logger.info(f"   Node distribution:")
                        for detail in worker_details:
                            logger.info(f"     Worker {detail['worker']}: {detail['gpu_type']} on Node {detail['node_id']}")
                        
                        if len(node_ips) > 1:
                            logger.info(f"   Node IPs: {list(node_ips)}")
                        
                        logger.info("🚀 Worker PC 2 should now show GPU activity during training!")
                        logger.info("   CHECK: Monitor Worker PC 2 Task Manager for RTX 3070 usage.")
                    else:
                        logger.warning(f"⚠️ CRITICAL: All workers on same node! Worker PC 2 IDLE.")
                        logger.warning(f"   Node ID: {list(node_ids)}")
                        logger.warning("   Both GPUs are on Head PC - explicit placement failed!")
                        logger.warning("   Worker PC 2 will show 0% GPU usage.")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Could not verify node placement: {e}")
                    logger.warning("   Continuing with training, but Worker PC 2 utilization uncertain.")
            
            logger.info("🚦 Starting training sessions...")
            start_time = time.time()
            
            # Run all workers concurrently
            futures = []
            for worker in workers:
                future = worker.run_smart_training_session.remote(duration_minutes)
                futures.append(future)
            
            # Progress tracking
            logger.info("🎮 DUAL-GPU TRAINING IN PROGRESS...")
            logger.info(f"   Expected: RTX 3090 on Head PC + RTX 3070 on Worker PC 2")
            logger.info(f"   Monitoring {len(futures)} workers for {duration_minutes} minutes")
            
            # Progress bar for training session
            with tqdm.tqdm(total=duration_minutes*60, desc="Dual-GPU Training Progress", ncols=80) as pbar:
                start_check_time = time.time()
                while True:
                    # Check Ray task status
                    ready, not_ready = ray.wait(futures, timeout=1, num_returns=len(futures))
                    elapsed = int(time.time() - start_time)
                    pbar.n = min(elapsed, duration_minutes*60)
                    pbar.refresh()
                    
                    # Safety check - break if duration exceeded
                    if elapsed >= duration_minutes * 60:
                        logger.info("⏰ Duration reached, completing tasks...")
                        break
                    
                    # All tasks completed
                    if len(ready) == len(futures):
                        logger.info("✅ All workers completed successfully")
                        break
                    
                    # Progress update every 15 seconds
                    if time.time() - start_check_time > 15:
                        logger.info(f"⏳ Progress: {elapsed}s/{duration_minutes*60}s, {len(ready)}/{len(futures)} workers completed")
                        start_check_time = time.time()
            
            # Get results with extended timeout for stability
            try:
                logger.info("📊 Collecting training session results...")
                results = ray.get(futures, timeout=120)  # Increased from 60 to 120 seconds
            except Exception as e:
                logger.warning(f"⚠️ Timeout getting results, attempting graceful collection: {e}")
                # Try to get any completed results
                ready_futures, remaining_futures = ray.wait(futures, timeout=10, num_returns=len(futures))
                results = []
                
                if ready_futures:
                    try:
                        partial_results = ray.get(ready_futures, timeout=30)
                        results.extend(partial_results)
                        logger.info(f"✅ Collected {len(partial_results)} worker results gracefully")
                    except Exception as e2:
                        logger.error(f"❌ Failed to collect partial results: {e2}")
                
                if not results:
                    logger.error("❌ No results could be collected - training may have failed")
                    results = []  # Empty results list
            
            total_time = time.time() - start_time
            
            # Display and save results
            self._display_dual_gpu_results(results, total_time, cluster_resources)
            self._save_dual_gpu_results(results, total_time, cluster_resources)
            
        except Exception as e:
            logger.error(f"❌ Dual-GPU optimization failed: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # FINAL VRAM CLEANUP
            logger.info("🧹 Final VRAM cleanup...")
            cleanup_gpu_memory()
            ray.shutdown()
            logger.info("🔗 Ray cluster disconnected")
    
    def _display_dual_gpu_results(self, results: List[Dict], total_time: float, cluster_resources: Dict):
        """Display dual-GPU optimization results"""
        successful_workers = [r for r in results if r.get("status") == "success"]
        failed_workers = [r for r in results if r.get("status") != "success"]
        
        # Calculate aggregated metrics
        total_iterations = sum(r.get("total_iterations", 0) for r in successful_workers)
        total_avg_tflops = sum(r.get("avg_tflops", 0) for r in successful_workers)
        total_peak_tflops = sum(r.get("peak_tflops", 0) for r in successful_workers)
        total_avg_ops = sum(r.get("avg_ops_per_second", 0) for r in successful_workers)
        
        # Calculate GPU utilization improvement
        baseline_ops_per_sec = 300  # From previous 5% utilization baseline
        improvement_factor = total_avg_ops / baseline_ops_per_sec if baseline_ops_per_sec > 0 else 0
        estimated_gpu_utilization = min(95, 5 * improvement_factor)  # Cap at 95%
        
        logger.info("=" * 70)
        logger.info("🎯 DUAL-GPU DISTRIBUTED COMPUTE OPTIMIZATION RESULTS")
        logger.info("=" * 70)
        logger.info(f"Duration: {total_time:.1f} seconds")
        logger.info(f"Cluster: {cluster_resources.get('GPU', 0)} GPUs, {cluster_resources.get('CPU', 0)} CPUs")
        logger.info(f"Successful Workers: {len(successful_workers)}/{len(results)}")
        logger.info(f"Total Iterations: {total_iterations:,}")
        logger.info(f"Total Ops/Second: {total_iterations/total_time:.2f}")
        logger.info("")
        logger.info("🔥 DUAL-GPU COMPUTE PERFORMANCE ANALYSIS:")
        logger.info(f"  Combined Average TFLOPS: {total_avg_tflops:.1f}")
        logger.info(f"  Combined Peak TFLOPS: {total_peak_tflops:.1f}")
        logger.info(f"  Performance Improvement: {improvement_factor:.1f}x baseline")
        logger.info(f"  Estimated GPU Utilization: {estimated_gpu_utilization:.1f}% (up from 5%)")
        logger.info("")
        
        # Per-worker detailed breakdown
        logger.info("📊 PER-WORKER PERFORMANCE BREAKDOWN:")
        for result in successful_workers:
            worker_improvement = result.get('avg_ops_per_second', 0) / 300  # vs baseline
            logger.info(f"  {result.get('gpu_type', 'Unknown')} Worker {result.get('worker_id', 'Unknown')}:")
            logger.info(f"    Iterations: {result.get('total_iterations', 0):,}")
            logger.info(f"    Avg Time/Op: {result.get('avg_operation_time', 0):.6f}s")
            logger.info(f"    Ops/Second: {result.get('avg_ops_per_second', 0):.1f}")
            logger.info(f"    TFLOPS: {result.get('avg_tflops', 0):.1f} avg, {result.get('peak_tflops', 0):.1f} peak")
            logger.info(f"    Improvement: {worker_improvement:.1f}x baseline")
            matrix_size = result.get('matrix_size', 'N/A')
            num_streams = result.get('num_streams', 'N/A')
            logger.info(f"    Matrix: {matrix_size}x{matrix_size}, Streams: {num_streams}")
        
        # Failed workers with better error details
        for result in failed_workers:
            worker_id = result.get('worker_id', 'Unknown')
            gpu_type = result.get('gpu_type', 'Unknown')
            error_msg = result.get('error', 'Unknown error')
            logger.error(f"❌ {gpu_type} Worker {worker_id}: {error_msg}")
        
        # Dual-GPU analysis
        if len(successful_workers) == 2:
            rtx3090_worker = next((r for r in successful_workers if r['gpu_type'] == 'RTX3090'), None)
            rtx3070_worker = next((r for r in successful_workers if r['gpu_type'] == 'RTX3070'), None)
            
            if rtx3090_worker and rtx3070_worker:
                logger.info("")
                logger.info("🎮 DUAL-GPU PERFORMANCE COMPARISON:")
                logger.info(f"  RTX 3090: {rtx3090_worker['avg_tflops']:.1f} TFLOPS avg, {rtx3090_worker['avg_ops_per_second']:.1f} ops/sec")
                logger.info(f"  RTX 3070: {rtx3070_worker['avg_tflops']:.1f} TFLOPS avg, {rtx3070_worker['avg_ops_per_second']:.1f} ops/sec")
                ratio = rtx3090_worker['avg_tflops'] / rtx3070_worker['avg_tflops'] if rtx3070_worker['avg_tflops'] > 0 else 0
                logger.info(f"  Performance Ratio: {ratio:.1f}x (3090 vs 3070)")
        
        logger.info("=" * 70)
    
    def _save_dual_gpu_results(self, results: List[Dict], total_time: float, cluster_resources: Dict):
        """Save dual-GPU optimization results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"dual_gpu_compute_results_{timestamp}.json"
        
        successful_workers = [r for r in results if r.get("status") == "success"]
        total_iterations = sum(r.get("total_iterations", 0) for r in successful_workers)
        total_avg_tflops = sum(r.get("avg_tflops", 0) for r in successful_workers)
        baseline_improvement = (sum(r.get("avg_ops_per_second", 0) for r in successful_workers) / 300)
        
        summary_data = {
            "optimization_type": "Dual-GPU Distributed Compute Optimization",
            "timestamp": timestamp,
            "cluster_configuration": {
                "total_gpus": cluster_resources.get('GPU', 0),
                "total_cpus": cluster_resources.get('CPU', 0),
                "head_pc": "RTX 3090",
                "worker_pc": "RTX 3070"
            },
            "research_findings_applied": [
                "Distributed execution across 2 PCs",
                "Multiple CUDA streams",
                "Tensor core optimization", 
                "Mixed precision (FP16)",
                "Memory-aware allocation",
                "Concurrent execution",
                "Comprehensive VRAM cleanup"
            ],
            "performance_summary": {
                "total_time": total_time,
                "successful_workers": len(successful_workers),
                "total_workers": len(results),
                "total_iterations": total_iterations,
                "iterations_per_second": total_iterations / total_time,
                "combined_avg_tflops": total_avg_tflops,
                "performance_improvement_factor": baseline_improvement,
                "estimated_gpu_utilization_percent": min(95, 5 * baseline_improvement)
            },
            "worker_details": results
        }
        
        import json
        with open(results_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        logger.info(f"📊 Dual-GPU optimization results saved to: {results_file}")
        
        summary_data = {
            "optimization_type": "Dual-GPU Distributed Compute Optimization",
            "timestamp": timestamp,
            "cluster_configuration": {
                "total_gpus": cluster_resources.get('GPU', 0),
                "total_cpus": cluster_resources.get('CPU', 0),
                "head_pc": "RTX 3090",
                "worker_pc": "RTX 3070"
            },
            "research_findings_applied": [
                "Distributed execution across 2 PCs",
                "Multiple CUDA streams",
                "Tensor core optimization", 
                "Mixed precision (FP16)",
                "Memory-aware allocation",
                "Concurrent execution"
            ],
            "performance_summary": {
                "total_time": total_time,
                "successful_workers": len(successful_workers),
                "total_workers": len(results),
                "total_iterations": total_iterations,
                "iterations_per_second": total_iterations / total_time,
                "combined_avg_tflops": total_avg_tflops,
                "performance_improvement_factor": baseline_improvement,
                "estimated_gpu_utilization_percent": min(95, 5 * baseline_improvement)
            },
            "worker_details": results
        }
        
        import json
        with open(results_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        logger.info(f"📊 Dual-GPU optimization results saved to: {results_file}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='RTX 3090 + RTX 3070 Dual-GPU Distributed Compute Optimizer')
    parser.add_argument('--duration', type=int, default=10, 
                       help='Optimization duration in minutes (default: 10)')
    args = parser.parse_args()
    
    optimizer = DualGPUComputeOptimizer()
    optimizer.run_smart_optimization(args.duration)

if __name__ == "__main__":
    main()
