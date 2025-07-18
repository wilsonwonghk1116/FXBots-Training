#!/usr/bin/env python3
"""
RTX 3090 Ultimate Optimization Test Suite
=========================================

Comprehensive testing for RTX 3090 OC 24GB optimizations
Tests all research-backed optimization features.

Usage: python test_rtx3090_ultimate_optimization.py --phase=1
"""

import logging
import time
import torch
from torch.amp.autocast_mode import autocast
import os
import sys
import json
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RTX3090OptimizationTester:
    """Comprehensive test suite for RTX 3090 optimizations"""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = time.time()
        
    def test_phase_1_basic_optimization(self):
        """Test Phase 1: Basic optimization (VRAM + Mixed Precision)"""
        logger.info("🧪 PHASE 1: BASIC OPTIMIZATION TEST")
        logger.info("=" * 50)
        
        results = {
            'phase': 1,
            'tests': {},
            'overall_success': True
        }
        
        # Test 1: CUDA availability and RTX 3090 detection
        logger.info("Test 1: RTX 3090 Detection...")
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            if "3090" in device_name and total_memory > 20:
                logger.info(f"✅ RTX 3090 detected: {device_name} ({total_memory:.1f}GB)")
                results['tests']['rtx3090_detection'] = True
            else:
                logger.warning(f"⚠️ Expected RTX 3090, found: {device_name} ({total_memory:.1f}GB)")
                results['tests']['rtx3090_detection'] = False
        else:
            logger.error("❌ CUDA not available")
            results['tests']['rtx3090_detection'] = False
            results['overall_success'] = False
        
        # Test 2: Mixed Precision Support
        logger.info("Test 2: Mixed Precision Support...")
        try:
            with autocast('cuda'):
                a = torch.randn(1024, 1024, device='cuda:0')
                b = torch.randn(1024, 1024, device='cuda:0')
                c = torch.matmul(a, b)
            
            logger.info("✅ Mixed precision (torch.cuda.amp) working")
            results['tests']['mixed_precision'] = True
        except Exception as e:
            logger.error(f"❌ Mixed precision failed: {e}")
            results['tests']['mixed_precision'] = False
            results['overall_success'] = False
        
        # Test 3: Large VRAM Allocation (22GB target)
        logger.info("Test 3: Large VRAM Allocation (22GB target)...")
        try:
            target_gb = 20.0  # Start with 20GB test
            target_bytes = int(target_gb * 1024**3)
            
            # Try to allocate large tensor
            large_tensor = torch.empty(target_bytes // 4, dtype=torch.float32, device='cuda:0')
            
            allocated_gb = torch.cuda.memory_allocated(0) / 1024**3
            utilization = (allocated_gb / 24.0) * 100  # Assuming 24GB total
            
            logger.info(f"✅ Large allocation successful: {allocated_gb:.2f}GB ({utilization:.1f}%)")
            results['tests']['large_vram_allocation'] = {
                'success': True,
                'allocated_gb': allocated_gb,
                'utilization_percent': utilization
            }
            
            # Cleanup
            del large_tensor
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"❌ Large VRAM allocation failed: {e}")
            results['tests']['large_vram_allocation'] = {
                'success': False,
                'error': str(e)
            }
            results['overall_success'] = False
        
        # Test 4: Tensor Core Operations (8192x8192)
        logger.info("Test 4: Tensor Core Operations...")
        try:
            matrix_size = 8192
            
            start_time = time.time()
            with autocast('cuda'):
                a = torch.randn(matrix_size, matrix_size, device='cuda:0', dtype=torch.float16)
                b = torch.randn(matrix_size, matrix_size, device='cuda:0', dtype=torch.float16)
                c = torch.matmul(a, b)
                torch.cuda.synchronize()
            
            operation_time = time.time() - start_time
            tflops = (2 * matrix_size**3) / (operation_time * 1e12)  # Rough TFLOPS calculation
            
            logger.info(f"✅ Tensor core operation: {operation_time:.3f}s ({tflops:.1f} TFLOPS)")
            results['tests']['tensor_core_ops'] = {
                'success': True,
                'operation_time': operation_time,
                'estimated_tflops': tflops
            }
            
        except Exception as e:
            logger.error(f"❌ Tensor core operation failed: {e}")
            results['tests']['tensor_core_ops'] = {
                'success': False,
                'error': str(e)
            }
            results['overall_success'] = False
        
        return results
    
    def test_phase_2_thermal_management(self):
        """Test Phase 2: Thermal management and monitoring"""
        logger.info("\n🧪 PHASE 2: THERMAL MANAGEMENT TEST")
        logger.info("=" * 50)
        
        results = {
            'phase': 2,
            'tests': {},
            'overall_success': True
        }
        
        # Test 1: Temperature Monitoring
        logger.info("Test 1: Temperature Monitoring...")
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            
            logger.info(f"✅ GPU Temperature: {temp}°C")
            results['tests']['temperature_monitoring'] = {
                'success': True,
                'temperature': temp
            }
            
            if temp > 85:
                logger.warning(f"⚠️ High temperature detected: {temp}°C")
                results['tests']['temperature_monitoring']['warning'] = f"High temp: {temp}°C"
                
        except Exception as e:
            logger.warning(f"⚠️ Temperature monitoring failed: {e}")
            logger.info("💡 Install nvidia-ml-py3: pip install nvidia-ml-py3")
            results['tests']['temperature_monitoring'] = {
                'success': False,
                'error': str(e)
            }
        
        # Test 2: Sustained Load Test (30 seconds)
        logger.info("Test 2: Sustained Load Test (30 seconds)...")
        try:
            start_temp = None
            end_temp = None
            max_temp = 0
            
            # Get initial temperature
            try:
                import pynvml
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                start_temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except:
                pass
            
            # Run sustained operations
            start_time = time.time()
            operation_count = 0
            
            while time.time() - start_time < 30:  # 30 second test
                with autocast('cuda'):
                    a = torch.randn(4096, 4096, device='cuda:0', dtype=torch.float16)
                    b = torch.randn(4096, 4096, device='cuda:0', dtype=torch.float16)
                    c = torch.matmul(a, b)
                    torch.cuda.synchronize()
                
                operation_count += 1
                
                # Monitor temperature
                try:
                    current_temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    max_temp = max(max_temp, current_temp)
                except:
                    pass
            
            # Get final temperature
            try:
                end_temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except:
                pass
            
            total_time = time.time() - start_time
            ops_per_second = operation_count / total_time
            
            logger.info(f"✅ Sustained load completed: {operation_count} ops in {total_time:.1f}s")
            logger.info(f"   Operations/second: {ops_per_second:.1f}")
            if start_temp and end_temp:
                logger.info(f"   Temperature: {start_temp}°C → {end_temp}°C (max: {max_temp}°C)")
            
            results['tests']['sustained_load'] = {
                'success': True,
                'operations': operation_count,
                'duration': total_time,
                'ops_per_second': ops_per_second,
                'start_temp': start_temp,
                'end_temp': end_temp,
                'max_temp': max_temp
            }
            
            if max_temp > 83:
                logger.warning(f"⚠️ High temperature during load: {max_temp}°C")
                results['tests']['sustained_load']['thermal_warning'] = True
            
        except Exception as e:
            logger.error(f"❌ Sustained load test failed: {e}")
            results['tests']['sustained_load'] = {
                'success': False,
                'error': str(e)
            }
            results['overall_success'] = False
        
        return results
    
    def test_phase_3_maximum_performance(self):
        """Test Phase 3: Maximum performance optimization"""
        logger.info("\n🧪 PHASE 3: MAXIMUM PERFORMANCE TEST")
        logger.info("=" * 50)
        
        results = {
            'phase': 3,
            'tests': {},
            'overall_success': True
        }
        
        # Test 1: Maximum VRAM Utilization
        logger.info("Test 1: Maximum VRAM Utilization (22GB target)...")
        try:
            # Gradually allocate memory until near maximum
            allocations = []
            total_allocated = 0
            target_gb = 22.0
            
            chunk_size_gb = 2.0
            chunk_bytes = int(chunk_size_gb * 1024**3)
            
            while total_allocated < target_gb:
                try:
                    chunk = torch.empty(chunk_bytes // 4, dtype=torch.float32, device='cuda:0')
                    allocations.append(chunk)
                    total_allocated += chunk_size_gb
                    
                    current_allocated = torch.cuda.memory_allocated(0) / 1024**3
                    logger.info(f"   Allocated: {current_allocated:.1f}GB ({(current_allocated/24)*100:.1f}%)")
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.info(f"   Reached memory limit at {total_allocated:.1f}GB")
                        break
                    else:
                        raise e
            
            final_allocated = torch.cuda.memory_allocated(0) / 1024**3
            utilization = (final_allocated / 24.0) * 100
            
            logger.info(f"✅ Maximum allocation: {final_allocated:.1f}GB ({utilization:.1f}%)")
            results['tests']['maximum_vram'] = {
                'success': True,
                'allocated_gb': final_allocated,
                'utilization_percent': utilization,
                'target_achieved': utilization >= 90
            }
            
            # Cleanup
            for chunk in allocations:
                del chunk
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"❌ Maximum VRAM test failed: {e}")
            results['tests']['maximum_vram'] = {
                'success': False,
                'error': str(e)
            }
            results['overall_success'] = False
        
        # Test 2: Performance Benchmark
        logger.info("Test 2: Performance Benchmark...")
        try:
            # Benchmark different configurations
            configs = [
                {'name': 'Baseline FP32', 'size': 4096, 'mixed_precision': False},
                {'name': 'Mixed Precision FP16', 'size': 4096, 'mixed_precision': True},
                {'name': 'Large Tensor Cores', 'size': 8192, 'mixed_precision': True}
            ]
            
            benchmark_results = []
            
            for config in configs:
                logger.info(f"   Testing {config['name']}...")
                
                times = []
                for _ in range(5):  # 5 iterations
                    start_time = time.time()
                    
                    if config['mixed_precision']:
                        with autocast('cuda'):
                            a = torch.randn(config['size'], config['size'], device='cuda:0')
                            b = torch.randn(config['size'], config['size'], device='cuda:0')
                            c = torch.matmul(a, b)
                            torch.cuda.synchronize()
                    else:
                        a = torch.randn(config['size'], config['size'], device='cuda:0')
                        b = torch.randn(config['size'], config['size'], device='cuda:0')
                        c = torch.matmul(a, b)
                        torch.cuda.synchronize()
                    
                    times.append(time.time() - start_time)
                
                avg_time = sum(times) / len(times)
                tflops = (2 * config['size']**3) / (avg_time * 1e12)
                
                result = {
                    'config': config['name'],
                    'avg_time': avg_time,
                    'tflops': tflops
                }
                benchmark_results.append(result)
                
                logger.info(f"     Average time: {avg_time:.3f}s ({tflops:.1f} TFLOPS)")
            
            # Calculate speedups
            baseline_tflops = benchmark_results[0]['tflops']
            for result in benchmark_results[1:]:
                speedup = result['tflops'] / baseline_tflops
                result['speedup'] = speedup
                logger.info(f"   {result['config']}: {speedup:.1f}x speedup")
            
            results['tests']['performance_benchmark'] = {
                'success': True,
                'results': benchmark_results
            }
            
        except Exception as e:
            logger.error(f"❌ Performance benchmark failed: {e}")
            results['tests']['performance_benchmark'] = {
                'success': False,
                'error': str(e)
            }
            results['overall_success'] = False
        
        return results
    
    def run_comprehensive_test(self, phases=None):
        """Run comprehensive test suite"""
        if phases is None:
            phases = [1, 2, 3]
        
        logger.info("🚀 RTX 3090 ULTIMATE OPTIMIZATION TEST SUITE")
        logger.info("=" * 60)
        logger.info(f"Test Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Phases to test: {phases}")
        logger.info("")
        
        all_results = {
            'test_suite': 'RTX 3090 Ultimate Optimization',
            'start_time': datetime.now().isoformat(),
            'phases_tested': phases,
            'results': {},
            'overall_success': True
        }
        
        # Run requested phases
        if 1 in phases:
            phase1_results = self.test_phase_1_basic_optimization()
            all_results['results']['phase_1'] = phase1_results
            if not phase1_results['overall_success']:
                all_results['overall_success'] = False
        
        if 2 in phases:
            phase2_results = self.test_phase_2_thermal_management()
            all_results['results']['phase_2'] = phase2_results
            if not phase2_results['overall_success']:
                all_results['overall_success'] = False
        
        if 3 in phases:
            phase3_results = self.test_phase_3_maximum_performance()
            all_results['results']['phase_3'] = phase3_results
            if not phase3_results['overall_success']:
                all_results['overall_success'] = False
        
        # Final summary
        total_time = time.time() - self.start_time
        all_results['total_time_seconds'] = total_time
        all_results['end_time'] = datetime.now().isoformat()
        
        logger.info("\n" + "=" * 60)
        logger.info("🎯 TEST SUITE SUMMARY")
        logger.info("=" * 60)
        
        for phase_name, phase_results in all_results['results'].items():
            phase_num = phase_results['phase']
            success_count = sum(1 for test in phase_results['tests'].values() 
                              if (isinstance(test, dict) and test.get('success', False)) or 
                              (isinstance(test, bool) and test))
            total_tests = len(phase_results['tests'])
            status = "✅ PASSED" if phase_results['overall_success'] else "❌ FAILED"
            
            logger.info(f"Phase {phase_num}: {status} ({success_count}/{total_tests} tests passed)")
        
        overall_status = "✅ ALL TESTS PASSED" if all_results['overall_success'] else "❌ SOME TESTS FAILED"
        logger.info(f"\nOverall Status: {overall_status}")
        logger.info(f"Total Time: {total_time:.1f} seconds")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"rtx3090_optimization_test_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logger.info(f"📊 Detailed results saved to: {results_file}")
        logger.info("=" * 60)
        
        return all_results

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='RTX 3090 Ultimate Optimization Test Suite')
    parser.add_argument('--phase', type=int, choices=[1, 2, 3], 
                       help='Test specific phase (1=basic, 2=thermal, 3=performance)')
    parser.add_argument('--all', action='store_true', help='Run all test phases')
    args = parser.parse_args()
    
    tester = RTX3090OptimizationTester()
    
    if args.all:
        phases = [1, 2, 3]
    elif args.phase:
        phases = [args.phase]
    else:
        # Default to phase 1
        phases = [1]
    
    results = tester.run_comprehensive_test(phases)
    
    # Exit code based on success
    sys.exit(0 if results['overall_success'] else 1)

if __name__ == "__main__":
    main() 