#!/usr/bin/env python3
"""
Ray Cluster Performance Monitor for Kelly Monte Carlo Bot System

Real-time monitoring of CPU, GPU, and memory utilization across
the Ray cluster to ensure we achieve target 75% resource usage.

Features:
- Real-time cluster resource monitoring
- GPU utilization tracking for both RTX 3090 and RTX 3070
- CPU usage across all 96 threads (80 + 16)
- Memory and VRAM tracking
- Performance optimization suggestions

Author: TaskMaster AI System
Date: 2025-01-12
"""

import time
import json
import logging
import threading
from datetime import datetime
from typing import Dict, List
import subprocess
import os

import ray
import psutil
import numpy as np
import pandas as pd

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("Warning: GPUtil not available. GPU monitoring disabled.")

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.dates import DateFormatter
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: Matplotlib not available. Real-time plotting disabled.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ray_cluster_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RayClusterMonitor:
    """
    Comprehensive Ray cluster performance monitor
    Tracks CPU, GPU, memory utilization across all nodes
    """
    
    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.monitoring_active = False
        self.performance_data = []
        
        # Target utilization thresholds
        self.target_cpu_utilization = 75.0
        self.target_gpu_utilization = 75.0
        self.target_memory_utilization = 70.0
        
        # Performance tracking
        self.start_time = None
        self.monitor_thread = None
        
        logger.info("Ray Cluster Monitor initialized")
        logger.info(f"Target utilization - CPU: {self.target_cpu_utilization}%, GPU: {self.target_gpu_utilization}%")
    
    def start_monitoring(self):
        """Start real-time cluster monitoring"""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        logger.info("Starting Ray cluster performance monitoring...")
        self.monitoring_active = True
        self.start_time = time.time()
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring and save results"""
        if not self.monitoring_active:
            return
        
        logger.info("Stopping performance monitoring...")
        self.monitoring_active = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        self._save_monitoring_results()
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect performance data
                performance_snapshot = self._collect_performance_data()
                
                if performance_snapshot:
                    self.performance_data.append(performance_snapshot)
                    
                    # Log performance summary every 10 seconds
                    if len(self.performance_data) % 10 == 0:
                        self._log_performance_summary(performance_snapshot)
                    
                    # Check for performance issues
                    self._check_performance_thresholds(performance_snapshot)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_performance_data(self) -> Dict:
        """Collect comprehensive performance data from Ray cluster"""
        timestamp = time.time()
        
        # Ray cluster status
        cluster_data = self._get_ray_cluster_stats()
        
        # Local system stats
        local_stats = self._get_local_system_stats()
        
        # GPU stats
        gpu_stats = self._get_gpu_stats()
        
        performance_snapshot = {
            'timestamp': timestamp,
            'datetime': datetime.fromtimestamp(timestamp).isoformat(),
            'ray_cluster': cluster_data,
            'local_system': local_stats,
            'gpu_stats': gpu_stats,
            'elapsed_time': timestamp - self.start_time if self.start_time else 0
        }
        
        return performance_snapshot
    
    def _get_ray_cluster_stats(self) -> Dict:
        """Get Ray cluster resource utilization"""
        try:
            if not ray.is_initialized():
                return {'status': 'not_initialized'}
            
            # Get cluster resources
            cluster_resources = ray.cluster_resources()
            available_resources = ray.available_resources()
            
            # Calculate utilization
            total_cpus = cluster_resources.get('CPU', 0)
            available_cpus = available_resources.get('CPU', 0)
            cpu_utilization = ((total_cpus - available_cpus) / total_cpus * 100) if total_cpus > 0 else 0
            
            total_gpus = cluster_resources.get('GPU', 0)
            available_gpus = available_resources.get('GPU', 0)
            gpu_utilization = ((total_gpus - available_gpus) / total_gpus * 100) if total_gpus > 0 else 0
            
            # Get node information
            nodes = ray.nodes()
            active_nodes = [node for node in nodes if node['Alive']]
            
            return {
                'status': 'active',
                'total_cpus': total_cpus,
                'available_cpus': available_cpus,
                'cpu_utilization': cpu_utilization,
                'total_gpus': total_gpus,
                'available_gpus': available_gpus,
                'gpu_utilization': gpu_utilization,
                'total_nodes': len(nodes),
                'active_nodes': len(active_nodes),
                'object_store_memory': cluster_resources.get('object_store_memory', 0)
            }
            
        except Exception as e:
            logger.debug(f"Ray cluster stats error: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _get_local_system_stats(self) -> Dict:
        """Get local system performance stats"""
        try:
            # CPU utilization
            cpu_percent = psutil.cpu_percent(interval=None)
            cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)
            
            # Memory utilization
            memory = psutil.virtual_memory()
            
            # Load average
            load_avg = os.getloadavg()
            
            return {
                'cpu_percent': cpu_percent,
                'cpu_per_core': cpu_per_core,
                'cpu_cores': psutil.cpu_count(),
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'memory_total_gb': memory.total / (1024**3),
                'load_average_1m': load_avg[0],
                'load_average_5m': load_avg[1],
                'load_average_15m': load_avg[2]
            }
            
        except Exception as e:
            logger.debug(f"Local system stats error: {e}")
            return {'error': str(e)}
    
    def _get_gpu_stats(self) -> List[Dict]:
        """Get GPU utilization stats"""
        gpu_stats = []
        
        if not GPU_AVAILABLE:
            return gpu_stats
        
        try:
            gpus = GPUtil.getGPUs()
            
            for i, gpu in enumerate(gpus):
                gpu_info = {
                    'gpu_id': i,
                    'name': gpu.name,
                    'load_percent': gpu.load * 100,
                    'memory_percent': gpu.memoryUtil * 100,
                    'memory_used_mb': gpu.memoryUsed,
                    'memory_total_mb': gpu.memoryTotal,
                    'temperature': gpu.temperature
                }
                gpu_stats.append(gpu_info)
                
        except Exception as e:
            logger.debug(f"GPU stats error: {e}")
        
        return gpu_stats
    
    def _log_performance_summary(self, snapshot: Dict):
        """Log performance summary"""
        ray_data = snapshot.get('ray_cluster', {})
        local_data = snapshot.get('local_system', {})
        gpu_data = snapshot.get('gpu_stats', [])
        
        logger.info("="*60)
        logger.info("CLUSTER PERFORMANCE SUMMARY")
        logger.info("="*60)
        
        # Ray cluster stats
        if ray_data.get('status') == 'active':
            logger.info(f"Ray Cluster:")
            logger.info(f"  CPU Usage: {ray_data.get('cpu_utilization', 0):.1f}% "
                       f"({ray_data.get('total_cpus', 0) - ray_data.get('available_cpus', 0):.0f}/"
                       f"{ray_data.get('total_cpus', 0):.0f} cores)")
            logger.info(f"  GPU Usage: {ray_data.get('gpu_utilization', 0):.1f}% "
                       f"({ray_data.get('total_gpus', 0) - ray_data.get('available_gpus', 0):.1f}/"
                       f"{ray_data.get('total_gpus', 0):.1f} GPUs)")
            logger.info(f"  Active Nodes: {ray_data.get('active_nodes', 0)}/{ray_data.get('total_nodes', 0)}")
        
        # Local system stats
        if local_data:
            logger.info(f"Local System:")
            logger.info(f"  CPU: {local_data.get('cpu_percent', 0):.1f}% "
                       f"({local_data.get('cpu_cores', 0)} cores)")
            logger.info(f"  Memory: {local_data.get('memory_percent', 0):.1f}% "
                       f"({local_data.get('memory_available_gb', 0):.1f}GB available)")
            logger.info(f"  Load Avg: {local_data.get('load_average_1m', 0):.2f}")
        
        # GPU stats
        for i, gpu in enumerate(gpu_data):
            logger.info(f"GPU {i} ({gpu.get('name', 'Unknown')}):")
            logger.info(f"  Usage: {gpu.get('load_percent', 0):.1f}%")
            logger.info(f"  VRAM: {gpu.get('memory_percent', 0):.1f}% "
                       f"({gpu.get('memory_used_mb', 0):.0f}MB/"
                       f"{gpu.get('memory_total_mb', 0):.0f}MB)")
            logger.info(f"  Temp: {gpu.get('temperature', 0):.0f}°C")
        
        logger.info("="*60)
    
    def _check_performance_thresholds(self, snapshot: Dict):
        """Check if performance meets target thresholds"""
        ray_data = snapshot.get('ray_cluster', {})
        local_data = snapshot.get('local_system', {})
        gpu_data = snapshot.get('gpu_stats', [])
        
        # Check CPU utilization
        cpu_util = local_data.get('cpu_percent', 0)
        if cpu_util < self.target_cpu_utilization * 0.8:  # 80% of target
            logger.warning(f"CPU utilization below target: {cpu_util:.1f}% < {self.target_cpu_utilization}%")
        
        # Check GPU utilization
        for gpu in gpu_data:
            gpu_util = gpu.get('load_percent', 0)
            if gpu_util < self.target_gpu_utilization * 0.8:
                logger.warning(f"GPU {gpu.get('gpu_id')} utilization below target: "
                             f"{gpu_util:.1f}% < {self.target_gpu_utilization}%")
    
    def _save_monitoring_results(self):
        """Save monitoring results to file"""
        if not self.performance_data:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"ray_cluster_performance_{timestamp}.json"
        
        # Calculate summary statistics
        summary = self._calculate_performance_summary()
        
        results = {
            'monitoring_summary': summary,
            'performance_data': self.performance_data,
            'monitoring_duration': time.time() - self.start_time if self.start_time else 0,
            'data_points': len(self.performance_data)
        }
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Performance monitoring results saved to {results_file}")
        
        # Log summary
        logger.info("MONITORING SUMMARY:")
        logger.info(f"  Duration: {summary.get('duration_minutes', 0):.1f} minutes")
        logger.info(f"  Avg CPU: {summary.get('avg_cpu_utilization', 0):.1f}%")
        logger.info(f"  Avg GPU: {summary.get('avg_gpu_utilization', 0):.1f}%")
        logger.info(f"  Peak CPU: {summary.get('peak_cpu_utilization', 0):.1f}%")
        logger.info(f"  Peak GPU: {summary.get('peak_gpu_utilization', 0):.1f}%")
    
    def _calculate_performance_summary(self) -> Dict:
        """Calculate performance summary statistics"""
        if not self.performance_data:
            return {}
        
        # Extract time series data
        cpu_utilizations = []
        gpu_utilizations = []
        memory_utilizations = []
        
        for snapshot in self.performance_data:
            local_data = snapshot.get('local_system', {})
            gpu_data = snapshot.get('gpu_stats', [])
            
            if local_data.get('cpu_percent'):
                cpu_utilizations.append(local_data['cpu_percent'])
            
            if local_data.get('memory_percent'):
                memory_utilizations.append(local_data['memory_percent'])
            
            for gpu in gpu_data:
                if gpu.get('load_percent'):
                    gpu_utilizations.append(gpu['load_percent'])
        
        # Calculate statistics
        summary = {
            'duration_minutes': (time.time() - self.start_time) / 60 if self.start_time else 0,
            'data_points': len(self.performance_data),
            'avg_cpu_utilization': np.mean(cpu_utilizations) if cpu_utilizations else 0,
            'peak_cpu_utilization': np.max(cpu_utilizations) if cpu_utilizations else 0,
            'min_cpu_utilization': np.min(cpu_utilizations) if cpu_utilizations else 0,
            'avg_gpu_utilization': np.mean(gpu_utilizations) if gpu_utilizations else 0,
            'peak_gpu_utilization': np.max(gpu_utilizations) if gpu_utilizations else 0,
            'min_gpu_utilization': np.min(gpu_utilizations) if gpu_utilizations else 0,
            'avg_memory_utilization': np.mean(memory_utilizations) if memory_utilizations else 0,
            'target_cpu_achieved': np.mean(cpu_utilizations) >= self.target_cpu_utilization * 0.8 if cpu_utilizations else False,
            'target_gpu_achieved': np.mean(gpu_utilizations) >= self.target_gpu_utilization * 0.8 if gpu_utilizations else False
        }
        
        return summary


def main():
    """Main monitoring execution"""
    print("RAY CLUSTER PERFORMANCE MONITOR")
    print("="*50)
    print("Real-time monitoring of CPU/GPU utilization")
    print("Target: 75% CPU + GPU usage across cluster")
    print("="*50)
    
    try:
        # Initialize monitor
        monitor = RayClusterMonitor(monitoring_interval=1.0)
        
        # Check if Ray is running
        if not ray.is_initialized():
            print("Warning: Ray is not initialized. Attempting to connect...")
            try:
                ray.init(address='auto')
                print("Connected to Ray cluster")
            except Exception as e:
                print(f"Failed to connect to Ray: {e}")
                print("Please start Ray cluster first using: ./start_ray_cluster.sh")
                return
        
        # Start monitoring
        monitor.start_monitoring()
        
        print("Monitoring started. Press Ctrl+C to stop.")
        print("Performance summary logged every 10 seconds.")
        print("")
        
        # Keep monitoring running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping monitor...")
            monitor.stop_monitoring()
            
    except Exception as e:
        logger.error(f"Monitor failed: {e}")
        raise
    finally:
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()
