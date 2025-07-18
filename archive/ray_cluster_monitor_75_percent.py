#!/usr/bin/env python3
"""
Real-time Ray Cluster Resource Monitor for 75% Utilization Target
Continuously monitors CPU, GPU, vRAM, and Ray task performance
"""

import ray
import time
import psutil
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from collections import deque
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ResourceSnapshot:
    """Single point-in-time resource measurement"""
    timestamp: str
    node_id: str
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    gpu_metrics: List[Dict]
    ray_resources: Dict
    active_tasks: int
    pending_tasks: int

@dataclass
class UtilizationSummary:
    """Summary of resource utilization over time"""
    avg_cpu: float
    max_cpu: float
    avg_memory: float
    max_memory: float
    avg_gpu_util: float
    max_gpu_util: float
    avg_vram_util: float
    max_vram_util: float
    target_achieved: bool
    measurement_count: int
    time_span_minutes: float

class RayClusterMonitor:
    """Real-time Ray cluster resource monitor"""
    
    def __init__(self, target_utilization: float = 75.0, measurement_interval: float = 2.0):
        self.target_utilization = target_utilization
        self.measurement_interval = measurement_interval
        self.snapshots: List[ResourceSnapshot] = []
        self.monitoring = False
        self.start_time = None
        
        # Initialize Ray connection
        if not ray.is_initialized():
            try:
                ray.init(address='auto')
                logger.info("Connected to Ray cluster")
            except Exception as e:
                logger.error(f"Failed to connect to Ray cluster: {e}")
                raise
        
        # Data storage for plotting
        if PLOTTING_AVAILABLE:
            self.cpu_data = deque(maxlen=300)  # Last 10 minutes at 2s intervals
            self.memory_data = deque(maxlen=300)
            self.gpu_data = deque(maxlen=300)
            self.vram_data = deque(maxlen=300)
            self.timestamps = deque(maxlen=300)
    
    def get_gpu_metrics(self) -> List[Dict]:
        """Get current GPU metrics"""
        if not GPU_AVAILABLE:
            return []
        
        gpu_metrics = []
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                gpu_metrics.append({
                    'id': gpu.id,
                    'name': gpu.name,
                    'memory_used_mb': gpu.memoryUsed,
                    'memory_total_mb': gpu.memoryTotal,
                    'memory_util_percent': gpu.memoryUtil * 100,
                    'gpu_util_percent': gpu.load * 100,
                    'temperature_c': gpu.temperature
                })
        except Exception as e:
            logger.warning(f"Failed to get GPU metrics: {e}")
        
        return gpu_metrics
    
    def get_ray_task_info(self) -> Dict:
        """Get Ray task information"""
        try:
            tasks = ray.get_runtime_context().list_tasks()
            active = sum(1 for task in tasks if task.get('state') == 'RUNNING')
            pending = sum(1 for task in tasks if task.get('state') == 'PENDING')
            return {'active': active, 'pending': pending}
        except:
            return {'active': 0, 'pending': 0}
    
    def take_snapshot(self) -> ResourceSnapshot:
        """Take a single resource measurement snapshot"""
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        # GPU metrics
        gpu_metrics = self.get_gpu_metrics()
        
        # Ray cluster info
        try:
            ray_resources = ray.cluster_resources()
            node_id = ray.get_runtime_context().get_node_id()[:8]
        except:
            ray_resources = {}
            node_id = "unknown"
        
        # Task info
        task_info = self.get_ray_task_info()
        
        snapshot = ResourceSnapshot(
            timestamp=datetime.now().isoformat(),
            node_id=node_id,
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_gb=memory.used / (1024**3),
            memory_total_gb=memory.total / (1024**3),
            gpu_metrics=gpu_metrics,
            ray_resources=ray_resources,
            active_tasks=task_info['active'],
            pending_tasks=task_info['pending']
        )
        
        return snapshot
    
    def calculate_utilization_summary(self, time_window_minutes: Optional[float] = None) -> UtilizationSummary:
        """Calculate utilization summary over specified time window"""
        if not self.snapshots:
            return UtilizationSummary(0, 0, 0, 0, 0, 0, 0, 0, False, 0, 0)
        
        # Filter snapshots by time window
        snapshots = self.snapshots
        if time_window_minutes:
            cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
            snapshots = [s for s in self.snapshots 
                        if datetime.fromisoformat(s.timestamp) > cutoff_time]
        
        if not snapshots:
            return UtilizationSummary(0, 0, 0, 0, 0, 0, 0, 0, False, 0, 0)
        
        # CPU and memory stats
        cpu_values = [s.cpu_percent for s in snapshots]
        memory_values = [s.memory_percent for s in snapshots]
        
        avg_cpu = sum(cpu_values) / len(cpu_values)
        max_cpu = max(cpu_values)
        avg_memory = sum(memory_values) / len(memory_values)
        max_memory = max(memory_values)
        
        # GPU stats
        gpu_util_values = []
        vram_util_values = []
        
        for snapshot in snapshots:
            for gpu in snapshot.gpu_metrics:
                gpu_util_values.append(gpu['gpu_util_percent'])
                vram_util_values.append(gpu['memory_util_percent'])
        
        avg_gpu = sum(gpu_util_values) / len(gpu_util_values) if gpu_util_values else 0
        max_gpu = max(gpu_util_values) if gpu_util_values else 0
        avg_vram = sum(vram_util_values) / len(vram_util_values) if vram_util_values else 0
        max_vram = max(vram_util_values) if vram_util_values else 0
        
        # Check if target is achieved
        target_achieved = (avg_cpu >= self.target_utilization * 0.9 and  # 90% of target
                          avg_gpu >= self.target_utilization * 0.9 and
                          avg_vram >= self.target_utilization * 0.9)
        
        # Time span
        first_time = datetime.fromisoformat(snapshots[0].timestamp)
        last_time = datetime.fromisoformat(snapshots[-1].timestamp)
        time_span = (last_time - first_time).total_seconds() / 60
        
        return UtilizationSummary(
            avg_cpu=avg_cpu,
            max_cpu=max_cpu,
            avg_memory=avg_memory,
            max_memory=max_memory,
            avg_gpu_util=avg_gpu,
            max_gpu_util=max_gpu,
            avg_vram_util=avg_vram,
            max_vram_util=max_vram,
            target_achieved=target_achieved,
            measurement_count=len(snapshots),
            time_span_minutes=time_span
        )
    
    def print_real_time_status(self, snapshot: ResourceSnapshot):
        """Print real-time status to console"""
        print(f"\r🔄 [{snapshot.timestamp[11:19]}] ", end="")
        print(f"CPU: {snapshot.cpu_percent:5.1f}% | ", end="")
        print(f"MEM: {snapshot.memory_percent:5.1f}% | ", end="")
        
        if snapshot.gpu_metrics:
            for i, gpu in enumerate(snapshot.gpu_metrics):
                print(f"GPU{i}: {gpu['gpu_util_percent']:5.1f}% | ", end="")
                print(f"VRAM{i}: {gpu['memory_util_percent']:5.1f}% | ", end="")
        
        print(f"Tasks: {snapshot.active_tasks:3d}A/{snapshot.pending_tasks:3d}P", end="", flush=True)
    
    def print_summary_report(self, time_window_minutes: float = 5.0):
        """Print detailed summary report"""
        summary = self.calculate_utilization_summary(time_window_minutes)
        
        print(f"\n" + "=" * 80)
        print(f"📊 RESOURCE UTILIZATION SUMMARY (Last {time_window_minutes:.1f} minutes)")
        print("=" * 80)
        
        status_emoji = "🎯" if summary.target_achieved else "⚠️"
        target_status = "ACHIEVED" if summary.target_achieved else "BELOW TARGET"
        
        print(f"{status_emoji} Target Status: {target_status} (Target: {self.target_utilization}%)")
        print(f"📏 Measurements: {summary.measurement_count} over {summary.time_span_minutes:.1f} minutes")
        print()
        
        print(f"🖥️  CPU Utilization:")
        print(f"   • Average: {summary.avg_cpu:6.1f}% {'✅' if summary.avg_cpu >= self.target_utilization * 0.9 else '⚠️'}")
        print(f"   • Maximum: {summary.max_cpu:6.1f}%")
        print()
        
        print(f"💾 Memory Utilization:")
        print(f"   • Average: {summary.avg_memory:6.1f}%")
        print(f"   • Maximum: {summary.max_memory:6.1f}%")
        print()
        
        if summary.avg_gpu_util > 0:
            print(f"🎮 GPU Utilization:")
            print(f"   • Average: {summary.avg_gpu_util:6.1f}% {'✅' if summary.avg_gpu_util >= self.target_utilization * 0.9 else '⚠️'}")
            print(f"   • Maximum: {summary.max_gpu_util:6.1f}%")
            print()
            
            print(f"🎯 VRAM Utilization:")
            print(f"   • Average: {summary.avg_vram_util:6.1f}% {'✅' if summary.avg_vram_util >= self.target_utilization * 0.9 else '⚠️'}")
            print(f"   • Maximum: {summary.max_vram_util:6.1f}%")
        else:
            print(f"🎮 GPU: No GPU metrics available")
        
        print("=" * 80)
        return summary
    
    def save_data_to_file(self, filename: Optional[str] = None):
        """Save monitoring data to JSON file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ray_cluster_monitor_{timestamp}.json"
        
        data = {
            'monitoring_session': {
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'end_time': datetime.now().isoformat(),
                'target_utilization': self.target_utilization,
                'measurement_interval': self.measurement_interval,
                'total_snapshots': len(self.snapshots)
            },
            'utilization_summary': asdict(self.calculate_utilization_summary()),
            'snapshots': [asdict(s) for s in self.snapshots[-100:]]  # Last 100 snapshots
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Monitoring data saved to {filename}")
        return filename
    
    def start_monitoring(self, duration_minutes: Optional[float] = None):
        """Start continuous monitoring"""
        self.monitoring = True
        self.start_time = datetime.now()
        
        print(f"🚀 Starting Ray cluster monitoring (Target: {self.target_utilization}%)")
        print(f"📊 Measurement interval: {self.measurement_interval}s")
        if duration_minutes:
            print(f"⏱️  Duration: {duration_minutes} minutes")
        print("Press Ctrl+C to stop monitoring")
        print()
        
        snapshot_count = 0
        last_summary_time = time.time()
        
        try:
            while self.monitoring:
                # Take snapshot
                snapshot = self.take_snapshot()
                self.snapshots.append(snapshot)
                snapshot_count += 1
                
                # Print real-time status
                self.print_real_time_status(snapshot)
                
                # Update plotting data
                if PLOTTING_AVAILABLE:
                    self.timestamps.append(datetime.now())
                    self.cpu_data.append(snapshot.cpu_percent)
                    self.memory_data.append(snapshot.memory_percent)
                    
                    if snapshot.gpu_metrics:
                        self.gpu_data.append(snapshot.gpu_metrics[0]['gpu_util_percent'])
                        self.vram_data.append(snapshot.gpu_metrics[0]['memory_util_percent'])
                    else:
                        self.gpu_data.append(0)
                        self.vram_data.append(0)
                
                # Print summary every 30 seconds
                if time.time() - last_summary_time >= 30:
                    summary = self.print_summary_report(time_window_minutes=2.0)
                    last_summary_time = time.time()
                
                # Check duration limit
                if duration_minutes:
                    elapsed = (datetime.now() - self.start_time).total_seconds() / 60
                    if elapsed >= duration_minutes:
                        break
                
                time.sleep(self.measurement_interval)
                
        except KeyboardInterrupt:
            print(f"\n\n⏹️  Monitoring stopped by user")
        
        # Final summary
        self.monitoring = False
        final_summary = self.print_summary_report()
        
        # Save data
        filename = self.save_data_to_file()
        
        print(f"\n🏁 Monitoring session complete:")
        print(f"   • Total snapshots: {len(self.snapshots)}")
        print(f"   • Data saved to: {filename}")
        print(f"   • Target achieved: {'YES' if final_summary.target_achieved else 'NO'}")
        
        return final_summary

def main():
    """Main monitoring function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ray Cluster Resource Monitor for 75% Utilization")
    parser.add_argument("--target", type=float, default=75.0, help="Target utilization percentage (default: 75)")
    parser.add_argument("--interval", type=float, default=2.0, help="Measurement interval in seconds (default: 2)")
    parser.add_argument("--duration", type=float, help="Monitoring duration in minutes (default: unlimited)")
    parser.add_argument("--output", type=str, help="Output filename for monitoring data")
    
    args = parser.parse_args()
    
    # Create monitor
    monitor = RayClusterMonitor(
        target_utilization=args.target,
        measurement_interval=args.interval
    )
    
    # Start monitoring
    try:
        summary = monitor.start_monitoring(duration_minutes=args.duration)
        
        if args.output:
            monitor.save_data_to_file(args.output)
            
    except Exception as e:
        logger.error(f"Monitoring failed: {e}")
        raise

if __name__ == "__main__":
    main()
