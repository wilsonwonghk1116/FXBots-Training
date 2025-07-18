#!/usr/bin/env python3
"""
Ultimate Ray-Distributed Kelly Monte Carlo Trading Bot System
Designed to maximize CPU/GPU utilization across 2 PCs to 75%+

Hardware Target:
- PC1: Xeon CPU + RTX 3090 (24GB VRAM)
- PC2: i9 CPU + RTX 3070 (8GB VRAM)
- Ray cluster with Python 3.12.2

This script will spawn multiple Ray actors to saturate all available resources.
"""

import ray
import numpy as np
import pandas as pd
import torch
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from dataclasses import dataclass, asdict
import psutil
import GPUtil

# Import our bot classes
from kelly_monte_bot import (
    KellyMonteBot, BotFleetManager, TradingParameters,
    DataManager, MonteCarloEngine, KellyCalculator,
    ScenarioResult, KellyEstimates
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ClusterConfig:
    """Configuration for Ray cluster deployment"""
    total_bots: int = 4000  # Increased for maximum load
    monte_carlo_scenarios: int = 200000  # Massive MC scenarios for GPU saturation
    max_cpu_actors_per_node: int = 32  # High CPU actor count
    max_gpu_actors_per_node: int = 8   # Multiple GPU actors per GPU
    cpu_cores_per_actor: float = 0.5   # Allow oversubscription for I/O bound tasks
    gpu_memory_per_actor: float = 0.8  # 80% GPU memory per actor
    batch_size_multiplier: int = 4     # Multiply batch sizes for maximum throughput

@ray.remote(num_cpus=1, num_gpus=0)
class ResourceMonitor:
    """Monitor cluster resource utilization"""
    
    def __init__(self):
        self.start_time = time.time()
        self.metrics_history = []
    
    def get_system_metrics(self):
        """Get current system resource utilization"""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        
        # GPU metrics
        gpu_metrics = []
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                gpu_metrics.append({
                    'id': gpu.id,
                    'name': gpu.name,
                    'utilization': gpu.load * 100,
                    'memory_used': gpu.memoryUsed,
                    'memory_total': gpu.memoryTotal,
                    'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                    'temperature': gpu.temperature
                })
        except Exception as e:
            logger.warning(f"Could not get GPU metrics: {e}")
        
        metrics = {
            'timestamp': time.time(),
            'cpu_percent': cpu_percent,
            'cpu_count': cpu_count,
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / (1024**3),
            'gpu_metrics': gpu_metrics,
            'uptime': time.time() - self.start_time
        }
        
        self.metrics_history.append(metrics)
        return metrics
    
    def get_average_utilization(self):
        """Get average utilization metrics"""
        if not self.metrics_history:
            return {}
        
        recent_metrics = self.metrics_history[-10:]  # Last 10 measurements
        
        avg_cpu = np.mean([m['cpu_percent'] for m in recent_metrics])
        avg_memory = np.mean([m['memory_percent'] for m in recent_metrics])
        
        avg_gpu_util = 0
        avg_gpu_memory = 0
        if recent_metrics[0]['gpu_metrics']:
            gpu_utils = []
            gpu_memories = []
            for m in recent_metrics:
                for gpu in m['gpu_metrics']:
                    gpu_utils.append(gpu['utilization'])
                    gpu_memories.append(gpu['memory_percent'])
            avg_gpu_util = np.mean(gpu_utils) if gpu_utils else 0
            avg_gpu_memory = np.mean(gpu_memories) if gpu_memories else 0
        
        return {
            'avg_cpu_percent': avg_cpu,
            'avg_memory_percent': avg_memory,
            'avg_gpu_utilization': avg_gpu_util,
            'avg_gpu_memory_percent': avg_gpu_memory,
            'sample_count': len(recent_metrics)
        }

@ray.remote(num_cpus=0.5, num_gpus=0.8)
class HeavyGPUWorker:
    """Heavy GPU worker for maximum GPU utilization"""
    
    def __init__(self, worker_id: int, scenarios_per_batch: int = 200000):
        self.worker_id = worker_id
        self.scenarios_per_batch = scenarios_per_batch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.params = TradingParameters(monte_carlo_scenarios=scenarios_per_batch)
        self.monte_carlo = MonteCarloEngine(self.params)
        self.kelly_calc = KellyCalculator(self.params)
        
        # Pre-allocate large GPU tensors for maximum VRAM utilization
        if self.device.type == 'cuda':
            self._preallocate_gpu_memory()
        
        logger.info(f"HeavyGPUWorker {worker_id} initialized on {self.device} with {scenarios_per_batch} scenarios")
    
    def _preallocate_gpu_memory(self):
        """Pre-allocate GPU memory for maximum utilization"""
        try:
            # Allocate large tensors to fill GPU memory
            memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"GPU has {memory_gb:.1f}GB total memory")
            
            # Pre-allocate multiple large tensors
            self.gpu_tensors = []
            tensor_size = min(50000000, int(memory_gb * 50000000))  # Scale with GPU memory
            
            for i in range(3):  # Multiple large tensors
                tensor = torch.randn(tensor_size, device=self.device, dtype=torch.float32)
                self.gpu_tensors.append(tensor)
            
            logger.info(f"Pre-allocated {len(self.gpu_tensors)} large GPU tensors")
        except Exception as e:
            logger.warning(f"Could not pre-allocate GPU memory: {e}")
    
    def process_massive_scenarios(self, 
                                 current_price: float,
                                 return_params: Dict,
                                 entry_signal: str,
                                 multiplier: int = 1) -> Dict:
        """Process massive number of Monte Carlo scenarios"""
        total_scenarios = self.scenarios_per_batch * multiplier
        
        start_time = time.time()
        
        # Generate scenarios with maximum GPU utilization
        scenarios = self.monte_carlo.generate_scenarios(
            current_price=current_price,
            return_params=return_params,
            entry_signal=entry_signal,
            n_scenarios=total_scenarios
        )
        
        # Calculate Kelly estimates
        kelly_estimates = self.kelly_calc.estimate_parameters(scenarios)
        
        processing_time = time.time() - start_time
        
        result = {
            'worker_id': self.worker_id,
            'scenarios_processed': len(scenarios),
            'processing_time': processing_time,
            'scenarios_per_second': len(scenarios) / processing_time if processing_time > 0 else 0,
            'kelly_fraction': kelly_estimates.kelly_fraction,
            'win_probability': kelly_estimates.win_probability,
            'payoff_ratio': kelly_estimates.payoff_ratio,
            'device': str(self.device)
        }
        
        return result

@ray.remote(num_cpus=1, num_gpus=0)
class CPUIntensiveWorker:
    """CPU-intensive worker for maximum CPU utilization"""
    
    def __init__(self, worker_id: int, n_bots_per_worker: int = 100):
        self.worker_id = worker_id
        self.n_bots_per_worker = n_bots_per_worker
        
        # Initialize bot fleet
        self.params = TradingParameters(monte_carlo_scenarios=25000)  # Moderate for CPU
        self.bots = []
        
        for i in range(n_bots_per_worker):
            bot = KellyMonteBot(
                bot_id=f"{worker_id}_{i}",
                initial_equity=100000.0,
                params=self.params
            )
            self.bots.append(bot)
        
        # Initialize all bots
        for bot in self.bots:
            bot.initialize("EURUSD")
        
        logger.info(f"CPUIntensiveWorker {worker_id} initialized with {n_bots_per_worker} bots")
    
    def process_bot_decisions(self, 
                             current_price: float,
                             market_data: Dict,
                             timestamp: str) -> List[Dict]:
        """Process trading decisions for all bots in this worker"""
        
        # Convert market data back to pandas Series
        market_series = pd.Series(market_data)
        pd_timestamp = pd.Timestamp(timestamp)
        
        decisions = []
        start_time = time.time()
        
        # Use all available CPU cores within this worker
        max_workers = mp.cpu_count()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    bot.make_trading_decision,
                    current_price,
                    market_series,
                    pd_timestamp
                ) for bot in self.bots
            ]
            
            for future in as_completed(futures):
                try:
                    decision = future.result()
                    if decision:
                        decisions.append(decision)
                except Exception as e:
                    logger.error(f"Bot decision failed: {e}")
        
        processing_time = time.time() - start_time
        
        return {
            'worker_id': self.worker_id,
            'decisions_made': len(decisions),
            'bots_processed': len(self.bots),
            'processing_time': processing_time,
            'decisions': decisions
        }

@ray.remote(num_cpus=2, num_gpus=0)
class FleetCoordinator:
    """Coordinates the entire bot fleet across all workers"""
    
    def __init__(self, config: ClusterConfig):
        self.config = config
        self.start_time = time.time()
        self.data_manager = DataManager()
        
        # Load market data
        self.market_data = self.data_manager.load_h1_data("EURUSD")
        self.return_params = self.data_manager.get_return_distribution_params()
        
        logger.info(f"FleetCoordinator initialized with {len(self.market_data)} data points")
    
    def get_current_market_data(self, index: int) -> tuple:
        """Get market data for given index"""
        if index >= len(self.market_data):
            index = index % len(self.market_data)
        
        row = self.market_data.iloc[index]
        current_price = row['close']
        timestamp = row.name
        
        # Convert to dictionary for Ray serialization
        market_dict = {
            'close': row['close'],
            'sma_20': row.get('sma_20', row['close']),
            'sma_50': row.get('sma_50', row['close']),
            'volume': row.get('volume', 1000)
        }
        
        return current_price, market_dict, str(timestamp)
    
    def aggregate_results(self, 
                         gpu_results: List[Dict],
                         cpu_results: List[Dict]) -> Dict:
        """Aggregate results from all workers"""
        
        # GPU metrics
        total_scenarios = sum(r['scenarios_processed'] for r in gpu_results)
        total_gpu_time = sum(r['processing_time'] for r in gpu_results)
        avg_scenarios_per_sec = np.mean([r['scenarios_per_second'] for r in gpu_results])
        
        # CPU metrics
        total_decisions = sum(r['decisions_made'] for r in cpu_results)
        total_cpu_time = sum(r['processing_time'] for r in cpu_results)
        total_bots_processed = sum(r['bots_processed'] for r in cpu_results)
        
        return {
            'timestamp': time.time(),
            'runtime': time.time() - self.start_time,
            'gpu_metrics': {
                'total_scenarios_processed': total_scenarios,
                'total_gpu_processing_time': total_gpu_time,
                'average_scenarios_per_second': avg_scenarios_per_sec,
                'gpu_workers': len(gpu_results)
            },
            'cpu_metrics': {
                'total_decisions_made': total_decisions,
                'total_cpu_processing_time': total_cpu_time,
                'total_bots_processed': total_bots_processed,
                'cpu_workers': len(cpu_results)
            },
            'throughput': {
                'scenarios_per_minute': (total_scenarios / (time.time() - self.start_time)) * 60,
                'decisions_per_minute': (total_decisions / (time.time() - self.start_time)) * 60
            }
        }

def setup_ray_cluster():
    """Setup Ray cluster with resource detection"""
    if not ray.is_initialized():
        ray.init(address='auto')  # Connect to existing cluster
    
    # Get cluster information
    cluster_resources = ray.cluster_resources()
    nodes = ray.nodes()
    
    logger.info("Ray Cluster Status:")
    logger.info(f"Total CPUs: {cluster_resources.get('CPU', 0)}")
    logger.info(f"Total GPUs: {cluster_resources.get('GPU', 0)}")
    logger.info(f"Total Memory: {cluster_resources.get('memory', 0) / (1024**3):.1f} GB")
    logger.info(f"Number of nodes: {len(nodes)}")
    
    for i, node in enumerate(nodes):
        logger.info(f"Node {i}: {node['Resources']}")
    
    return cluster_resources

def create_worker_actors(config: ClusterConfig, cluster_resources: Dict):
    """Create and distribute worker actors across the cluster"""
    
    total_cpus = int(cluster_resources.get('CPU', 0))
    total_gpus = int(cluster_resources.get('GPU', 0))
    
    logger.info(f"Creating workers for {total_cpus} CPUs and {total_gpus} GPUs")
    
    # Create GPU workers - multiple per GPU for maximum utilization
    gpu_workers = []
    if total_gpus > 0:
        workers_per_gpu = config.max_gpu_actors_per_node
        total_gpu_workers = total_gpus * workers_per_gpu
        
        for i in range(total_gpu_workers):
            worker = HeavyGPUWorker.remote(
                worker_id=i,
                scenarios_per_batch=config.monte_carlo_scenarios
            )
            gpu_workers.append(worker)
        
        logger.info(f"Created {len(gpu_workers)} GPU workers")
    
    # Create CPU workers - maximize CPU utilization
    cpu_workers = []
    if total_cpus > 0:
        # Create many CPU workers for maximum parallel processing
        num_cpu_workers = min(config.max_cpu_actors_per_node, total_cpus * 2)  # Allow oversubscription
        bots_per_worker = max(1, config.total_bots // num_cpu_workers)
        
        for i in range(num_cpu_workers):
            worker = CPUIntensiveWorker.remote(
                worker_id=i,
                n_bots_per_worker=bots_per_worker
            )
            cpu_workers.append(worker)
        
        logger.info(f"Created {len(cpu_workers)} CPU workers with {bots_per_worker} bots each")
    
    return gpu_workers, cpu_workers

async def run_ultimate_kelly_fleet():
    """Run the ultimate Kelly Monte Carlo fleet with maximum resource utilization"""
    
    # Configuration for maximum performance
    config = ClusterConfig(
        total_bots=4000,
        monte_carlo_scenarios=200000,
        max_cpu_actors_per_node=32,
        max_gpu_actors_per_node=8,
        batch_size_multiplier=4
    )
    
    logger.info("Starting Ultimate Kelly Monte Carlo Fleet")
    logger.info(f"Target: {config.total_bots} bots, {config.monte_carlo_scenarios} MC scenarios")
    
    # Setup Ray cluster
    cluster_resources = setup_ray_cluster()
    
    # Create resource monitor
    monitor = ResourceMonitor.remote()
    
    # Create fleet coordinator
    coordinator = FleetCoordinator.remote(config)
    
    # Create worker actors
    gpu_workers, cpu_workers = create_worker_actors(config, cluster_resources)
    
    if not gpu_workers and not cpu_workers:
        logger.error("No workers created! Check cluster resources.")
        return
    
    # Start resource monitoring
    monitor_future = monitor.get_system_metrics.remote()
    
    # Run simulation for multiple iterations
    total_iterations = 1000  # Run many iterations for sustained load
    results = []
    
    logger.info(f"Starting {total_iterations} iterations of maximum load testing...")
    
    for iteration in range(total_iterations):
        iteration_start = time.time()
        
        # Get current market data
        current_price, market_dict, timestamp = ray.get(
            coordinator.get_current_market_data.remote(iteration)
        )
        
        # Launch GPU work - multiple scenarios per worker for maximum load
        gpu_futures = []
        if gpu_workers:
            for worker in gpu_workers:
                future = worker.process_massive_scenarios.remote(
                    current_price=current_price,
                    return_params=ray.get(coordinator.return_params.remote()),
                    entry_signal='BUY' if iteration % 2 == 0 else 'SELL',
                    multiplier=config.batch_size_multiplier
                )
                gpu_futures.append(future)
        
        # Launch CPU work - decision making for all bots
        cpu_futures = []
        if cpu_workers:
            for worker in cpu_workers:
                future = worker.process_bot_decisions.remote(
                    current_price=current_price,
                    market_data=market_dict,
                    timestamp=timestamp
                )
                cpu_futures.append(future)
        
        # Wait for all work to complete
        all_futures = gpu_futures + cpu_futures
        
        if all_futures:
            completed_results = ray.get(all_futures)
            
            # Separate GPU and CPU results
            gpu_results = completed_results[:len(gpu_futures)]
            cpu_results = completed_results[len(gpu_futures):]
            
            # Aggregate results
            iteration_summary = ray.get(
                coordinator.aggregate_results.remote(gpu_results, cpu_results)
            )
            
            iteration_time = time.time() - iteration_start
            iteration_summary['iteration'] = iteration
            iteration_summary['iteration_time'] = iteration_time
            
            results.append(iteration_summary)
            
            # Log progress every 10 iterations
            if iteration % 10 == 0:
                logger.info(f"Iteration {iteration}/{total_iterations} completed in {iteration_time:.2f}s")
                logger.info(f"GPU: {iteration_summary['gpu_metrics']['total_scenarios_processed']:,} scenarios")
                logger.info(f"CPU: {iteration_summary['cpu_metrics']['total_decisions_made']:,} decisions")
                
                # Get resource utilization
                if iteration % 50 == 0:  # Every 50 iterations
                    try:
                        current_metrics = ray.get(monitor.get_system_metrics.remote())
                        avg_metrics = ray.get(monitor.get_average_utilization.remote())
                        
                        logger.info("=== RESOURCE UTILIZATION ===")
                        logger.info(f"CPU: {avg_metrics.get('avg_cpu_percent', 0):.1f}%")
                        logger.info(f"GPU: {avg_metrics.get('avg_gpu_utilization', 0):.1f}%")
                        logger.info(f"GPU Memory: {avg_metrics.get('avg_gpu_memory_percent', 0):.1f}%")
                        logger.info(f"System Memory: {avg_metrics.get('avg_memory_percent', 0):.1f}%")
                        logger.info("============================")
                    except Exception as e:
                        logger.warning(f"Could not get resource metrics: {e}")
    
    # Final aggregation and results
    logger.info("Simulation completed! Generating final report...")
    
    # Calculate final statistics
    total_scenarios = sum(r['gpu_metrics']['total_scenarios_processed'] for r in results)
    total_decisions = sum(r['cpu_metrics']['total_decisions_made'] for r in results)
    total_runtime = results[-1]['runtime'] if results else 0
    
    # Get final resource utilization
    final_metrics = ray.get(monitor.get_average_utilization.remote())
    
    final_report = {
        'configuration': asdict(config),
        'cluster_resources': cluster_resources,
        'performance_summary': {
            'total_iterations': len(results),
            'total_scenarios_processed': total_scenarios,
            'total_decisions_made': total_decisions,
            'total_runtime_seconds': total_runtime,
            'average_scenarios_per_second': total_scenarios / total_runtime if total_runtime > 0 else 0,
            'average_decisions_per_second': total_decisions / total_runtime if total_runtime > 0 else 0
        },
        'resource_utilization': final_metrics,
        'worker_counts': {
            'gpu_workers': len(gpu_workers),
            'cpu_workers': len(cpu_workers)
        },
        'detailed_results': results[-10:],  # Last 10 iterations
        'timestamp': datetime.now().isoformat()
    }
    
    # Save results
    filename = f"ray_ultimate_kelly_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(final_report, f, indent=2, default=str)
    
    logger.info("=== FINAL RESULTS ===")
    logger.info(f"Total scenarios processed: {total_scenarios:,}")
    logger.info(f"Total decisions made: {total_decisions:,}")
    logger.info(f"Runtime: {total_runtime:.1f} seconds")
    logger.info(f"Average CPU utilization: {final_metrics.get('avg_cpu_percent', 0):.1f}%")
    logger.info(f"Average GPU utilization: {final_metrics.get('avg_gpu_utilization', 0):.1f}%")
    logger.info(f"Results saved to: {filename}")
    logger.info("====================")
    
    return final_report

if __name__ == "__main__":
    import asyncio
    
    logger.info("Ultimate Kelly Monte Carlo Ray Cluster - Maximum Resource Utilization")
    logger.info("Target: 75%+ CPU and GPU utilization across 2 PCs")
    
    try:
        # Run the ultimate fleet
        results = asyncio.run(run_ultimate_kelly_fleet())
        
        # Check if we achieved target utilization
        final_utilization = results.get('resource_utilization', {})
        cpu_util = final_utilization.get('avg_cpu_percent', 0)
        gpu_util = final_utilization.get('avg_gpu_utilization', 0)
        
        logger.info("\n=== UTILIZATION TARGET CHECK ===")
        logger.info(f"CPU Utilization: {cpu_util:.1f}% (Target: 75%+)")
        logger.info(f"GPU Utilization: {gpu_util:.1f}% (Target: 75%+)")
        
        if cpu_util >= 75 and gpu_util >= 75:
            logger.info("✅ SUCCESS: Achieved 75%+ utilization on both CPU and GPU!")
        elif cpu_util >= 75:
            logger.info("✅ CPU target achieved, GPU needs optimization")
        elif gpu_util >= 75:
            logger.info("✅ GPU target achieved, CPU needs optimization")
        else:
            logger.info("❌ Both CPU and GPU below 75% - check resource allocation")
        
        logger.info("================================")
        
    except Exception as e:
        logger.error(f"Ultimate fleet execution failed: {e}")
        raise
    finally:
        if ray.is_initialized():
            ray.shutdown()
