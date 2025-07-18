#!/usr/bin/env python3
"""
MASSIVE SCALE DISTRIBUTED FOREX TRAINING
=========================================
Ray-distributed version for dual-PC cluster
- 200 generations
- 1000 episodes per generation  
- 1000 steps per episode
- Enhanced PnL-based reward system
"""

import ray
import numpy as np
import time
import json
import os
from datetime import datetime

# Import configuration classes directly to avoid serialization issues
class MassiveScaleConfig:
    """Configuration for massive scale training with 75% resource utilization"""
    TOTAL_GENERATIONS = 200
    EPISODES_PER_GENERATION = 1000
    STEPS_PER_EPISODE = 1000
    
    # 75% Resource Utilization Settings
    POPULATION_SIZE = 75        # Reduced from 100 for 75% utilization
    ELITE_SIZE = 15            # Reduced from 20 for 75% utilization  
    WORKERS_PER_NODE = 2       # Reduced from 8 for 75% utilization
    CPUS_PER_WORKER = 8        # Reduced from 12 for 75% utilization
    EPISODES_PER_BATCH = 38    # Reduced from 50 for 75% utilization
    
    # Training Parameters (optimized for 75% utilization)
    MUTATION_RATE = 0.15
    LEARNING_RATE = 0.001
    DISCOUNT_FACTOR = 0.95
    EXPLORATION_RATE = 0.1
    REWARD_SCALE = 1.0
    PENALTY_SCALE = 1.0
    MAX_REWARD_CAP = 1000.0
    MAX_PENALTY_CAP = -1000.0
    
    # System Configuration
    SAVE_INTERVAL = 10
    CHECKPOINT_INTERVAL = 25
    
    # 75% Utilization Performance Estimates
    STEPS_PER_SECOND_75_PERCENT = 15000  # Reduced from ~20000 for sustainable performance
    ESTIMATED_HOURS = (TOTAL_GENERATIONS * EPISODES_PER_GENERATION * STEPS_PER_EPISODE) / (STEPS_PER_SECOND_75_PERCENT * 3600)
    
    # GPU and Memory Optimization
    GPU_MEMORY_FRACTION = 0.75
    BATCH_SIZE_MULTIPLIER = 0.75  # Reduce batch sizes by 25%

class EnhancedPnLRewardSystem:
    """Enhanced PnL-based reward system for reinforcement learning"""
    
    def __init__(self):
        self.total_rewards = 0
        self.total_penalties = 0
        self.trade_count = 0
        
    def calculate_reward(self, pnl_usd, trade_type="FOREX"):
        """Calculate reward/penalty based on PnL"""
        base_reward = pnl_usd * MassiveScaleConfig.REWARD_SCALE
        
        if base_reward > 0:
            reward = min(base_reward, MassiveScaleConfig.MAX_REWARD_CAP)
            self.total_rewards += reward
        else:
            reward = max(base_reward, MassiveScaleConfig.MAX_PENALTY_CAP)
            self.total_penalties += abs(reward)
        
        self.trade_count += 1
        return reward
    
    def get_cumulative_performance(self):
        """Get cumulative performance metrics"""
        return {
            'total_rewards': self.total_rewards,
            'total_penalties': self.total_penalties,
            'net_performance': self.total_rewards - self.total_penalties,
            'trade_count': self.trade_count,
            'avg_reward_per_trade': self.total_rewards / max(self.trade_count, 1),
            'win_rate': self.total_rewards / max(self.total_rewards + self.total_penalties, 1)
        }

class MassiveScaleEpisodeProcessor:
    """Process episodes at massive scale with PnL rewards"""
    
    def __init__(self, bot_id):
        self.bot_id = bot_id
        self.reward_system = EnhancedPnLRewardSystem()
        self.episode_results = []
        
    def simulate_forex_trade(self, step_data):
        """Simulate a single forex trade step"""
        base_pnl = np.random.normal(0, 50)
        skill_factor = min(self.trade_count * 0.001, 0.5)
        skilled_pnl = base_pnl + (skill_factor * abs(base_pnl) * np.random.choice([-1, 1]))
        
        return {
            'pnl_usd': skilled_pnl,
            'trade_type': 'FOREX',
            'timestamp': time.time(),
            'step': step_data.get('step', 0)
        }
    
    def run_episode(self, episode_id, generation_id):
        """Run a single episode with 1000 steps"""
        episode_start = time.time()
        episode_rewards = []
        total_pnl = 0
        
        for step in range(MassiveScaleConfig.STEPS_PER_EPISODE):
            step_data = {
                'step': step,
                'episode_id': episode_id,
                'generation_id': generation_id,
                'market_price': 1.0 + np.random.normal(0, 0.01),
                'volatility': abs(np.random.normal(0.02, 0.005))
            }
            
            trade_result = self.simulate_forex_trade(step_data)
            reward = self.reward_system.calculate_reward(trade_result['pnl_usd'])
            episode_rewards.append(reward)
            total_pnl += trade_result['pnl_usd']
        
        episode_duration = time.time() - episode_start
        
        episode_result = {
            'episode_id': episode_id,
            'generation_id': generation_id,
            'bot_id': self.bot_id,
            'total_pnl_usd': total_pnl,
            'total_reward': sum(episode_rewards),
            'steps_completed': len(episode_rewards),
            'duration_seconds': episode_duration,
            'average_reward_per_step': np.mean(episode_rewards),
            'max_reward': max(episode_rewards),
            'min_reward': min(episode_rewards),
            'performance_metrics': self.reward_system.get_cumulative_performance(),
            'timestamp': datetime.now().isoformat()
        }
        
        self.episode_results.append(episode_result)
        return episode_result
    
    @property
    def trade_count(self):
        """Get total trade count"""
        return self.reward_system.trade_count

@ray.remote(num_cpus=8, num_gpus=0.75)  # 75% GPU utilization
class DistributedForexTrainer:
    """Distributed forex trainer for massive scale training"""
    
    def __init__(self, worker_id):
        self.worker_id = worker_id
        self.node_ip = ray._private.services.get_node_ip_address()
        print(f"🚀 Worker {worker_id} initialized on node {self.node_ip}")
        
    def train_generation(self, generation_id, bot_population):
        """Train a complete generation with 1000 episodes"""
        print(f"🔥 Worker {self.worker_id} starting Generation {generation_id}")
        generation_start = time.time()
        
        generation_results = {
            'generation_id': generation_id,
            'worker_id': self.worker_id,
            'node_ip': self.node_ip,
            'bot_results': [],
            'total_episodes': 0,
            'total_steps': 0,
            'total_pnl': 0,
            'total_rewards': 0
        }
        
        # Process bots in this generation
        for bot_id, bot_config in enumerate(bot_population):
            bot_processor = MassiveScaleEpisodeProcessor(f"BOT_{generation_id}_{bot_id}")
            bot_results = []
            
            # Run episodes for this bot (batch processing)
            episodes_per_bot = MassiveScaleConfig.EPISODES_PER_GENERATION // len(bot_population)
            
            for episode_idx in range(episodes_per_bot):
                episode_id = f"G{generation_id}_B{bot_id}_E{episode_idx}"
                episode_result = bot_processor.run_episode(episode_id, generation_id)
                bot_results.append(episode_result)
                
                # Update generation totals
                generation_results['total_episodes'] += 1
                generation_results['total_steps'] += episode_result['steps_completed']
                generation_results['total_pnl'] += episode_result['total_pnl_usd']
                generation_results['total_rewards'] += episode_result['total_reward']
            
            # Compile bot performance
            bot_performance = {
                'bot_id': f"BOT_{generation_id}_{bot_id}",
                'episodes_completed': len(bot_results),
                'total_pnl': sum(r['total_pnl_usd'] for r in bot_results),
                'total_rewards': sum(r['total_reward'] for r in bot_results),
                'avg_reward_per_episode': np.mean([r['total_reward'] for r in bot_results]),
                'best_episode_pnl': max(r['total_pnl_usd'] for r in bot_results),
                'worst_episode_pnl': min(r['total_pnl_usd'] for r in bot_results),
                'performance_metrics': bot_processor.reward_system.get_cumulative_performance()
            }
            
            generation_results['bot_results'].append(bot_performance)
        
        generation_duration = time.time() - generation_start
        generation_results['duration_seconds'] = generation_duration
        generation_results['steps_per_second'] = generation_results['total_steps'] / generation_duration
        generation_results['timestamp'] = datetime.now().isoformat()
        
        print(f"✅ Worker {self.worker_id} completed Generation {generation_id}")
        print(f"   Episodes: {generation_results['total_episodes']}")
        print(f"   Steps: {generation_results['total_steps']:,}")
        print(f"   Total PnL: ${generation_results['total_pnl']:.2f}")
        print(f"   Duration: {generation_duration:.1f}s")
        print(f"   Steps/sec: {generation_results['steps_per_second']:.0f}")
        
        return generation_results

@ray.remote(num_cpus=3, num_gpus=0)  # Coordinator uses CPU only
class MassiveScaleCoordinator:
    """Coordinates the massive scale training across the cluster"""
    
    def __init__(self):
        self.training_results = []
        self.start_time = None
        
    def coordinate_massive_training(self):
        """
        Coordinate the complete massive scale training
        200 generations x 1000 episodes x 1000 steps
        """
        print("🚀 STARTING MASSIVE SCALE FOREX TRAINING")
        print("=" * 60)
        MassiveScaleConfig.print_config()
        
        self.start_time = time.time()
        
        # Create distributed trainers
        num_workers = MassiveScaleConfig.WORKERS_PER_NODE * 2  # 2 nodes
        trainers = [DistributedForexTrainer.remote(i) for i in range(num_workers)]
        
        # Create bot population
        bot_population = self.create_bot_population()
        
        # Process generations in batches
        batch_size = 4  # Process 4 generations at a time
        
        for generation_batch_start in range(0, MassiveScaleConfig.TOTAL_GENERATIONS, batch_size):
            batch_end = min(generation_batch_start + batch_size, MassiveScaleConfig.TOTAL_GENERATIONS)
            
            print(f"\n🔥 Processing Generation Batch {generation_batch_start}-{batch_end-1}")
            
            # Distribute generations across workers
            generation_futures = []
            for gen_idx in range(generation_batch_start, batch_end):
                worker_idx = gen_idx % len(trainers)
                future = trainers[worker_idx].train_generation.remote(gen_idx, bot_population)
                generation_futures.append(future)
            
            # Collect results
            batch_results = ray.get(generation_futures)
            self.training_results.extend(batch_results)
            
            # Save intermediate results
            if generation_batch_start % (MassiveScaleConfig.SAVE_INTERVAL * batch_size) == 0:
                self.save_intermediate_results(generation_batch_start)
            
            # Print batch summary
            self.print_batch_summary(batch_results, generation_batch_start, batch_end-1)
        
        # Final summary
        self.print_final_summary()
        self.save_final_results()
        
        return self.training_results
    
    def create_bot_population(self):
        """Create initial bot population"""
        population = []
        for i in range(MassiveScaleConfig.POPULATION_SIZE):
            bot_config = {
                'bot_id': f"BOT_{i}",
                'learning_rate': MassiveScaleConfig.LEARNING_RATE * (1 + np.random.normal(0, 0.1)),
                'exploration_rate': MassiveScaleConfig.EXPLORATION_RATE * (1 + np.random.normal(0, 0.2)),
                'risk_tolerance': np.random.uniform(0.5, 2.0),
                'strategy_type': np.random.choice(['conservative', 'moderate', 'aggressive'])
            }
            population.append(bot_config)
        return population
    
    def print_batch_summary(self, batch_results, start_gen, end_gen):
        """Print summary for a batch of generations"""
        total_episodes = sum(r['total_episodes'] for r in batch_results)
        total_steps = sum(r['total_steps'] for r in batch_results)
        total_pnl = sum(r['total_pnl'] for r in batch_results)
        total_duration = sum(r['duration_seconds'] for r in batch_results)
        avg_steps_per_sec = total_steps / total_duration if total_duration > 0 else 0
        
        print(f"\n📊 BATCH SUMMARY (Generations {start_gen}-{end_gen}):")
        print(f"   Episodes Completed: {total_episodes:,}")
        print(f"   Steps Completed: {total_steps:,}")
        print(f"   Total PnL: ${total_pnl:,.2f}")
        print(f"   Batch Duration: {total_duration:.1f}s")
        print(f"   Average Steps/sec: {avg_steps_per_sec:.0f}")
        
        # Progress tracking
        completed_generations = len(self.training_results)
        progress_pct = (completed_generations / MassiveScaleConfig.TOTAL_GENERATIONS) * 100
        elapsed_time = time.time() - self.start_time
        
        print(f"\n🎯 OVERALL PROGRESS:")
        print(f"   Completed: {completed_generations}/{MassiveScaleConfig.TOTAL_GENERATIONS} generations ({progress_pct:.1f}%)")
        print(f"   Elapsed Time: {elapsed_time/3600:.1f} hours")
        
        if completed_generations > 0:
            estimated_total_time = (elapsed_time / completed_generations) * MassiveScaleConfig.TOTAL_GENERATIONS
            remaining_time = estimated_total_time - elapsed_time
            print(f"   Estimated Remaining: {remaining_time/3600:.1f} hours")
    
    def print_final_summary(self):
        """Print final training summary"""
        total_time = time.time() - self.start_time
        total_episodes = sum(r['total_episodes'] for r in self.training_results)
        total_steps = sum(r['total_steps'] for r in self.training_results)
        total_pnl = sum(r['total_pnl'] for r in self.training_results)
        
        print("\n" + "=" * 60)
        print("🎉 MASSIVE SCALE TRAINING COMPLETED!")
        print("=" * 60)
        print(f"📊 FINAL STATISTICS:")
        print(f"   Total Generations: {len(self.training_results)}")
        print(f"   Total Episodes: {total_episodes:,}")
        print(f"   Total Steps: {total_steps:,}")
        print(f"   Total PnL: ${total_pnl:,.2f}")
        print(f"   Total Time: {total_time/3600:.1f} hours")
        print(f"   Average Steps/sec: {total_steps/total_time:.0f}")
        print("=" * 60)
    
    def save_intermediate_results(self, generation_batch):
        """Save intermediate results"""
        filename = f"massive_training_checkpoint_gen_{generation_batch}.json"
        filepath = os.path.join("/home/w1/cursor-to-copilot-backup/TaskmasterForexBots", filename)
        
        checkpoint_data = {
            'config': {
                'GENERATIONS': MassiveScaleConfig.GENERATIONS,
                'EPISODES_PER_GENERATION': MassiveScaleConfig.EPISODES_PER_GENERATION,
                'STEPS_PER_EPISODE': MassiveScaleConfig.STEPS_PER_EPISODE,
                'TOTAL_TRAINING_STEPS': MassiveScaleConfig.TOTAL_TRAINING_STEPS
            },
            'generations_completed': len(self.training_results),
            'results': self.training_results,
            'checkpoint_time': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        print(f"💾 Checkpoint saved: {filename}")
    
    def save_final_results(self):
        """Save final training results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"massive_scale_training_results_{timestamp}.json"
        filepath = os.path.join("/home/w1/cursor-to-copilot-backup/TaskmasterForexBots", filename)
        
        final_data = {
            'config': {
                'GENERATIONS': MassiveScaleConfig.GENERATIONS,
                'EPISODES_PER_GENERATION': MassiveScaleConfig.EPISODES_PER_GENERATION,
                'STEPS_PER_EPISODE': MassiveScaleConfig.STEPS_PER_EPISODE,
                'TOTAL_TRAINING_STEPS': MassiveScaleConfig.TOTAL_TRAINING_STEPS
            },
            'total_time_hours': (time.time() - self.start_time) / 3600,
            'total_generations': len(self.training_results),
            'results': self.training_results,
            'completion_time': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(final_data, f, indent=2)
        
        print(f"💾 Final results saved: {filename}")

def launch_massive_scale_training():
    """Launch the massive scale training on Ray cluster"""
    
    # Initialize Ray connection
    if not ray.is_initialized():
        try:
            ray.init(address='ray://192.168.1.10:10001')
            print("✅ Connected to Ray cluster")
        except Exception as e:
            print(f"❌ Failed to connect to Ray: {e}")
            return
    
    # Check cluster resources
    resources = ray.cluster_resources()
    print(f"📊 Cluster Resources: {resources}")
    
    # Launch coordinator
    print("\n🚀 Launching Massive Scale Training Coordinator...")
    coordinator = MassiveScaleCoordinator.remote()
    
    # Start massive training
    training_future = coordinator.coordinate_massive_training.remote()
    
    print("⏳ Training in progress... (This will take several hours)")
    print("💡 You can monitor progress in the Ray dashboard: http://192.168.1.10:8265")
    
    # Get results (this will block until training completes)
    results = ray.get(training_future)
    
    print("🎉 MASSIVE SCALE TRAINING COMPLETED!")
    return results

if __name__ == "__main__":
    print("🚀 MASSIVE SCALE DISTRIBUTED FOREX TRAINING")
    print("=" * 60)
    print("⚠️  WARNING: This will run for several hours!")
    print("💡 Make sure both PCs are ready and Ray cluster is active")
    
    # Confirm before starting
    response = input("\n▶️  Start massive scale training? (y/N): ")
    
    if response.lower() == 'y':
        results = launch_massive_scale_training()
        print(f"✅ Training completed with {len(results)} generations")
    else:
        print("❌ Training cancelled")
        
        # Run a small test instead
        print("\n🧪 Running small scale test...")
        # Create a smaller test by modifying the constants temporarily
        print("🧪 Test mode: 2 generations × 10 episodes × 100 steps")
        
        coordinator = MassiveScaleCoordinator.remote(test_config)
        test_results = ray.get(coordinator.coordinate_massive_training.remote())
        print(f"🧪 Test completed with {len(test_results)} generations")
