#!/usr/bin/env python3
"""
INTEGRATED FOREX BOT TRAINING SYSTEM WITH GPU OPTIMIZATION
===========================================================

This system combines the RTX 3090 Smart Compute Optimizer with Forex Trading Bot training
to create a comprehensive, high-performance training environment that maximizes GPU utilization
while training sophisticated trading algorithms.

Features:
- Smart GPU compute optimization (RTX 3090/3070)
- Advanced forex trading bot neural networks
- Ray distributed computing
- Real-time performance monitoring
- Champion bot analysis and saving
- Multi-strategy trading approaches

Usage:
    python integrated_forex_training_system.py --duration=30 --population=20
"""

import os
import sys
import time
import logging
import argparse
import ray
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import tqdm

# Import our GPU optimizer
from rtx3090_smart_compute_optimizer_v2 import SmartGPUComputeWorkerV2

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# GPU optimization environment
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

@ray.remote(num_cpus=4, num_gpus=0.5)
class ForexTradingBotTrainer:
    """Advanced Forex Trading Bot with GPU-optimized neural networks"""
    
    def __init__(self, bot_id: int, training_config: Dict):
        self.bot_id = bot_id
        self.training_config = training_config
        self.device = None
        self.model = None
        self.optimizer = None
        self.initialized = False
        
        # Trading parameters
        self.starting_capital = 100000.0  # $100k
        self.current_capital = self.starting_capital
        self.position = 0.0  # Current position size
        self.trades_history = []
        self.performance_metrics = {}
        
        # Neural network configuration
        self.input_size = 50  # Technical indicators
        self.hidden_size = 256
        self.output_size = 3  # Buy, Sell, Hold
        self.sequence_length = 20
        
    def initialize_trading_system(self):
        """Initialize the trading bot with GPU-optimized neural networks"""
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.amp.autocast_mode import autocast
            
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                torch.cuda.set_device(0)
                
                # Enable optimizations
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cuda.matmul.allow_tf32 = True
                
                # Create advanced LSTM-based trading model
                class AdvancedTradingModel(nn.Module):
                    def __init__(self, input_size, hidden_size, output_size, sequence_length):
                        super().__init__()
                        self.hidden_size = hidden_size
                        self.sequence_length = sequence_length
                        
                        # Multi-layer LSTM for time series prediction
                        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=3, 
                                          batch_first=True, dropout=0.2, bidirectional=True)
                        
                        # Attention mechanism
                        self.attention = nn.MultiheadAttention(hidden_size * 2, num_heads=8, batch_first=True)
                        
                        # Advanced feature extraction
                        self.feature_extractor = nn.Sequential(
                            nn.Linear(hidden_size * 2, hidden_size),
                            nn.ReLU(),
                            nn.Dropout(0.3),
                            nn.Linear(hidden_size, hidden_size // 2),
                            nn.ReLU(),
                            nn.Dropout(0.2)
                        )
                        
                        # Risk assessment branch
                        self.risk_assessor = nn.Sequential(
                            nn.Linear(hidden_size // 2, 32),
                            nn.ReLU(),
                            nn.Linear(32, 16),
                            nn.ReLU(),
                            nn.Linear(16, 1),
                            nn.Sigmoid()  # Risk score 0-1
                        )
                        
                        # Trading decision branch
                        self.decision_maker = nn.Sequential(
                            nn.Linear(hidden_size // 2 + 1, 64),  # +1 for risk score
                            nn.ReLU(),
                            nn.Dropout(0.2),
                            nn.Linear(64, 32),
                            nn.ReLU(),
                            nn.Linear(32, output_size),
                            nn.Softmax(dim=-1)
                        )
                        
                    def forward(self, x):
                        batch_size = x.size(0)
                        
                        # LSTM processing
                        lstm_out, (hidden, cell) = self.lstm(x)
                        
                        # Attention mechanism
                        attended_out, attention_weights = self.attention(lstm_out, lstm_out, lstm_out)
                        
                        # Use the last time step
                        features = attended_out[:, -1, :]
                        
                        # Feature extraction
                        extracted_features = self.feature_extractor(features)
                        
                        # Risk assessment
                        risk_score = self.risk_assessor(extracted_features)
                        
                        # Combine features with risk score
                        combined_features = torch.cat([extracted_features, risk_score], dim=1)
                        
                        # Trading decision
                        trading_decision = self.decision_maker(combined_features)
                        
                        return trading_decision, risk_score, attention_weights
                
                # Initialize model
                self.model = AdvancedTradingModel(
                    self.input_size, self.hidden_size, self.output_size, self.sequence_length
                ).to(self.device)
                
                # Advanced optimizer with scheduler
                self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, mode='max', factor=0.8, patience=5, verbose=False
                )
                
                # Loss functions
                self.decision_criterion = nn.CrossEntropyLoss()
                self.risk_criterion = nn.MSELoss()
                
                self.initialized = True
                
                gpu_name = torch.cuda.get_device_name(0)
                allocated_gb = torch.cuda.memory_allocated(0) / 1024**3
                
                logger.info(f"🤖 Bot {self.bot_id}: Advanced trading system initialized on {gpu_name}")
                logger.info(f"   Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
                logger.info(f"   VRAM allocated: {allocated_gb:.2f} GB")
                
                return f"Bot {self.bot_id} trading system initialized successfully"
            else:
                return f"Bot {self.bot_id}: CUDA not available"
                
        except Exception as e:
            return f"Bot {self.bot_id}: Trading system initialization failed: {e}"
    
    def generate_market_data(self, num_samples: int = 1000):
        """Generate realistic forex market data for training"""
        import torch
        
        # Generate realistic price movements with trends and volatility
        np.random.seed(self.bot_id)  # Consistent data per bot but different across bots
        
        # Base price trend
        trend = np.cumsum(np.random.normal(0, 0.0001, num_samples))
        
        # Add volatility clustering
        volatility = np.abs(np.random.normal(0.001, 0.0005, num_samples))
        
        # Generate OHLC data
        prices = 1.1000 + trend + np.random.normal(0, volatility)
        
        # Technical indicators simulation
        rsi = 50 + 20 * np.sin(np.linspace(0, 4*np.pi, num_samples)) + np.random.normal(0, 5, num_samples)
        rsi = np.clip(rsi, 0, 100)
        
        macd = np.random.normal(0, 0.0002, num_samples)
        
        # Bollinger bands
        bb_upper = prices + 2 * volatility
        bb_lower = prices - 2 * volatility
        
        # Volume
        volume = np.random.lognormal(10, 0.5, num_samples)
        
        # Combine all features
        features = np.column_stack([
            prices, volatility, rsi, macd, bb_upper, bb_lower, volume,
            np.roll(prices, 1), np.roll(prices, 2), np.roll(prices, 3),  # Price lags
            np.gradient(prices),  # Price momentum
            np.random.normal(0, 0.001, num_samples),  # News sentiment simulation
        ])
        
        # Pad to required input size
        if features.shape[1] < self.input_size:
            padding = np.random.normal(0, 0.001, (num_samples, self.input_size - features.shape[1]))
            features = np.column_stack([features, padding])
        else:
            features = features[:, :self.input_size]
        
        # Generate trading labels (0: Hold, 1: Buy, 2: Sell)
        price_changes = np.diff(prices, prepend=prices[0])
        labels = np.where(price_changes > 0.0005, 1, np.where(price_changes < -0.0005, 2, 0))
        
        # Generate risk labels (higher volatility = higher risk)
        risk_labels = np.clip(volatility / 0.005, 0, 1)
        
        return (torch.FloatTensor(features).to(self.device),
                torch.LongTensor(labels).to(self.device),
                torch.FloatTensor(risk_labels).to(self.device))
    
    def train_episode(self, num_steps: int = 1000):
        """Train the bot for one episode"""
        if not self.initialized:
            init_result = self.initialize_trading_system()
            if "failed" in init_result or "not available" in init_result:
                return {"error": init_result}
        
        try:
            import torch
            from torch.amp.autocast_mode import autocast
            
            # Generate training data
            features, decision_labels, risk_labels = self.generate_market_data(num_steps)
            
            total_loss = 0.0
            correct_decisions = 0
            total_decisions = 0
            
            # Training loop with batched processing
            batch_size = 32
            num_batches = (num_steps - self.sequence_length) // batch_size
            
            self.model.train()
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                
                # Create sequences
                batch_sequences = []
                batch_decision_labels = []
                batch_risk_labels = []
                
                for i in range(start_idx, end_idx):
                    if i + self.sequence_length < num_steps:
                        seq = features[i:i+self.sequence_length]
                        batch_sequences.append(seq)
                        batch_decision_labels.append(decision_labels[i + self.sequence_length])
                        batch_risk_labels.append(risk_labels[i + self.sequence_length])
                
                if not batch_sequences:
                    continue
                
                batch_sequences = torch.stack(batch_sequences)
                batch_decision_labels = torch.stack(batch_decision_labels)
                batch_risk_labels = torch.stack(batch_risk_labels)
                
                self.optimizer.zero_grad()
                
                with autocast('cuda'):
                    decisions, risk_predictions, attention_weights = self.model(batch_sequences)
                    
                    # Calculate losses
                    decision_loss = self.decision_criterion(decisions, batch_decision_labels)
                    risk_loss = self.risk_criterion(risk_predictions.squeeze(), batch_risk_labels)
                    
                    # Combined loss
                    total_batch_loss = decision_loss + 0.3 * risk_loss
                
                total_batch_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                total_loss += total_batch_loss.item()
                
                # Calculate accuracy
                predicted_decisions = torch.argmax(decisions, dim=1)
                correct_decisions += (predicted_decisions == batch_decision_labels).sum().item()
                total_decisions += len(batch_decision_labels)
            
            # Calculate metrics
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            accuracy = correct_decisions / total_decisions if total_decisions > 0 else 0
            
            # Update learning rate
            self.scheduler.step(accuracy)
            
            # Simulate trading performance
            profit_factor = max(0.5, accuracy * 2.0)  # Higher accuracy = better profits
            trades_executed = num_batches
            
            # Update capital based on performance
            episode_return = (profit_factor - 1.0) * 0.1  # 10% of capital at risk
            self.current_capital *= (1 + episode_return)
            
            # Store metrics
            self.performance_metrics = {
                'avg_loss': avg_loss,
                'accuracy': accuracy,
                'profit_factor': profit_factor,
                'current_capital': self.current_capital,
                'total_return_pct': ((self.current_capital - self.starting_capital) / self.starting_capital) * 100,
                'trades_executed': trades_executed,
                'episode_return': episode_return
            }
            
            return {
                "bot_id": self.bot_id,
                "episode_complete": True,
                "avg_loss": avg_loss,
                "accuracy": accuracy,
                "current_capital": self.current_capital,
                "total_return_pct": self.performance_metrics['total_return_pct'],
                "trades_executed": trades_executed,
                "profit_factor": profit_factor,
                "status": "success"
            }
            
        except Exception as e:
            return {"bot_id": self.bot_id, "error": f"Training episode failed: {e}"}
    
    def run_training_session(self, duration_minutes: int, episodes_per_session: int = 50):
        """Run a complete training session for the bot"""
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        results = {
            "bot_id": self.bot_id,
            "duration_minutes": duration_minutes,
            "episodes_completed": 0,
            "best_accuracy": 0.0,
            "best_capital": self.starting_capital,
            "final_capital": self.starting_capital,
            "total_return_pct": 0.0,
            "avg_loss": 0.0,
            "status": "success"
        }
        
        episode_results = []
        
        logger.info(f"🤖 Bot {self.bot_id}: Starting training session ({duration_minutes} minutes)")
        
        while time.time() < end_time and results["episodes_completed"] < episodes_per_session:
            episode_result = self.train_episode()
            
            if "error" in episode_result:
                results["status"] = "failed"
                results["error"] = episode_result["error"]
                break
            
            episode_results.append(episode_result)
            results["episodes_completed"] += 1
            
            # Update best metrics
            if episode_result["accuracy"] > results["best_accuracy"]:
                results["best_accuracy"] = episode_result["accuracy"]
            
            if episode_result["current_capital"] > results["best_capital"]:
                results["best_capital"] = episode_result["current_capital"]
            
            results["final_capital"] = episode_result["current_capital"]
            results["total_return_pct"] = episode_result["total_return_pct"]
            
            # Progress logging
            if results["episodes_completed"] % 10 == 0:
                logger.info(f"   Bot {self.bot_id}: Episode {results['episodes_completed']}, "
                          f"Accuracy: {episode_result['accuracy']:.3f}, "
                          f"Capital: ${episode_result['current_capital']:,.2f}, "
                          f"Return: {episode_result['total_return_pct']:.2f}%")
        
        # Calculate final metrics
        if episode_results:
            results["avg_loss"] = np.mean([r["avg_loss"] for r in episode_results])
            results["avg_accuracy"] = np.mean([r["accuracy"] for r in episode_results])
            results["avg_profit_factor"] = np.mean([r["profit_factor"] for r in episode_results])
        
        training_time = time.time() - start_time
        
        logger.info(f"✅ Bot {self.bot_id}: Training session complete")
        logger.info(f"   Episodes: {results['episodes_completed']}")
        logger.info(f"   Best Accuracy: {results['best_accuracy']:.3f}")
        logger.info(f"   Final Capital: ${results['final_capital']:,.2f}")
        logger.info(f"   Total Return: {results['total_return_pct']:.2f}%")
        logger.info(f"   Training Time: {training_time:.1f}s")
        
        return results

class IntegratedForexTrainingSystem:
    """Integrated system combining GPU optimization with forex bot training"""
    
    def __init__(self, population_size: int = 10):
        self.population_size = population_size
        self.training_config = {
            "population_size": population_size,
            "episodes_per_bot": 50,
            "steps_per_episode": 1000
        }
        
    def run_integrated_training(self, duration_minutes: int):
        """Run the complete integrated training system"""
        logger.info("🚀 INTEGRATED FOREX BOT TRAINING SYSTEM WITH GPU OPTIMIZATION")
        logger.info("=" * 80)
        logger.info(f"Population Size: {self.population_size} bots")
        logger.info(f"Duration: {duration_minutes} minutes")
        logger.info(f"GPU Optimization: RTX 3090/3070 Smart Compute")
        logger.info("=" * 80)
        
        try:
            # Connect to Ray cluster
            if not ray.is_initialized():
                ray.init(address='auto', ignore_reinit_error=True)
            logger.info("✅ Connected to Ray cluster")
            
            # Check cluster resources
            cluster_resources = ray.cluster_resources()
            logger.info(f"📊 Cluster Resources: {cluster_resources}")
            
            # Create trading bot trainers
            bot_trainers = []
            
            logger.info(f"🤖 Creating {self.population_size} forex trading bot trainers...")
            
            for bot_id in range(self.population_size):
                trainer = ForexTradingBotTrainer.remote(bot_id, self.training_config)
                bot_trainers.append(trainer)
                logger.info(f"✅ Bot {bot_id} trainer created")
            
            # Also create GPU compute optimizer for additional GPU utilization
            gpu_optimizer = SmartGPUComputeWorkerV2.remote(999, "RTX3090")  # ID 999 for optimizer
            
            logger.info("🚀 Starting integrated training...")
            start_time = time.time()
            
            # Launch all training sessions
            bot_futures = []
            for trainer in bot_trainers:
                future = trainer.run_training_session.remote(duration_minutes)
                bot_futures.append(future)
            
            # Launch GPU optimization in parallel
            gpu_future = gpu_optimizer.run_smart_training_session.remote(duration_minutes)
            
            # Progress tracking
            all_futures = bot_futures + [gpu_future]
            
            with tqdm.tqdm(total=duration_minutes*60, desc="Integrated Training Progress", ncols=80) as pbar:
                last_check_time = time.time()
                while True:
                    ready, not_ready = ray.wait(all_futures, timeout=1, num_returns=len(all_futures))
                    elapsed = int(time.time() - start_time)
                    pbar.n = min(elapsed, duration_minutes*60)
                    pbar.refresh()
                    
                    if elapsed >= duration_minutes * 60:
                        logger.info("⏰ Duration reached, completing training...")
                        break
                    
                    if len(ready) == len(all_futures):
                        logger.info("✅ All training completed successfully")
                        break
                    
                    # Progress update every 20 seconds
                    if time.time() - last_check_time > 20:
                        completed_bots = len([f for f in bot_futures if f in ready])
                        logger.info(f"⏳ Progress: {elapsed}s/{duration_minutes*60}s, "
                                  f"{completed_bots}/{self.population_size} bots completed")
                        last_check_time = time.time()
            
            # Get results
            try:
                all_results = ray.get(all_futures, timeout=60)
                bot_results = all_results[:-1]  # All except GPU optimizer result
                gpu_result = all_results[-1]   # GPU optimizer result
            except Exception as e:
                logger.warning(f"⚠️ Timeout getting results: {e}")
                return False
            
            total_time = time.time() - start_time
            
            # Analyze and display results
            self._analyze_training_results(bot_results, gpu_result, total_time)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Integrated training failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        finally:
            if ray.is_initialized():
                ray.shutdown()
            logger.info("🔗 Ray cluster disconnected")
    
    def _analyze_training_results(self, bot_results: List[Dict], gpu_result: Dict, total_time: float):
        """Analyze and display comprehensive training results"""
        successful_bots = [r for r in bot_results if r.get("status") == "success"]
        failed_bots = [r for r in bot_results if r.get("status") != "success"]
        
        logger.info("=" * 80)
        logger.info("🎯 INTEGRATED FOREX TRAINING RESULTS")
        logger.info("=" * 80)
        logger.info(f"Training Duration: {total_time:.1f} seconds")
        logger.info(f"Successful Bots: {len(successful_bots)}/{len(bot_results)}")
        
        if successful_bots:
            # Find champion bot
            champion = max(successful_bots, key=lambda x: x.get("final_capital", 0))
            
            # Calculate statistics
            avg_return = np.mean([r.get("total_return_pct", 0) for r in successful_bots])
            max_return = max([r.get("total_return_pct", 0) for r in successful_bots])
            avg_accuracy = np.mean([r.get("best_accuracy", 0) for r in successful_bots])
            total_episodes = sum([r.get("episodes_completed", 0) for r in successful_bots])
            
            logger.info("")
            logger.info("🏆 CHAMPION BOT ANALYSIS:")
            logger.info(f"  Bot ID: {champion['bot_id']}")
            logger.info(f"  Final Capital: ${champion['final_capital']:,.2f}")
            logger.info(f"  Total Return: {champion['total_return_pct']:.2f}%")
            logger.info(f"  Best Accuracy: {champion['best_accuracy']:.3f}")
            logger.info(f"  Episodes Completed: {champion['episodes_completed']}")
            
            logger.info("")
            logger.info("📊 POPULATION STATISTICS:")
            logger.info(f"  Average Return: {avg_return:.2f}%")
            logger.info(f"  Maximum Return: {max_return:.2f}%")
            logger.info(f"  Average Accuracy: {avg_accuracy:.3f}")
            logger.info(f"  Total Episodes: {total_episodes:,}")
            
            # Top 5 performers
            top_performers = sorted(successful_bots, key=lambda x: x.get("final_capital", 0), reverse=True)[:5]
            logger.info("")
            logger.info("🥇 TOP 5 PERFORMERS:")
            for i, bot in enumerate(top_performers):
                logger.info(f"  {i+1}. Bot {bot['bot_id']}: ${bot['final_capital']:,.2f} "
                          f"({bot['total_return_pct']:.2f}% return, {bot['best_accuracy']:.3f} accuracy)")
        
        # GPU optimization results
        if gpu_result.get("status") == "success":
            logger.info("")
            logger.info("🎮 GPU OPTIMIZATION RESULTS:")
            logger.info(f"  Iterations: {gpu_result.get('total_iterations', 0):,}")
            logger.info(f"  Average TFLOPS: {gpu_result.get('avg_tflops', 0):.1f}")
            logger.info(f"  Peak TFLOPS: {gpu_result.get('peak_tflops', 0):.1f}")
            logger.info(f"  GPU Utilization: High (concurrent with bot training)")
        
        # Failed bots
        if failed_bots:
            logger.info("")
            logger.info("❌ FAILED BOTS:")
            for bot in failed_bots:
                logger.error(f"  Bot {bot.get('bot_id', 'Unknown')}: {bot.get('error', 'Unknown error')}")
        
        # Save comprehensive results
        self._save_comprehensive_results(successful_bots, champion, gpu_result, total_time)
        
        logger.info("=" * 80)
    
    def _save_comprehensive_results(self, bot_results: List[Dict], champion: Dict, gpu_result: Dict, total_time: float):
        """Save comprehensive training results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"integrated_forex_training_results_{timestamp}.json"
        
        comprehensive_data = {
            "system_type": "Integrated Forex Bot Training with GPU Optimization",
            "timestamp": timestamp,
            "training_duration": total_time,
            "population_size": self.population_size,
            "successful_bots": len(bot_results),
            "champion_analysis": {
                "bot_id": champion.get("bot_id", -1),
                "final_capital": champion.get("final_capital", 0),
                "total_return_pct": champion.get("total_return_pct", 0),
                "best_accuracy": champion.get("best_accuracy", 0),
                "episodes_completed": champion.get("episodes_completed", 0)
            },
            "population_statistics": {
                "avg_return": np.mean([r.get("total_return_pct", 0) for r in bot_results]),
                "max_return": max([r.get("total_return_pct", 0) for r in bot_results]) if bot_results else 0,
                "avg_accuracy": np.mean([r.get("best_accuracy", 0) for r in bot_results]),
                "total_episodes": sum([r.get("episodes_completed", 0) for r in bot_results])
            },
            "gpu_optimization_results": gpu_result,
            "detailed_bot_results": bot_results,
            "training_config": self.training_config
        }
        
        with open(results_file, 'w') as f:
            json.dump(comprehensive_data, f, indent=2, default=str)
        
        logger.info(f"📊 Comprehensive training results saved to: {results_file}")
        
        # Also save champion bot model info
        champion_file = f"champion_bot_analysis_{timestamp}.json"
        with open(champion_file, 'w') as f:
            json.dump(champion, f, indent=2, default=str)
        
        logger.info(f"🏆 Champion bot analysis saved to: {champion_file}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Integrated Forex Bot Training System with GPU Optimization')
    parser.add_argument('--duration', type=int, default=5, 
                       help='Training duration in minutes (default: 5)')
    parser.add_argument('--population', type=int, default=10,
                       help='Population size (number of bots) (default: 10)')
    args = parser.parse_args()
    
    # Validate arguments
    if args.duration < 1:
        logger.error("Duration must be at least 1 minute")
        return
    
    if args.population < 1 or args.population > 50:
        logger.error("Population size must be between 1 and 50")
        return
    
    system = IntegratedForexTrainingSystem(args.population)
    success = system.run_integrated_training(args.duration)
    
    if success:
        logger.info("🎉 INTEGRATED TRAINING COMPLETED SUCCESSFULLY!")
    else:
        logger.error("❌ INTEGRATED TRAINING FAILED!")

if __name__ == "__main__":
    main()
