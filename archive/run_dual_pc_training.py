#!/usr/bin/env python3
"""
SINGLE MACHINE TRAINING SCRIPT
Runs forex bot training on this machine
"""
import torch
import torch.nn as nn
import numpy as np
import time
import logging
from datetime import datetime
from typing import List, Dict
import threading
import signal
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import from your existing training system
from run_smart_real_training import SmartTradingBot, SmartForexEnvironment

class SingleMachineTrainer:
    """Single machine trainer for forex bots"""
    
    def __init__(self):
        """Initialize single machine trainer"""
        self.running = False
        self.population_size = 100  # Smaller population for single machine
        self.generations = 50
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("🚀 Single Machine Trainer initialized")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info("🛑 Shutdown signal received, stopping training...")
        self.running = False
        sys.exit(0)
    def create_population(self) -> List[SmartTradingBot]:
        """Create local population of bots"""
        logger.info(f"🧬 Creating population of {self.population_size} bots...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        population = [SmartTradingBot().to(device) for _ in range(self.population_size)]  # type: ignore
        logger.info(f"✅ Created {len(population)} bots on {device}")
        return population
    
    def evaluate_bot(self, bot: SmartTradingBot, bot_id: int) -> Dict:
        """Evaluate a single bot"""
        env = SmartForexEnvironment()
        # Assign bot ID for tracking
        env.bot_id = bot_id
        obs, _ = env.reset()
        # Determine the device the bot is on
        device = next(bot.parameters()).device
        total_reward = 0
        steps = 0
        
        for _ in range(1000):  # Max steps
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                action_probs, position_size = bot(obs_tensor)
                action = int(torch.argmax(action_probs).item())
            
            obs, reward, done, _, info = env.step(action, position_size.item())
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        # Return performance metrics along with bot ID
        return {
            'bot_id': bot_id,
            'total_reward': total_reward,
            'final_balance': env.balance,
            'steps': steps,
            'trades': len(env.trades)
        }
    
    def evaluate_population(self, population: List[SmartTradingBot]) -> List[Dict]:
        """Evaluate entire population sequentially"""
        logger.info(f"⚡ Evaluating {len(population)} bots...")
        start_time = time.time()
        
        results = []
        for i, bot in enumerate(population):
            # Pass bot index as bot_id
            result = self.evaluate_bot(bot, i)
            results.append(result)
            progress = (i + 1) / len(population) * 100
            logger.info(f"📊 Evaluation progress: {progress:.1f}% ({i + 1}/{len(population)})")
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Evaluation completed in {elapsed:.1f}s")
        
        # Sort by performance
        results.sort(key=lambda x: x['total_reward'], reverse=True)
        return results
    
    def run_training(self):
        """Run the main training loop"""
        try:
            self.running = True
            logger.info("🏁 === SINGLE MACHINE TRAINING STARTED ===")
            
            # Create initial population
            population = self.create_population()
            best_overall_score = float('-inf')
            best_bot = None
            
            for generation in range(self.generations):
                if not self.running:
                    break
                
                logger.info(f"\n🎯 === GENERATION {generation + 1}/{self.generations} ===")
                
                # Evaluate population
                results = self.evaluate_population(population)
                
                # Get champion
                champion = results[0]
                logger.info(f"🏆 Generation Champion:")
                logger.info(f"   Reward: {champion['total_reward']:.2f}")
                logger.info(f"   Balance: ${champion['final_balance']:.2f}")
                logger.info(f"   Trades: {champion['trades']}")
                
                # Check if new overall champion
                if champion['total_reward'] > best_overall_score:
                    best_overall_score = champion['total_reward']
                    best_bot = population[0]  # First one is best after sorting
                    
                    logger.info(f"🎉 NEW OVERALL CHAMPION! Score: {best_overall_score:.2f}")
                    
                    # Save champion
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    champion_file = f"CHAMPION_{timestamp}.pth"
                    torch.save(best_bot.state_dict(), champion_file)
                    logger.info(f"💾 Champion saved: {champion_file}")
                
                # Evolution (simplified)
                if generation < self.generations - 1:
                    logger.info("🧬 Evolving population...")
                    elite_count = self.population_size // 10  # Top 10%
                    
                    for i in range(elite_count, self.population_size):
                        # Mutate a random elite bot
                        elite_idx = np.random.randint(0, elite_count)
                        elite_bot = population[elite_idx]
                        target_bot = population[i]
                        
                        # Copy elite to target and mutate
                        target_bot.load_state_dict(elite_bot.state_dict())
                        with torch.no_grad():
                            for param in target_bot.parameters():
                                if np.random.random() < 0.1:
                                    noise = torch.randn_like(param) * 0.01
                                    param.add_(noise)
                
                progress = (generation + 1) / self.generations * 100
                logger.info(f"📈 Training Progress: {progress:.1f}%")
                
                # Brief pause between generations
                time.sleep(1)
            
            # Training complete
            logger.info("\n🏁 === TRAINING COMPLETED ===")
            if best_bot:
                logger.info(f"🏆 FINAL CHAMPION:")
                final_result = self.evaluate_bot(best_bot, 0) # Pass 0 for final evaluation
                logger.info(f"   Reward: {final_result['total_reward']:.2f}")
                logger.info(f"   Balance: ${final_result['final_balance']:.2f}")
                logger.info(f"   Trades: {final_result['trades']}")
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            
        finally:
            self.running = False
    
    def stop_training(self):
        """Stop training and cleanup"""
        self.running = False
        logger.info("🛑 Training stopped")

def main():
    """Main function"""
    print("🚀 === SINGLE MACHINE TRAINING ===")
    print("This script will run forex bot training on this machine")
    print()
    
    trainer = SingleMachineTrainer()
    
    try:
        trainer.run_training()
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
        trainer.stop_training()
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        trainer.stop_training()

if __name__ == "__main__":
    main() 