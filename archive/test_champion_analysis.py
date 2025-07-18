#!/usr/bin/env python3
"""
Test Champion Analysis System - FIXED VERSION
Quick test for the enhanced forex training system with improved action selection
"""

import torch
import logging
from run_stable_85_percent_trainer import StableForexEnvironment, StableTradingBot, Stable85PercentTrainer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_single_bot_actions():
    """Test a single bot to see its action distribution with detailed debugging"""
    logger.info("=== Testing Single Bot Action Distribution ===")
    
    env = StableForexEnvironment()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bot = StableTradingBot().to(device)
    
    # Test the bot's actions with detailed debugging
    logger.info("=== DEBUGGING ACTION SELECTION ===")
    
    for test_step in range(10):  # Test 10 steps
        obs = env._get_observation()
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Get raw network output (logits)
            raw_logits = bot.network(obs_tensor)
            
            # Get final probabilities after temperature scaling
            action_probs = bot(obs_tensor)
            
            # Manual action selection like in simulate_detailed
            entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-8))
            noise = torch.randn_like(action_probs) * 0.01
            noisy_probs = torch.softmax(torch.log(action_probs + 1e-8) + noise, dim=-1)
            action_dist = torch.distributions.Categorical(noisy_probs)
            action = int(action_dist.sample().item())
            
            logger.info(f"Step {test_step}: Raw logits: {raw_logits.squeeze().cpu().numpy()}")
            logger.info(f"Step {test_step}: Final probs: {action_probs.squeeze().cpu().numpy()}")
            logger.info(f"Step {test_step}: Noisy probs: {noisy_probs.squeeze().cpu().numpy()}")
            logger.info(f"Step {test_step}: Selected action: {action}, Entropy: {entropy.item():.3f}")
            logger.info(f"Step {test_step}: Temperature: {bot.temperature.item():.3f}")
            logger.info("---")
        
        # Step the environment
        obs, reward, done, _, info = env.step(action)
        if done:
            break
    
    # Run full simulation
    result = env.simulate_detailed(bot, steps=100)  # Shorter for debugging
    logger.info(f"Full simulation result: {result['total_trades']} trades, action_counts: {result.get('action_counts', 'N/A')}")
    
    return result

def test_champion_analysis():
    """Test the champion analysis system with fixed trainer"""
    logger.info("=== Testing Fixed Champion Analysis System ===")
    
    # Test single bot first
    single_result = test_single_bot_actions()
    
    # Initialize components
    env = StableForexEnvironment()
    trainer = Stable85PercentTrainer()
    
    # Create a small test population
    logger.info("Creating test population...")
    population = []
    for i in range(5):  # Even smaller population for debugging
        bot = StableTradingBot().to(trainer.device)
        population.append(bot)
    
    # Evaluate population
    logger.info("Evaluating test population...")
    results = trainer.evaluate_population(population)
    
    # Show results for all bots
    for i, result in enumerate(results):
        logger.info(f"Bot {i}: {result['total_trades']} trades, balance: ${result['final_balance']:.2f}")
    
    # Analyze champion
    logger.info("Analyzing champion bot...")
    champion_bot = population[results[0]['bot_id']]
    analysis = trainer.analyze_champion(champion_bot, results)
    
    # Display results
    champion = analysis['champion_analysis']
    logger.info("\n=== FIXED CHAMPION RESULTS ===")
    logger.info(f"Final Balance: ${champion['final_balance']:.2f}")
    logger.info(f"Total Return: {champion['total_return_pct']:.2f}%")
    logger.info(f"Win Rate: {champion['win_rate']:.2f}")
    logger.info(f"Total Trades: {champion['total_trades']}")
    logger.info(f"Profit Factor: {champion['profit_factor']:.2f}")
    logger.info(f"Max Drawdown: {champion['max_drawdown']:.2f}%")
    logger.info(f"Sharpe Ratio: {champion['sharpe_ratio']:.2f}")
    
    # Save champion
    model_path = trainer.save_champion(champion_bot, analysis)
    logger.info(f"Champion saved: {model_path}")
    
    logger.info("=== Test Complete ===")
    return champion

if __name__ == "__main__":
    test_champion_analysis()
