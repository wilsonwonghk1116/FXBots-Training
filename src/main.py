"""
Main entry point for Smart Real Training System.
"""

import logging
from src.trainer import VRAMOptimizedTrainer
from src.utils import monitor_system_resources, load_config
from torch.utils.tensorboard import SummaryWriter
import os

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    config = load_config('config.yaml')
    writer = SummaryWriter('logs/smart_trading')
    population_size = config['trainer']['population_size']
    generations = config['trainer']['generations']
    trainer = VRAMOptimizedTrainer(population_size=population_size)
    logger.info("=== Smart Real Training System Started ===")
    logger.info("Targeting 95% VRAM utilization on RTX 3090 24GB")
    population = trainer.create_population()
    for generation in range(generations):
        logger.info(f"\n=== Generation {generation + 1}/{generations} ===")
        monitor_system_resources()
        population, results = trainer.evolve_generation(population)
        best = max(results, key=lambda x: x.get('total_trades', 0) or x.get('final_balance', 0))
        trades = best.get('total_trades', 0)
        if trades == 0:
            logger.warning("No trades executed - adjusting strategy thresholds")
            trainer.adjust_trading_thresholds(0.1)
        logger.info(f"Best Bot: Balance={best.get('final_balance', 0):.2f}, "
                   f"Return={best.get('total_return_pct', 0):.2f}%, "
                   f"Win Rate={best.get('win_rate', 0):.2f}, "
                   f"Trades={trades}, "
                   f"Avg Win={best.get('avg_win', 0):.1f}pips, "
                   f"Avg Loss={best.get('avg_loss', 0):.1f}pips, "
                   f"Profit Factor={best.get('profit_factor', 0):.2f}")
        writer.add_scalar('Balance', best['final_balance'], generation)
        writer.add_scalar('Return', best['total_return_pct'], generation)
        writer.add_scalar('Trades', best['total_trades'], generation)
        if (generation + 1) % 10 == 0:
            champion_bot = population[best['bot_id']]
            checkpoint = {
                'generation': generation,
                'model_state_dict': champion_bot.state_dict(),
                'metrics': best
            }
            os.makedirs('checkpoints', exist_ok=True)
            import torch
            torch.save(checkpoint, f'checkpoints/gen_{generation}.pth')
            analysis = trainer.analyze_champion(champion_bot, results)
            model_path = trainer.save_champion(champion_bot, analysis)
            logger.info("\n=== CHAMPION ANALYSIS ===")
            champion = analysis['champion_analysis']
            logger.info(f"Final Balance: ${champion['final_balance']:.2f}")
            logger.info(f"Total Return: {champion['total_return_pct']:.2f}%")
            logger.info(f"Win Rate: {champion['win_rate']:.2f}")
            logger.info(f"Profit Factor: {champion['profit_factor']:.2f}")
            logger.info(f"Max Drawdown: {champion['max_drawdown']:.2f}%")
            logger.info(f"Sharpe Ratio: {champion['sharpe_ratio']:.2f}")
    logger.info("\n=== FINAL CHAMPION ANALYSIS ===")
    final_results = trainer.evaluate_population(population)
    champion_bot = population[final_results[0]['bot_id']]
    final_analysis = trainer.analyze_champion(champion_bot, final_results)
    final_model_path = trainer.save_champion(champion_bot, final_analysis)
    champion = final_analysis['champion_analysis']
    logger.info(f"🏆 CHAMPION BOT PERFORMANCE:")
    logger.info(f"   Final Balance: ${champion['final_balance']:.2f}")
    logger.info(f"   Total Return: {champion['total_return_pct']:.2f}%")
    logger.info(f"   Win Rate: {champion['win_rate']:.2f}")
    logger.info(f"   Total Trades: {champion['total_trades']}")
    logger.info(f"   Profit Factor: {champion['profit_factor']:.2f}")
    logger.info(f"   Average Win: {champion['average_win']:.2f} pips")
    logger.info(f"   Average Loss: {champion['average_loss']:.2f} pips")
    logger.info(f"   Risk/Reward: {champion['risk_reward_ratio']:.2f}")
    logger.info(f"   Max Drawdown: {champion['max_drawdown']:.2f}%")
    logger.info(f"   Recovery Factor: {champion['recovery_factor']:.2f}")
    logger.info(f"   Sharpe Ratio: {champion['sharpe_ratio']:.2f}")
    logger.info(f"   Calmar Ratio: {champion['calmar_ratio']:.2f}")
    logger.info(f"   Model saved: {final_model_path}")
    logger.info("\n=== Training Complete ===")
