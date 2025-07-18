#!/usr/bin/env python3
"""
Kelly Monte Carlo Bot Fleet Runner
Implements and runs a fleet of 2000 FOREX trading bots using Kelly Criterion and Monte Carlo simulation
"""

import sys
import os
import time
import logging
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from kelly_monte_bot import (
    KellyMonteBot, BotFleetManager, TradingParameters,
    DataManager, MonteCarloEngine, KellyCalculator
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('kelly_fleet.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class KellyFleetAnalyzer:
    """Advanced analytics for Kelly Monte Carlo bot fleet"""
    
    def __init__(self):
        self.results_dir = Path("kelly_results")
        self.results_dir.mkdir(exist_ok=True)
        
    def analyze_fleet_performance(self, fleet_manager: BotFleetManager, save_plots: bool = True):
        """
        Comprehensive analysis of fleet performance
        
        Args:
            fleet_manager: BotFleetManager instance
            save_plots: Whether to save analysis plots
        """
        logger.info("Analyzing fleet performance...")
        
        # Get fleet performance metrics
        fleet_metrics = fleet_manager.get_fleet_performance()
        
        if not fleet_metrics:
            logger.warning("No fleet metrics available")
            return
        
        # Collect individual bot metrics
        bot_metrics = []
        for bot in fleet_manager.bots:
            metrics = bot.get_performance_metrics()
            if metrics:
                bot_metrics.append(metrics)
        
        if not bot_metrics:
            logger.warning("No bot metrics available")
            return
        
        # Create comprehensive analysis
        analysis = {
            'fleet_summary': fleet_metrics,
            'bot_performance_analysis': self._analyze_bot_performance(bot_metrics),
            'kelly_analysis': self._analyze_kelly_performance(fleet_manager.bots),
            'risk_analysis': self._analyze_risk_metrics(bot_metrics),
            'trade_analysis': self._analyze_trade_patterns(fleet_manager.bots)
        }
        
        # Save analysis
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_file = self.results_dir / f"kelly_fleet_analysis_{timestamp}.json"
        
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        logger.info(f"Analysis saved to {analysis_file}")
        
        # Generate plots if requested
        if save_plots:
            self._create_performance_plots(bot_metrics, timestamp)
        
        # Print summary
        self._print_fleet_summary(fleet_metrics, bot_metrics)
        
        return analysis
    
    def _analyze_bot_performance(self, bot_metrics):
        """Analyze individual bot performance statistics"""
        if not bot_metrics:
            return {}
        
        returns = [m['total_return_pct'] for m in bot_metrics]
        sharpes = [m['sharpe_ratio'] for m in bot_metrics if not np.isnan(m['sharpe_ratio'])]
        drawdowns = [m['max_drawdown'] for m in bot_metrics]
        win_rates = [m['win_rate'] for m in bot_metrics]
        
        return {
            'return_statistics': {
                'mean': np.mean(returns),
                'median': np.median(returns),
                'std': np.std(returns),
                'min': np.min(returns),
                'max': np.max(returns),
                'percentile_25': np.percentile(returns, 25),
                'percentile_75': np.percentile(returns, 75)
            },
            'sharpe_statistics': {
                'mean': np.mean(sharpes) if sharpes else 0,
                'median': np.median(sharpes) if sharpes else 0,
                'std': np.std(sharpes) if sharpes else 0
            },
            'risk_statistics': {
                'avg_max_drawdown': np.mean(drawdowns),
                'worst_drawdown': np.max(drawdowns),
                'best_drawdown': np.min(drawdowns)
            },
            'win_rate_statistics': {
                'mean': np.mean(win_rates),
                'median': np.median(win_rates),
                'std': np.std(win_rates)
            },
            'top_performers': sorted(bot_metrics, key=lambda x: x['total_return_pct'], reverse=True)[:10],
            'worst_performers': sorted(bot_metrics, key=lambda x: x['total_return_pct'])[:10]
        }
    
    def _analyze_kelly_performance(self, bots):
        """Analyze Kelly Criterion effectiveness"""
        kelly_data = []
        
        for bot in bots:
            if bot.trade_history:
                for trade in bot.trade_history:
                    if 'kelly_fraction' in trade:
                        kelly_data.append({
                            'bot_id': bot.bot_id,
                            'kelly_fraction': trade['kelly_fraction'],
                            'win_probability': trade.get('win_probability', 0),
                            'payoff_ratio': trade.get('payoff_ratio', 0),
                            'pnl': trade.get('pnl', 0),
                            'pips_gained': trade.get('pips_gained', 0)
                        })
        
        if not kelly_data:
            return {}
        
        kelly_df = pd.DataFrame(kelly_data)
        
        return {
            'kelly_fraction_stats': {
                'mean': kelly_df['kelly_fraction'].mean(),
                'median': kelly_df['kelly_fraction'].median(),
                'std': kelly_df['kelly_fraction'].std(),
                'min': kelly_df['kelly_fraction'].min(),
                'max': kelly_df['kelly_fraction'].max()
            },
            'kelly_vs_performance': {
                'correlation_pnl': kelly_df['kelly_fraction'].corr(kelly_df['pnl']),
                'correlation_pips': kelly_df['kelly_fraction'].corr(kelly_df['pips_gained'])
            },
            'win_probability_stats': {
                'mean': kelly_df['win_probability'].mean(),
                'std': kelly_df['win_probability'].std()
            },
            'payoff_ratio_stats': {
                'mean': kelly_df['payoff_ratio'].mean(),
                'std': kelly_df['payoff_ratio'].std()
            }
        }
    
    def _analyze_risk_metrics(self, bot_metrics):
        """Analyze risk-adjusted performance metrics"""
        if not bot_metrics:
            return {}
        
        # Calculate risk-adjusted returns
        risk_adjusted = []
        for m in bot_metrics:
            if m['max_drawdown'] > 0:
                calmar_ratio = m['total_return_pct'] / (m['max_drawdown'] * 100)
            else:
                calmar_ratio = float('inf') if m['total_return_pct'] > 0 else 0
            
            risk_adjusted.append({
                'bot_id': m['bot_id'],
                'return_pct': m['total_return_pct'],
                'max_drawdown': m['max_drawdown'],
                'sharpe_ratio': m['sharpe_ratio'],
                'calmar_ratio': calmar_ratio,
                'profit_factor': m.get('profit_factor', 0)
            })
        
        risk_df = pd.DataFrame(risk_adjusted)
        
        return {
            'calmar_ratio_stats': {
                'mean': risk_df['calmar_ratio'].mean(),
                'median': risk_df['calmar_ratio'].median(),
                'std': risk_df['calmar_ratio'].std()
            },
            'sharpe_distribution': {
                'positive_sharpe_pct': (risk_df['sharpe_ratio'] > 0).mean() * 100,
                'high_sharpe_pct': (risk_df['sharpe_ratio'] > 1.0).mean() * 100
            },
            'profit_factor_stats': {
                'mean': risk_df['profit_factor'].mean(),
                'median': risk_df['profit_factor'].median(),
                'profitable_bots_pct': (risk_df['profit_factor'] > 1.0).mean() * 100
            }
        }
    
    def _analyze_trade_patterns(self, bots):
        """Analyze trading patterns across the fleet"""
        all_trades = []
        
        for bot in bots:
            for trade in bot.trade_history:
                trade_data = trade.copy()
                trade_data['bot_id'] = bot.bot_id
                all_trades.append(trade_data)
        
        if not all_trades:
            return {}
        
        trades_df = pd.DataFrame(all_trades)
        
        return {
            'total_trades': len(all_trades),
            'signal_distribution': trades_df['signal'].value_counts().to_dict(),
            'exit_reason_distribution': trades_df['exit_reason'].value_counts().to_dict(),
            'trade_duration_stats': {
                'trades_per_bot_mean': len(all_trades) / len(bots),
                'trades_per_bot_std': trades_df.groupby('bot_id').size().std()
            },
            'pnl_distribution': {
                'mean': trades_df['pnl'].mean(),
                'median': trades_df['pnl'].median(),
                'std': trades_df['pnl'].std(),
                'positive_trades_pct': (trades_df['pnl'] > 0).mean() * 100
            }
        }
    
    def _create_performance_plots(self, bot_metrics, timestamp):
        """Create comprehensive performance visualization plots"""
        logger.info("Creating performance plots...")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Kelly Monte Carlo Bot Fleet Performance Analysis', fontsize=16)
        
        # 1. Return distribution
        returns = [m['total_return_pct'] for m in bot_metrics]
        axes[0, 0].hist(returns, bins=50, alpha=0.7, color='blue')
        axes[0, 0].axvline(np.mean(returns), color='red', linestyle='--', label=f'Mean: {np.mean(returns):.2f}%')
        axes[0, 0].set_xlabel('Return (%)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Return Distribution')
        axes[0, 0].legend()
        
        # 2. Sharpe ratio distribution
        sharpes = [m['sharpe_ratio'] for m in bot_metrics if not np.isnan(m['sharpe_ratio'])]
        if sharpes:
            axes[0, 1].hist(sharpes, bins=30, alpha=0.7, color='green')
            axes[0, 1].axvline(np.mean(sharpes), color='red', linestyle='--', label=f'Mean: {np.mean(sharpes):.2f}')
            axes[0, 1].set_xlabel('Sharpe Ratio')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Sharpe Ratio Distribution')
            axes[0, 1].legend()
        
        # 3. Max drawdown distribution
        drawdowns = [m['max_drawdown'] * 100 for m in bot_metrics]
        axes[0, 2].hist(drawdowns, bins=30, alpha=0.7, color='orange')
        axes[0, 2].axvline(np.mean(drawdowns), color='red', linestyle='--', label=f'Mean: {np.mean(drawdowns):.2f}%')
        axes[0, 2].set_xlabel('Max Drawdown (%)')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Max Drawdown Distribution')
        axes[0, 2].legend()
        
        # 4. Win rate distribution
        win_rates = [m['win_rate'] * 100 for m in bot_metrics]
        axes[1, 0].hist(win_rates, bins=30, alpha=0.7, color='purple')
        axes[1, 0].axvline(np.mean(win_rates), color='red', linestyle='--', label=f'Mean: {np.mean(win_rates):.1f}%')
        axes[1, 0].set_xlabel('Win Rate (%)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Win Rate Distribution')
        axes[1, 0].legend()
        
        # 5. Return vs Sharpe scatter
        if sharpes:
            sharpe_dict = {m['bot_id']: m['sharpe_ratio'] for m in bot_metrics if not np.isnan(m['sharpe_ratio'])}
            returns_for_sharpe = [m['total_return_pct'] for m in bot_metrics if m['bot_id'] in sharpe_dict]
            sharpes_for_plot = [sharpe_dict[m['bot_id']] for m in bot_metrics if m['bot_id'] in sharpe_dict]
            
            axes[1, 1].scatter(sharpes_for_plot, returns_for_sharpe, alpha=0.6)
            axes[1, 1].set_xlabel('Sharpe Ratio')
            axes[1, 1].set_ylabel('Return (%)')
            axes[1, 1].set_title('Return vs Sharpe Ratio')
        
        # 6. Profit factor distribution
        profit_factors = [m.get('profit_factor', 0) for m in bot_metrics if m.get('profit_factor', 0) < 10]  # Cap at 10 for visualization
        if profit_factors:
            axes[1, 2].hist(profit_factors, bins=30, alpha=0.7, color='cyan')
            axes[1, 2].axvline(np.mean(profit_factors), color='red', linestyle='--', label=f'Mean: {np.mean(profit_factors):.2f}')
            axes[1, 2].set_xlabel('Profit Factor')
            axes[1, 2].set_ylabel('Frequency')
            axes[1, 2].set_title('Profit Factor Distribution')
            axes[1, 2].legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.results_dir / f"kelly_fleet_performance_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Performance plots saved to {plot_file}")
    
    def _print_fleet_summary(self, fleet_metrics, bot_metrics):
        """Print a comprehensive fleet summary"""
        print("\\n" + "="*80)
        print("KELLY MONTE CARLO BOT FLEET SUMMARY")
        print("="*80)
        
        if fleet_metrics:
            print(f"Fleet Size: {fleet_metrics.get('n_active_bots', 0)} active bots")
            print(f"Total Trades: {fleet_metrics.get('total_trades', 0):,}")
            print(f"Fleet Win Rate: {fleet_metrics.get('fleet_win_rate', 0)*100:.1f}%")
            print(f"Total P&L: ${fleet_metrics.get('total_pnl', 0):,.2f}")
            print(f"Average Return: {fleet_metrics.get('average_return_pct', 0):.2f}%")
            print(f"Average Sharpe: {fleet_metrics.get('average_sharpe_ratio', 0):.2f}")
            print(f"Average Max DD: {fleet_metrics.get('average_max_drawdown', 0)*100:.2f}%")
        
        if bot_metrics:
            returns = [m['total_return_pct'] for m in bot_metrics]
            profitable_bots = sum(1 for r in returns if r > 0)
            print(f"\\nProfitable Bots: {profitable_bots}/{len(bot_metrics)} ({profitable_bots/len(bot_metrics)*100:.1f}%)")
            print(f"Best Performer: {max(returns):.2f}% return")
            print(f"Worst Performer: {min(returns):.2f}% return")
            print(f"Return Std Dev: {np.std(returns):.2f}%")
        
        print("="*80)

def run_fleet_simulation(n_bots: int = 2000, 
                        simulation_hours: int = 1000,
                        save_results: bool = True) -> BotFleetManager:\n    \"\"\"\n    Run complete Kelly Monte Carlo bot fleet simulation\n    \n    Args:\n        n_bots: Number of bots in the fleet\n        simulation_hours: Number of hours to simulate\n        save_results: Whether to save results\n        \n    Returns:\n        BotFleetManager instance with results\n    \"\"\"\n    logger.info(f\"Starting Kelly Monte Carlo fleet simulation with {n_bots} bots\")\n    logger.info(f\"Simulation duration: {simulation_hours} hours\")\n    \n    # Configure trading parameters for realistic simulation\n    params = TradingParameters(\n        max_risk_per_trade=0.02,  # 2% max risk per trade\n        stop_loss_pips=30.0,      # 30 pip stop loss\n        take_profit_pips=60.0,    # 60 pip take profit (2:1 RR)\n        monte_carlo_scenarios=500, # 500 MC scenarios for speed\n        update_frequency=25,       # Update parameters every 25 trades\n        rolling_history_size=500   # 500-period rolling history\n    )\n    \n    # Initialize fleet manager\n    fleet_manager = BotFleetManager(\n        n_bots=n_bots,\n        initial_equity=100000.0,  # $100k per bot\n        params=params\n    )\n    \n    # Load market data for simulation\n    data_manager = DataManager()\n    market_data = data_manager.load_h1_data(\"EURUSD\")\n    \n    logger.info(f\"Loaded {len(market_data)} hours of market data\")\n    \n    # Run simulation\n    simulation_start = time.time()\n    trades_executed = 0\n    \n    # Use a subset of data for simulation\n    sim_data = market_data.tail(simulation_hours)\n    \n    logger.info(f\"Running simulation on {len(sim_data)} hours of data...\")\n    \n    for i, (timestamp, row) in enumerate(sim_data.iterrows()):\n        if i % 100 == 0:\n            logger.info(f\"Processing hour {i+1}/{len(sim_data)} - Trades executed: {trades_executed}\")\n        \n        current_price = row['close']\n        \n        # Get trading decisions from all bots\n        decisions = fleet_manager.run_parallel_decisions(\n            current_price=current_price,\n            market_data=row,\n            timestamp=timestamp\n        )\n        \n        # Execute trades for bots with decisions\n        for j, decision in enumerate(decisions):\n            if decision is not None:\n                try:\n                    bot = fleet_manager.bots[j]\n                    \n                    # Execute trade if bot doesn't have open position\n                    if bot.current_position is None:\n                        bot.execute_trade(decision)\n                        trades_executed += 1\n                \n                except Exception as e:\n                    logger.error(f\"Error executing trade for bot {j}: {e}\")\n        \n        # Check for trade closures (simplified - close after random time)\n        for bot in fleet_manager.bots:\n            if (bot.current_position is not None and \n                np.random.random() < 0.1):  # 10% chance to close each hour\n                try:\n                    # Simulate price movement\n                    price_change = np.random.normal(0, 0.001) * current_price\n                    exit_price = current_price + price_change\n                    \n                    # Determine exit reason based on price movement\n                    entry_price = bot.current_position['entry_price']\n                    signal = bot.current_position['signal']\n                    \n                    if signal == 'BUY':\n                        pips_moved = (exit_price - entry_price) / params.pip_value\n                    else:\n                        pips_moved = (entry_price - exit_price) / params.pip_value\n                    \n                    if pips_moved >= params.take_profit_pips:\n                        exit_reason = \"TAKE_PROFIT\"\n                    elif pips_moved <= -params.stop_loss_pips:\n                        exit_reason = \"STOP_LOSS\"\n                    else:\n                        exit_reason = \"NATURAL\"\n                    \n                    bot.close_trade(exit_price, exit_reason)\n                \n                except Exception as e:\n                    logger.error(f\"Error closing trade for bot {bot.bot_id}: {e}\")\n    \n    simulation_time = time.time() - simulation_start\n    logger.info(f\"Simulation completed in {simulation_time:.2f} seconds\")\n    logger.info(f\"Total trades executed: {trades_executed:,}\")\n    \n    # Save results if requested\n    if save_results:\n        timestamp = datetime.now().strftime(\"%Y%m%d_%H%M%S\")\n        results_file = f\"kelly_fleet_results_{timestamp}.json\"\n        fleet_manager.save_results(results_file)\n        logger.info(f\"Results saved to {results_file}\")\n    \n    return fleet_manager\n\ndef main():\n    \"\"\"Main execution function\"\"\"\n    print(\"Kelly Monte Carlo FOREX Bot Fleet\")\n    print(\"=\" * 50)\n    \n    # Configuration\n    N_BOTS = int(os.getenv('N_BOTS', '2000'))  # Default 2000 bots\n    SIMULATION_HOURS = int(os.getenv('SIM_HOURS', '1000'))  # Default 1000 hours\n    \n    print(f\"Fleet Size: {N_BOTS} bots\")\n    print(f\"Simulation: {SIMULATION_HOURS} hours\")\n    print(f\"CPU Cores Available: {mp.cpu_count()}\")\n    \n    try:\n        # Run the fleet simulation\n        fleet_manager = run_fleet_simulation(\n            n_bots=N_BOTS,\n            simulation_hours=SIMULATION_HOURS,\n            save_results=True\n        )\n        \n        # Analyze results\n        analyzer = KellyFleetAnalyzer()\n        analysis = analyzer.analyze_fleet_performance(\n            fleet_manager=fleet_manager,\n            save_plots=True\n        )\n        \n        logger.info(\"Kelly Monte Carlo bot fleet simulation completed successfully!\")\n        \n    except KeyboardInterrupt:\n        logger.info(\"Simulation interrupted by user\")\n    except Exception as e:\n        logger.error(f\"Simulation failed: {e}\")\n        raise\n\nif __name__ == \"__main__\":\n    main()\n
