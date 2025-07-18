#!/usr/bin/env python3
"""
Live Demo System - Simulates Real Trading with Dynamic Updates
Creates live-updating fleet data to demonstrate the real-time GUI
"""

import json
import time
import numpy as np
from datetime import datetime
from pathlib import Path
import threading
import signal
import sys

class LiveDemoSystem:
    """Simulates live trading fleet with dynamic performance updates"""
    
    def __init__(self, n_bots: int = 100):
        self.n_bots = n_bots
        self.is_running = False
        self.results_file = "fleet_results.json"
        self.update_interval = 3.0  # Update every 3 seconds
        
        # Initialize bot data
        self.bot_data = self._initialize_bots()
        
    def _initialize_bots(self):
        """Initialize bot fleet with realistic starting data"""
        np.random.seed(42)
        bots = []
        
        for bot_id in range(self.n_bots):
            base_equity = 100000.0
            initial_performance = np.random.normal(1.0, 0.05)  # Start near break-even
            
            bot = {
                'bot_id': bot_id,
                'initial_equity': base_equity,
                'current_equity': base_equity * initial_performance,
                'total_trades': np.random.randint(20, 100),
                'win_rate': np.random.uniform(0.4, 0.7),
                'sharpe_ratio': np.random.normal(0.5, 0.3),
                'max_drawdown': np.random.uniform(0.05, 0.2),
                'momentum': np.random.normal(0, 0.001),  # Price momentum
                'volatility': np.random.uniform(0.02, 0.08),  # Individual volatility
                'last_update': datetime.now()
            }
            
            # Calculate derived metrics
            bot['total_pnl'] = bot['current_equity'] - bot['initial_equity']
            bot['total_return_pct'] = (bot['total_pnl'] / bot['initial_equity']) * 100
            bot['winning_trades'] = int(bot['total_trades'] * bot['win_rate'])
            bot['profit_factor'] = np.random.uniform(0.8, 2.0)
            bot['average_win'] = abs(bot['total_pnl']) / max(1, bot['winning_trades']) if bot['winning_trades'] > 0 else 0
            bot['average_loss'] = abs(bot['total_pnl']) / max(1, bot['total_trades'] - bot['winning_trades']) if bot['total_trades'] > bot['winning_trades'] else 0
            bot['total_pips'] = np.random.uniform(-200, 800)
            bot['trade_history'] = []
            
            bots.append(bot)
        
        return bots
    
    def _update_bot_performance(self, bot):
        """Simulate realistic bot performance updates"""
        # Market factor (affects all bots)
        market_factor = np.random.normal(0, 0.002)
        
        # Individual bot performance
        individual_factor = np.random.normal(bot['momentum'], bot['volatility'])
        
        # Combined performance change
        total_change = market_factor + individual_factor
        
        # Update equity
        old_equity = bot['current_equity']
        bot['current_equity'] *= (1 + total_change)
        
        # Prevent equity from going below 10% of initial
        bot['current_equity'] = max(bot['current_equity'], bot['initial_equity'] * 0.1)
        
        # Update derived metrics
        bot['total_pnl'] = bot['current_equity'] - bot['initial_equity']
        bot['total_return_pct'] = (bot['total_pnl'] / bot['initial_equity']) * 100
        
        # Occasionally add trades
        if np.random.random() < 0.1:  # 10% chance per update
            bot['total_trades'] += 1
            
            # Update win rate with some randomness
            if np.random.random() < bot['win_rate']:
                bot['winning_trades'] += 1
            
            bot['win_rate'] = bot['winning_trades'] / bot['total_trades']
        
        # Update Sharpe ratio with some drift
        bot['sharpe_ratio'] += np.random.normal(0, 0.05)
        bot['sharpe_ratio'] = np.clip(bot['sharpe_ratio'], -2.0, 3.0)
        
        # Update drawdown
        if bot['current_equity'] < old_equity:
            current_dd = abs(bot['total_pnl']) / bot['current_equity'] if bot['current_equity'] > 0 else 0.5
            bot['max_drawdown'] = max(bot['max_drawdown'], current_dd)
        
        # Update momentum (trend following)
        performance_change = (bot['current_equity'] - old_equity) / old_equity
        bot['momentum'] = 0.9 * bot['momentum'] + 0.1 * performance_change
        
        bot['last_update'] = datetime.now()
        
    def _save_fleet_results(self):
        """Save current fleet state in the format expected by GUI"""
        # Sort bots by current equity (descending)
        sorted_bots = sorted(self.bot_data, key=lambda x: x['current_equity'], reverse=True)
        top_20 = sorted_bots[:20]
        
        # Calculate fleet metrics
        fleet_metrics = {
            'n_active_bots': len([b for b in self.bot_data if b['total_trades'] > 0]),
            'total_trades': sum(b['total_trades'] for b in self.bot_data),
            'total_winning_trades': sum(b['winning_trades'] for b in self.bot_data),
            'fleet_win_rate': np.mean([b['win_rate'] for b in self.bot_data]),
            'total_pnl': sum(b['total_pnl'] for b in self.bot_data),
            'total_equity': sum(b['current_equity'] for b in self.bot_data),
            'average_return_pct': np.mean([b['total_return_pct'] for b in self.bot_data]),
            'average_sharpe_ratio': np.mean([b['sharpe_ratio'] for b in self.bot_data]),
            'average_max_drawdown': np.mean([b['max_drawdown'] for b in self.bot_data]),
            'best_performer': max(self.bot_data, key=lambda x: x['total_return_pct']),
            'worst_performer': min(self.bot_data, key=lambda x: x['total_return_pct'])
        }
        
        # Create results structure
        results = {
            'fleet_performance': fleet_metrics,
            'bot_metrics': top_20,  # Only save top 20 for GUI
            'parameters': {
                'n_bots': self.n_bots,
                'initial_equity': 100000.0,
                'training_progress': 100.0,  # Simulation mode
                'current_batch': 1000,
                'total_batches': 1000
            },
            'timestamp': datetime.now().isoformat(),
            'training_status': 'live_demo'
        }
        
        # Save to file
        with open(self.results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    def start_live_demo(self):
        """Start the live demo simulation"""
        print("🚀 Starting Live Demo System")
        print(f"📊 Simulating {self.n_bots} trading bots with real-time updates")
        print(f"🔄 Updates every {self.update_interval} seconds")
        print(f"💾 Saving to: {self.results_file}")
        print("🛑 Press Ctrl+C to stop")
        print("=" * 50)
        
        self.is_running = True
        update_count = 0
        
        try:
            while self.is_running:
                update_count += 1
                
                # Update all bots
                for bot in self.bot_data:
                    self._update_bot_performance(bot)
                
                # Save results
                self._save_fleet_results()
                
                # Display progress
                top_bot = max(self.bot_data, key=lambda x: x['current_equity'])
                worst_bot = min(self.bot_data, key=lambda x: x['current_equity'])
                
                print(f"Update #{update_count:4d} | "
                      f"Best: Bot #{top_bot['bot_id']:03d} (${top_bot['current_equity']:,.0f}) | "
                      f"Worst: Bot #{worst_bot['bot_id']:03d} (${worst_bot['current_equity']:,.0f}) | "
                      f"Fleet: ${sum(b['current_equity'] for b in self.bot_data):,.0f}")
                
                time.sleep(self.update_interval)
                
        except KeyboardInterrupt:
            print("\n🛑 Demo stopped by user")
            self.stop()
    
    def stop(self):
        """Stop the live demo"""
        self.is_running = False
        print("📊 Final fleet summary:")
        
        sorted_bots = sorted(self.bot_data, key=lambda x: x['current_equity'], reverse=True)
        
        print(f"   🏆 Top performer: Bot #{sorted_bots[0]['bot_id']} - ${sorted_bots[0]['current_equity']:,.2f} ({sorted_bots[0]['total_return_pct']:+.2f}%)")
        print(f"   📉 Worst performer: Bot #{sorted_bots[-1]['bot_id']} - ${sorted_bots[-1]['current_equity']:,.2f} ({sorted_bots[-1]['total_return_pct']:+.2f}%)")
        print(f"   💰 Fleet total: ${sum(b['current_equity'] for b in self.bot_data):,.2f}")
        print(f"   📈 Average return: {np.mean([b['total_return_pct'] for b in self.bot_data]):+.2f}%")

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n🛑 Received interrupt signal...")
    if hasattr(signal_handler, 'demo_system'):
        signal_handler.demo_system.stop()
    sys.exit(0)

def main():
    """Main demo entry point"""
    print("🎯 Live Fleet Demo System")
    print("========================")
    
    # Parse command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("""
Usage:
  python3 kelly_demo_live.py                    # Start live demo with 100 bots
  python3 kelly_demo_live.py --bots 200         # Start with 200 bots
  python3 kelly_demo_live.py --interval 1       # Update every 1 second
  python3 kelly_demo_live.py --help             # Show this help

Controls:
  Ctrl+C                                        # Stop demo gracefully
  
Files Created:
  fleet_results.json                            # Real-time data for GUI
  
Recommended Usage:
  1. Terminal 1: python3 kelly_demo_live.py
  2. Terminal 2: python3 integrated_training_with_gui.py
  3. Watch real-time updates in GUI!
""")
        return
    
    # Configuration
    n_bots = 100
    update_interval = 3.0
    
    # Parse arguments
    for i, arg in enumerate(sys.argv):
        if arg == "--bots" and i + 1 < len(sys.argv):
            n_bots = int(sys.argv[i + 1])
        elif arg == "--interval" and i + 1 < len(sys.argv):
            update_interval = float(sys.argv[i + 1])
    
    # Create and start demo
    demo_system = LiveDemoSystem(n_bots=n_bots)
    demo_system.update_interval = update_interval
    
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal_handler.demo_system = demo_system
    
    # Start the demo
    demo_system.start_live_demo()

if __name__ == "__main__":
    main()
