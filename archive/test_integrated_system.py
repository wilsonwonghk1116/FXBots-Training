#!/usr/bin/env python3
"""
Quick Test Script for Integrated Training System
Creates sample data and tests GUI components
"""

import json
import time
import numpy as np
from datetime import datetime
from pathlib import Path

def create_sample_fleet_data():
    """Create sample fleet results for GUI testing"""
    print("🧪 Creating sample fleet data for GUI testing...")
    
    # Generate realistic bot performance data
    np.random.seed(42)
    bot_metrics = []
    
    for bot_id in range(100):  # Generate 100 bots, GUI will show top 20
        base_equity = 100000.0
        
        # Create varied performance levels
        if bot_id < 10:  # Top 10% high performers
            performance_factor = np.random.normal(1.15, 0.05)  # 15% avg gain
        elif bot_id < 30:  # Next 20% good performers
            performance_factor = np.random.normal(1.08, 0.08)  # 8% avg gain
        elif bot_id < 70:  # 40% average performers
            performance_factor = np.random.normal(1.02, 0.12)  # 2% avg gain
        else:  # Bottom 30% underperformers
            performance_factor = np.random.normal(0.96, 0.15)  # -4% avg loss
        
        current_equity = base_equity * performance_factor
        total_pnl = current_equity - base_equity
        win_rate = np.random.uniform(0.35, 0.85)
        total_trades = int(np.random.uniform(25, 350))
        sharpe_ratio = np.random.uniform(-1.5, 3.0)
        max_drawdown = np.random.uniform(0.01, 0.35)
        
        bot_metrics.append({
            'bot_id': bot_id,
            'total_trades': total_trades,
            'winning_trades': int(total_trades * win_rate),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return_pct': (total_pnl / base_equity) * 100,
            'current_equity': current_equity,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'profit_factor': np.random.uniform(0.5, 3.5),
            'average_win': total_pnl / (total_trades * win_rate) if win_rate > 0 else 0,
            'average_loss': -total_pnl / (total_trades * (1 - win_rate)) if win_rate < 1 else 0,
            'total_pips': np.random.uniform(-1000, 2500),
            'trade_history': []
        })
    
    # Sort by current equity to get realistic top performers
    bot_metrics.sort(key=lambda x: x['current_equity'], reverse=True)
    
    # Create fleet summary
    total_equity = sum(b['current_equity'] for b in bot_metrics)
    total_pnl = sum(b['total_pnl'] for b in bot_metrics)
    avg_return = np.mean([b['total_return_pct'] for b in bot_metrics])
    
    fleet_results = {
        'fleet_performance': {
            'n_active_bots': len(bot_metrics),
            'total_trades': sum(b['total_trades'] for b in bot_metrics),
            'total_winning_trades': sum(b['winning_trades'] for b in bot_metrics),
            'fleet_win_rate': np.mean([b['win_rate'] for b in bot_metrics]),
            'total_pnl': total_pnl,
            'total_equity': total_equity,
            'average_return_pct': avg_return,
            'average_sharpe_ratio': np.mean([b['sharpe_ratio'] for b in bot_metrics]),
            'average_max_drawdown': np.mean([b['max_drawdown'] for b in bot_metrics]),
            'best_performer': max(bot_metrics, key=lambda x: x['total_return_pct']),
            'worst_performer': min(bot_metrics, key=lambda x: x['total_return_pct'])
        },
        'bot_metrics': bot_metrics,
        'parameters': {
            'n_bots': len(bot_metrics),
            'initial_equity': 100000.0,
            'training_progress': 65.5,  # Simulated training progress
            'current_batch': 327,
            'total_batches': 500
        },
        'timestamp': datetime.now().isoformat(),
        'training_status': 'demo_mode'
    }
    
    # Save sample data
    with open("fleet_results.json", 'w') as f:
        json.dump(fleet_results, f, indent=2, default=str)
    
    print(f"✅ Created sample data:")
    print(f"   • {len(bot_metrics)} bots generated")
    print(f"   • Top performer: Bot #{bot_metrics[0]['bot_id']} with {bot_metrics[0]['total_return_pct']:.2f}% return")
    print(f"   • Fleet total equity: ${total_equity:,.2f}")
    print(f"   • Saved to: fleet_results.json")
    
    return fleet_results

def simulate_live_updates():
    """Simulate live trading updates for testing"""
    print("\n🔄 Starting live update simulation...")
    print("   (This will update the data every 3 seconds)")
    print("   (Launch the GUI in another terminal to see real-time updates)")
    print("   (Press Ctrl+C to stop)")
    
    try:
        update_count = 0
        while True:
            # Load existing data
            if Path("fleet_results.json").exists():
                with open("fleet_results.json", 'r') as f:
                    data = json.load(f)
                
                # Simulate market movements - update bot performances
                for bot in data['bot_metrics']:
                    # Small random changes to simulate live trading
                    change_factor = np.random.normal(1.0, 0.002)  # 0.2% volatility
                    bot['current_equity'] *= change_factor
                    bot['total_pnl'] = bot['current_equity'] - 100000.0
                    bot['total_return_pct'] = (bot['total_pnl'] / 100000.0) * 100
                    
                    # Occasionally add a trade
                    if np.random.random() < 0.1:  # 10% chance
                        bot['total_trades'] += 1
                        if np.random.random() < bot['win_rate']:
                            bot['winning_trades'] += 1
                
                # Re-sort by current equity (this is key for top 20 ranking)
                data['bot_metrics'].sort(key=lambda x: x['current_equity'], reverse=True)
                
                # Update fleet summary
                data['fleet_performance']['total_equity'] = sum(b['current_equity'] for b in data['bot_metrics'])
                data['fleet_performance']['total_pnl'] = sum(b['total_pnl'] for b in data['bot_metrics'])
                data['fleet_performance']['average_return_pct'] = np.mean([b['total_return_pct'] for b in data['bot_metrics']])
                data['fleet_performance']['best_performer'] = max(data['bot_metrics'], key=lambda x: x['total_return_pct'])
                data['fleet_performance']['worst_performer'] = min(data['bot_metrics'], key=lambda x: x['total_return_pct'])
                
                # Update metadata
                data['timestamp'] = datetime.now().isoformat()
                update_count += 1
                data['parameters']['training_progress'] = min(99.9, 65.5 + (update_count * 0.5))
                
                # Save updated data
                with open("fleet_results.json", 'w') as f:
                    json.dump(data, f, indent=2, default=str)
                
                # Progress indicator
                top_bot = data['bot_metrics'][0]
                print(f"📊 Update #{update_count}: Top performer is Bot #{top_bot['bot_id']} with ${top_bot['current_equity']:,.2f} capital ({top_bot['total_return_pct']:+.2f}%)")
                
            time.sleep(3)  # Update every 3 seconds
            
    except KeyboardInterrupt:
        print("\n⏹️  Live update simulation stopped.")

def test_gui_components():
    """Test GUI availability"""
    print("\n🧪 Testing GUI components...")
    
    try:
        import PyQt6
        print("✅ PyQt6: Available")
        
        from PyQt6.QtWidgets import QApplication
        print("✅ QApplication: Can import")
        
        # Test if display is available (for headless systems)
        import sys
        app = QApplication(sys.argv)
        print("✅ Display: Available")
        app.quit()
        
        return True
        
    except ImportError as e:
        print(f"❌ PyQt6 not available: {e}")
        print("   Install with: pip install PyQt6")
        return False
    except Exception as e:
        print(f"⚠️  Display issue: {e}")
        print("   (This is normal for headless systems)")
        return False

def main():
    """Main test function"""
    print("🚀 Testing Integrated Training System")
    print("=" * 50)
    
    # Test GUI components
    gui_available = test_gui_components()
    
    # Create sample data
    fleet_data = create_sample_fleet_data()
    
    print(f"\n📋 Test Summary:")
    print(f"   • GUI Components: {'✅ Ready' if gui_available else '❌ Issues detected'}")
    print(f"   • Sample Data: ✅ Created")
    print(f"   • File: fleet_results.json")
    
    # Ask user what to do next
    print(f"\n🎯 Next Steps:")
    print(f"   1. Launch GUI: python3 integrated_training_with_gui.py")
    print(f"   2. Or use launcher: ./launch_training_gui.sh")
    print(f"   3. Or simulate live updates: python3 test_integrated_system.py --live")
    
    # Check for live mode
    if "--live" in sys.argv:
        simulate_live_updates()

if __name__ == "__main__":
    import sys
    main()
