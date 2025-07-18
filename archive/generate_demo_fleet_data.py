#!/usr/bin/env python3
"""
Quick Demo Script to Generate Fleet Results for Dashboard Testing
Creates realistic bot trading data with trade history for GUI visualization
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

def generate_demo_fleet_data(n_bots=20, trades_per_bot_range=(10, 100)):
    """Generate demo fleet data with realistic trading results"""
    
    fleet_data = {
        "fleet_performance": {},
        "bot_metrics": [],
        "parameters": {
            "n_bots": n_bots,
            "initial_equity": 100000.0,
            "trading_params": {
                "max_risk_per_trade": 0.02,
                "stop_loss_pips": 30.0,
                "take_profit_pips": 60.0
            }
        }
    }
    
    all_bot_metrics = []
    total_trades = 0
    total_winning = 0
    total_pnl = 0
    total_equity = 0
    
    base_time = datetime.now() - timedelta(days=30)
    
    for bot_id in range(n_bots):
        initial_equity = 100000.0
        n_trades = random.randint(*trades_per_bot_range)
        
        # Generate realistic trading history
        trade_history = []
        current_equity = initial_equity
        winning_trades = 0
        bot_total_pnl = 0
        
        for trade_id in range(1, n_trades + 1):
            # Simulate realistic win/loss with 45-55% win rate
            is_win = random.random() < (0.45 + random.random() * 0.1)
            
            # Realistic PnL: wins around +$200-800, losses around -$150-400
            if is_win:
                pnl = random.uniform(150, 800)
                winning_trades += 1
            else:
                pnl = -random.uniform(100, 450)
            
            current_equity += pnl
            bot_total_pnl += pnl
            
            trade_time = base_time + timedelta(hours=trade_id * random.uniform(1, 8))
            
            trade = {
                "trade_id": trade_id,
                "bot_id": bot_id,
                "timestamp": trade_time.isoformat(),
                "signal": random.choice(["BUY", "SELL"]),
                "entry_price": round(1.2000 + random.uniform(-0.1, 0.1), 5),
                "exit_price": round(1.2000 + random.uniform(-0.1, 0.1), 5),
                "position_size": random.uniform(10000, 50000),
                "pnl": round(pnl, 2),
                "status": "CLOSED",
                "exit_reason": "TAKE_PROFIT" if is_win else "STOP_LOSS",
                "equity_after": round(current_equity, 2)
            }
            trade_history.append(trade)
        
        # Calculate bot performance metrics
        win_rate = winning_trades / n_trades if n_trades > 0 else 0
        total_return_pct = (current_equity - initial_equity) / initial_equity * 100
        
        # Calculate Sharpe ratio (simplified)
        if n_trades > 1:
            returns = [t["pnl"] / initial_equity for t in trade_history]
            returns_mean = np.mean(returns)
            returns_std = np.std(returns)
            sharpe_ratio = (returns_mean / returns_std * np.sqrt(252)) if returns_std > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Calculate max drawdown
        peak_equity = initial_equity
        max_drawdown = 0
        for trade in trade_history:
            equity = trade["equity_after"]
            if equity > peak_equity:
                peak_equity = equity
            drawdown = (peak_equity - equity) / peak_equity
            max_drawdown = max(max_drawdown, drawdown)
        
        bot_metrics = {
            "bot_id": bot_id,
            "total_trades": n_trades,
            "winning_trades": winning_trades,
            "win_rate": win_rate,
            "total_pnl": round(bot_total_pnl, 2),
            "total_return_pct": total_return_pct,
            "current_equity": round(current_equity, 2),
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "profit_factor": abs(sum(t["pnl"] for t in trade_history if t["pnl"] > 0) / 
                                sum(t["pnl"] for t in trade_history if t["pnl"] < 0)) if any(t["pnl"] < 0 for t in trade_history) else float('inf'),
            "average_win": np.mean([t["pnl"] for t in trade_history if t["pnl"] > 0]) if winning_trades > 0 else 0,
            "average_loss": np.mean([t["pnl"] for t in trade_history if t["pnl"] < 0]) if (n_trades - winning_trades) > 0 else 0,
            "total_pips": sum(random.uniform(-30, 60) for _ in range(n_trades)),  # Simulate pip gains
            "trade_history": trade_history
        }
        
        all_bot_metrics.append(bot_metrics)
        
        # Aggregate fleet stats
        total_trades += n_trades
        total_winning += winning_trades
        total_pnl += bot_total_pnl
        total_equity += current_equity
    
    # Sort bots by total capital (current_equity) for top performers
    all_bot_metrics.sort(key=lambda x: x["current_equity"], reverse=True)
    
    # Calculate fleet performance
    fleet_performance = {
        "n_active_bots": n_bots,
        "total_trades": total_trades,
        "total_winning_trades": total_winning,
        "fleet_win_rate": total_winning / total_trades if total_trades > 0 else 0,
        "total_pnl": round(total_pnl, 2),
        "total_equity": round(total_equity, 2),
        "average_return_pct": np.mean([bot["total_return_pct"] for bot in all_bot_metrics]),
        "average_sharpe_ratio": np.mean([bot["sharpe_ratio"] for bot in all_bot_metrics if not np.isnan(bot["sharpe_ratio"])]),
        "average_max_drawdown": np.mean([bot["max_drawdown"] for bot in all_bot_metrics]),
        "best_performer": max(all_bot_metrics, key=lambda x: x["total_return_pct"]),
        "worst_performer": min(all_bot_metrics, key=lambda x: x["total_return_pct"])
    }
    
    fleet_data["fleet_performance"] = fleet_performance
    fleet_data["bot_metrics"] = all_bot_metrics
    
    return fleet_data

def main():
    """Generate demo data and save to JSON file"""
    print("🚀 Generating demo fleet results for dashboard testing...")
    
    # Generate demo data with 50 bots for a good demo
    demo_data = generate_demo_fleet_data(n_bots=50, trades_per_bot_range=(20, 150))
    
    # Save to file
    filename = "fleet_results.json"
    with open(filename, 'w') as f:
        json.dump(demo_data, f, indent=2, default=str)
    
    print(f"✅ Demo fleet results saved to: {filename}")
    print(f"📊 Generated data for {demo_data['fleet_performance']['n_active_bots']} bots")
    print(f"💼 Total trades: {demo_data['fleet_performance']['total_trades']:,}")
    print(f"💰 Fleet total P&L: ${demo_data['fleet_performance']['total_pnl']:,.2f}")
    print(f"🏆 Best performer: Bot #{demo_data['fleet_performance']['best_performer']['bot_id']} (+{demo_data['fleet_performance']['best_performer']['total_return_pct']:.2f}%)")
    print(f"📉 Worst performer: Bot #{demo_data['fleet_performance']['worst_performer']['bot_id']} ({demo_data['fleet_performance']['worst_performer']['total_return_pct']:.2f}%)")
    
    print("\n🎯 Now you can run the dashboard:")
    print("python kelly_bot_dashboard.py --results fleet_results.json")

if __name__ == "__main__":
    main()
