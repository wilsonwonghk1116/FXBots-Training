#!/bin/bash
# 🚀 COMPLETE TRAINING SYSTEM LAUNCHER
# Integrated Kelly Monte Carlo Fleet with Real-Time GUI Dashboard

clear
echo "🚀 KELLY MONTE CARLO FLEET TRAINING SYSTEM"
echo "=========================================="
echo ""
echo "🎯 Real-time monitoring of TOP 20 bots ranked by total capital"
echo "📊 Live updates every 2-3 seconds with auto-sorting"
echo "💰 Track: PnL, Win Rate, Sharpe Ratio, Drawdown, and more!"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}Select your option:${NC}"
echo ""
echo "1. 🔥 FULL RAY TRAINING + GUI (Distributed 2-PC System)"
echo "2. 🎮 LIVE DEMO + GUI (Simulated Real-time Trading)"
echo "3. 🖥️  GUI ONLY (Monitor existing results)"
echo "4. ⚙️  RAY TRAINING ONLY (Background training)"
echo "5. 🧪 TEST SYSTEM (Verify components)"
echo "6. ❓ HELP (Show detailed usage)"
echo ""
echo -n "Enter choice [1-6]: "
read choice

case $choice in
    1)
        echo ""
        echo -e "${GREEN}🚀 LAUNCHING FULL TRAINING SYSTEM${NC}"
        echo "=================================="
        echo ""
        echo "This will:"
        echo "• Start Ray distributed training (2000 bots)"
        echo "• Launch real-time GUI dashboard"
        echo "• Monitor top 20 performers by total capital"
        echo "• Target 75% CPU/GPU/VRAM utilization"
        echo ""
        echo -e "${YELLOW}⚠️  Ensure your Ray cluster is running first!${NC}"
        echo "   PC1 (head): ray start --head --port=8265"
        echo "   PC2 (worker): ray start --address='PC1_IP:10001'"
        echo ""
        echo -n "Continue? [y/N]: "
        read confirm
        
        if [[ $confirm =~ ^[Yy]$ ]]; then
            echo ""
            echo -e "${BLUE}Starting integrated training system...${NC}"
            python3 integrated_training_with_gui.py
        else
            echo "Cancelled."
        fi
        ;;
        
    2)
        echo ""
        echo -e "${GREEN}🎮 LAUNCHING LIVE DEMO SYSTEM${NC}"
        echo "============================="
        echo ""
        echo "This will:"
        echo "• Simulate 100 trading bots with realistic performance"
        echo "• Update every 3 seconds with live data"
        echo "• Launch GUI to show top 20 performers"
        echo "• Perfect for testing and demonstration"
        echo ""
        echo -e "${CYAN}Opening two terminals:${NC}"
        echo "• Terminal 1: Live demo simulation"
        echo "• Terminal 2: Real-time GUI dashboard"
        echo ""
        
        # Start live demo in background
        echo -e "${BLUE}Starting live demo simulation...${NC}"
        gnome-terminal --title="Live Demo - Bot Simulation" -- bash -c "python3 kelly_demo_live.py; read -p 'Press Enter to close'"
        
        sleep 2
        
        # Start GUI
        echo -e "${BLUE}Starting GUI dashboard...${NC}"
        python3 integrated_training_with_gui.py
        ;;
        
    3)
        echo ""
        echo -e "${GREEN}🖥️  LAUNCHING GUI MONITOR${NC}"
        echo "========================="
        echo ""
        echo "This will:"
        echo "• Launch GUI dashboard only"
        echo "• Monitor existing fleet_results.json"
        echo "• Show top 20 performers by total capital"
        echo "• Refresh data every 3 seconds"
        echo ""
        
        if [ -f "fleet_results.json" ]; then
            echo -e "${GREEN}✅ Found existing data file${NC}"
        else
            echo -e "${YELLOW}⚠️  No existing data file found${NC}"
            echo "Creating sample data for demonstration..."
            python3 -c "
import json
import numpy as np
from datetime import datetime

np.random.seed(42)
bots = []
for i in range(20):
    equity = 100000 * np.random.normal(1.1, 0.2)
    pnl = equity - 100000
    bots.append({
        'bot_id': i,
        'current_equity': equity,
        'total_pnl': pnl,
        'total_return_pct': (pnl/100000)*100,
        'win_rate': np.random.uniform(0.4, 0.8),
        'total_trades': np.random.randint(50, 200),
        'sharpe_ratio': np.random.uniform(-0.5, 2.0),
        'max_drawdown': np.random.uniform(0.05, 0.3),
        'winning_trades': 0
    })

bots.sort(key=lambda x: x['current_equity'], reverse=True)

result = {
    'bot_metrics': bots,
    'fleet_performance': {'n_active_bots': 20},
    'timestamp': datetime.now().isoformat()
}

with open('fleet_results.json', 'w') as f:
    json.dump(result, f, indent=2)
print('✅ Sample data created')
"
        fi
        
        echo -e "${BLUE}Starting GUI dashboard...${NC}"
        python3 integrated_training_with_gui.py
        ;;
        
    4)
        echo ""
        echo -e "${GREEN}⚙️  LAUNCHING RAY TRAINING ONLY${NC}"
        echo "=============================="
        echo ""
        echo "This will:"
        echo "• Start Ray distributed training only"
        echo "• Process 2000 bots across cluster"
        echo "• Save results to timestamped files"
        echo "• No GUI interface"
        echo ""
        echo -e "${YELLOW}⚠️  Ensure Ray cluster is running!${NC}"
        echo ""
        echo -n "Continue? [y/N]: "
        read confirm
        
        if [[ $confirm =~ ^[Yy]$ ]]; then
            echo -e "${BLUE}Starting Ray training...${NC}"
            python3 ray_kelly_ultimate_75_percent.py
        else
            echo "Cancelled."
        fi
        ;;
        
    5)
        echo ""
        echo -e "${GREEN}🧪 TESTING SYSTEM COMPONENTS${NC}"
        echo "============================"
        echo ""
        echo "This will:"
        echo "• Test PyQt6 GUI components"
        echo "• Test Ray distributed computing"
        echo "• Create sample data files"
        echo "• Verify all dependencies"
        echo ""
        
        echo -e "${BLUE}Running system tests...${NC}"
        python3 -c "
import sys
print('🐍 Python:', sys.version)

try:
    import PyQt6
    print('✅ PyQt6: Available')
except ImportError:
    print('❌ PyQt6: Not found - Run: pip install PyQt6')

try:
    import ray
    print('✅ Ray: Available')
except ImportError:
    print('❌ Ray: Not found - Run: pip install ray[default]')

try:
    import torch
    print('✅ PyTorch: Available')
    if torch.cuda.is_available():
        print(f'✅ CUDA: {torch.cuda.device_count()} GPU(s) detected')
    else:
        print('⚠️  CUDA: No GPUs detected')
except ImportError:
    print('❌ PyTorch: Not found - Run: pip install torch')

try:
    import pandas, numpy
    print('✅ Data libraries: Available')
except ImportError:
    print('❌ Data libraries: Missing - Run: pip install pandas numpy')
"
        
        echo ""
        echo -e "${BLUE}Creating test data...${NC}"
        python3 test_integrated_system.py
        
        echo ""
        echo -e "${GREEN}🎯 Test completed!${NC}"
        ;;
        
    6)
        echo ""
        echo -e "${GREEN}❓ DETAILED USAGE GUIDE${NC}"
        echo "======================"
        echo ""
        echo -e "${CYAN}🚀 FULL TRAINING SYSTEM:${NC}"
        echo "• Requires 2-PC Ray cluster setup"
        echo "• Processes 2000 bots with 300k scenarios each"
        echo "• Real-time GUI shows top 20 by total capital"
        echo "• Targets 75% CPU/GPU/VRAM utilization"
        echo ""
        echo -e "${CYAN}🎮 LIVE DEMO MODE:${NC}"
        echo "• Simulates realistic trading performance"
        echo "• Updates every 3 seconds"
        echo "• Perfect for testing GUI interface"
        echo "• No Ray cluster required"
        echo ""
        echo -e "${CYAN}📊 GUI FEATURES:${NC}"
        echo "• Real-time top 20 bot rankings by total capital"
        echo "• Auto-sorting and live updates"
        echo "• Comprehensive performance metrics"
        echo "• Color-coded performance indicators"
        echo "• Training progress monitoring"
        echo ""
        echo -e "${CYAN}🔧 SETUP REQUIREMENTS:${NC}"
        echo "• PyQt6: pip install PyQt6"
        echo "• Ray: pip install ray[default]"
        echo "• PyTorch: pip install torch"
        echo "• Other: pip install pandas numpy psutil GPUtil"
        echo ""
        echo -e "${CYAN}📁 KEY FILES:${NC}"
        echo "• integrated_training_with_gui.py - Main system"
        echo "• ray_kelly_ultimate_75_percent.py - Training backend"
        echo "• kelly_demo_live.py - Live demo simulation"
        echo "• fleet_results.json - Real-time data file"
        echo ""
        echo -e "${CYAN}🎯 RAY CLUSTER SETUP:${NC}"
        echo "PC1 (Head Node):"
        echo "  ray start --head --port=8265"
        echo ""
        echo "PC2 (Worker Node):"
        echo "  ray start --address='PC1_IP:10001'"
        echo ""
        echo "Verify cluster:"
        echo "  ray status"
        echo ""
        ;;
        
    *)
        echo ""
        echo -e "${RED}Invalid choice. Please select 1-6.${NC}"
        ;;
esac

echo ""
echo -e "${CYAN}🎯 System ready! Choose your preferred launch mode above.${NC}"
echo ""
