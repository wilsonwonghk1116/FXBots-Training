#!/bin/bash
# Quick Launch Script for Integrated Training with GUI
# Launches Kelly Monte Carlo Fleet Training with Real-Time Performance Monitor

echo "🚀 Kelly Monte Carlo Fleet Training with GUI"
echo "=============================================="
echo ""
echo "🔧 Checking dependencies..."

# Check Python version
python_version=$(python3 --version 2>&1)
echo "✅ Python: $python_version"

# Check PyQt6
if python3 -c "import PyQt6" 2>/dev/null; then
    echo "✅ PyQt6: Available"
else
    echo "❌ PyQt6: Not found - Installing..."
    pip install PyQt6
fi

# Check Ray
if python3 -c "import ray" 2>/dev/null; then
    echo "✅ Ray: Available"
else
    echo "❌ Ray: Not found - Installing..."
    pip install ray[default]
fi

echo ""
echo "🎯 Launching Integrated Training System..."
echo "   - Real-time GUI dashboard"
echo "   - Top 20 bot performance tracking"
echo "   - Live capital ranking"
echo "   - Training process monitoring"
echo ""

# Launch the integrated system
python3 integrated_training_with_gui.py

echo ""
echo "🏁 Training session completed."
