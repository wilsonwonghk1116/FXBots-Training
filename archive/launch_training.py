#!/usr/bin/env python3
"""
LAUNCHER FOR COMPLETE AUTOMATED TRAINING SYSTEM
==============================================

Simple launcher script that handles all dependencies and starts the training.

Usage:
    python launch_training.py

Author: AI Assistant  
Date: July 13, 2025
"""

import sys
import os

def main():
    """Launch the complete automated training system"""
    print("🚀 LAUNCHING COMPLETE AUTOMATED TRAINING SYSTEM")
    print("=" * 60)
    
    try:
        # Import and run the complete system
        from complete_automated_training_system import CompleteAutomatedTrainingSystem
        
        print("✅ All imports successful!")
        print("🎯 Initializing automated training system...")
        
        # Create and run the system
        system = CompleteAutomatedTrainingSystem()
        success = system.run_complete_system()
        
        if success:
            print("\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
            print("📊 Check the generated files for results:")
            print("   - Champion analysis: CHAMPION_ANALYSIS_*.json")
            print("   - Champion models: CHAMPION_BOT_*.pth")
            print("   - Training progress: training_progress_gen_*.json")
        else:
            print("\n❌ TRAINING FAILED!")
            print("📋 Check the logs above for error details")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Try installing missing packages:")
        print("   conda install matplotlib seaborn tensorflow scikit-learn")
        print("   or")
        print("   pip install matplotlib seaborn tensorflow scikit-learn")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
