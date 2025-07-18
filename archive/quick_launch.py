#!/usr/bin/env python3
"""
Quick Cluster Status and Training Launcher
=========================================
This script quickly checks cluster status and launches training
"""

import subprocess
import sys
import os

def quick_check():
    print("🔍 QUICK CLUSTER STATUS CHECK")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("automated_cluster_training.py"):
        print("❌ Please run from TaskmasterForexBots directory")
        return False
        
    # Check current Ray status
    try:
        result = subprocess.run("ray status", shell=True, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Ray cluster is running!")
            print(result.stdout)
            return True
        else:
            print("⚠️  Ray cluster not detected")
            print("Run: python automated_cluster_training.py")
            return False
    except:
        print("❌ Ray command failed")
        return False

def launch_training():
    print("\n🚀 LAUNCHING AUTOMATED CLUSTER TRAINING")
    print("=" * 50)
    
    try:
        # Import and run the automated training
        import automated_cluster_training
        
        # Create manager and run
        manager = automated_cluster_training.AutomatedClusterManager()
        
        print("Select training mode:")
        print("1. Test Scale (5 generations, ~5 minutes)")
        print("2. Full Scale (200 generations, ~3 hours)")
        
        choice = input("Choice (1/2): ").strip()
        
        if choice == "1":
            success = manager.run_automated_training("test")
        elif choice == "2":
            success = manager.run_automated_training("full")
        else:
            print("❌ Invalid choice")
            return False
            
        return success
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

if __name__ == "__main__":
    # Quick status check
    cluster_ready = quick_check()
    
    if cluster_ready:
        print("\n✅ Cluster appears ready!")
        launch = input("Launch training now? (y/N): ").strip().lower()
        if launch == 'y':
            launch_training()
    else:
        print("\n🔧 Setting up cluster...")
        launch_training()
