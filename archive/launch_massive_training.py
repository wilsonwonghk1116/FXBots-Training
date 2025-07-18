#!/usr/bin/env python3
"""
SIMPLE LAUNCHER for Massive Scale Training
==========================================
Easy-to-use launcher for the massive scale forex training
"""

import subprocess
import sys
import os

def check_ray_cluster():
    """Check if Ray cluster is active"""
    try:
        result = subprocess.run(['ray', 'status'], capture_output=True, text=True, timeout=10)
        if 'Active:' in result.stdout and 'node_' in result.stdout:
            print("✅ Ray cluster is active")
            return True
        else:
            print("❌ Ray cluster not found")
            return False
    except Exception as e:
        print(f"❌ Error checking Ray cluster: {e}")
        return False

def launch_cluster_if_needed():
    """Launch Ray cluster if not active"""
    if not check_ray_cluster():
        print("🚀 Starting Ray cluster...")
        try:
            subprocess.run(['/home/w1/cursor-to-copilot-backup/TaskmasterForexBots/launch_dual_pc_training.sh'], 
                         check=True, timeout=60)
            print("✅ Ray cluster started")
            return True
        except Exception as e:
            print(f"❌ Failed to start cluster: {e}")
            return False
    return True

def main():
    print("🚀 MASSIVE SCALE FOREX TRAINING LAUNCHER")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists('massive_scale_config.py'):
        print("❌ Please run this from the TaskmasterForexBots directory")
        sys.exit(1)
    
    # Check/start Ray cluster
    if not launch_cluster_if_needed():
        print("❌ Cannot proceed without Ray cluster")
        sys.exit(1)
    
    print("\n📋 TRAINING CONFIGURATION:")
    print("   🎯 200 generations")
    print("   📈 1,000 episodes per generation")
    print("   ⚡ 1,000 steps per episode")
    print("   💰 PnL-based reward system")
    print("   🖥️  Dual-PC Ray cluster")
    print("   ⏱️  Estimated time: ~3 hours")
    
    print("\n⚠️  WARNING: This is a massive scale operation!")
    print("   - Will run for several hours")
    print("   - Uses both PC1 and PC2 at full capacity")
    print("   - Generates ~200 million training steps")
    
    # Get user confirmation
    choice = input("\n🎮 Choose training mode:\n   [1] Full Scale (200 generations)\n   [2] Test Scale (2 generations)\n   [3] Cancel\n\nChoice (1/2/3): ")
    
    if choice == '1':
        print("\n🚀 Starting FULL SCALE training...")
        subprocess.run([sys.executable, 'massive_scale_distributed_training.py'], 
                      input='y\n', text=True)
    
    elif choice == '2':
        print("\n🧪 Starting TEST SCALE training...")
        subprocess.run([sys.executable, 'massive_scale_distributed_training.py'], 
                      input='n\n', text=True)
    
    else:
        print("❌ Training cancelled")

if __name__ == "__main__":
    main()
