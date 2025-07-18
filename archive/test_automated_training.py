#!/usr/bin/env python3
"""
Test script for automated cluster training system
"""

import subprocess
import sys

def test_automated_system():
    """Test the automated system with predefined inputs"""
    
    print("🧪 TESTING AUTOMATED CLUSTER TRAINING SYSTEM")
    print("=" * 50)
    
    # Test with option 2 (test scale) and confirm
    test_input = "2\ny\n"
    
    try:
        # Run the automated training with test inputs
        process = subprocess.Popen(
            ["python", "automated_cluster_training.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd="/home/w1/cursor-to-copilot-backup/TaskmasterForexBots"
        )
        
        # Send inputs
        stdout, _ = process.communicate(input=test_input, timeout=60)
        
        print("📋 OUTPUT:")
        print(stdout)
        
        if process.returncode == 0:
            print("✅ Test completed successfully!")
        else:
            print(f"❌ Test failed with return code: {process.returncode}")
            
    except subprocess.TimeoutExpired:
        print("⏰ Test timed out - this is expected for longer operations")
        process.kill()
    except Exception as e:
        print(f"❌ Test error: {e}")

if __name__ == "__main__":
    test_automated_system()
