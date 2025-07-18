#!/usr/bin/env python3
"""
QUICK TEST: Massive Scale Training
==================================
Simple test to verify the system works before full deployment
"""

import ray
import time
from massive_scale_config import MassiveScaleConfig, MassiveScaleEpisodeProcessor

def quick_test():
    """Run a quick test of the massive scale system"""
    
    print("🧪 QUICK TEST: Massive Scale Training System")
    print("=" * 50)
    
    # Connect to Ray
    if not ray.is_initialized():
        try:
            ray.init(address='ray://192.168.1.10:10001')
            print("✅ Connected to Ray cluster")
        except Exception as e:
            print(f"❌ Ray connection failed: {e}")
            return
    
    # Test configuration
    config = MassiveScaleConfig()
    config.TOTAL_GENERATIONS = 1  # Just 1 generation for test
    config.EPISODES_PER_GENERATION = 5  # Just 5 episodes
    config.STEPS_PER_EPISODE = 100  # Just 100 steps
    
    print(f"\n🔬 TEST CONFIGURATION:")
    print(f"   Generations: {config.TOTAL_GENERATIONS}")
    print(f"   Episodes: {config.EPISODES_PER_GENERATION}")
    print(f"   Steps: {config.STEPS_PER_EPISODE}")
    
    # Test episode processor
    processor = MassiveScaleEpisodeProcessor("TEST_BOT")
    
    print(f"\n🚀 Running test episodes...")
    start_time = time.time()
    
    results = []
    for episode in range(config.EPISODES_PER_GENERATION):
        result = processor.run_episode(episode, 0)
        results.append(result)
        print(f"   Episode {episode}: PnL=${result['total_pnl_usd']:.2f}, Reward={result['total_reward']:.2f}")
    
    test_duration = time.time() - start_time
    
    # Summary
    total_steps = sum(r['steps_completed'] for r in results)
    total_pnl = sum(r['total_pnl_usd'] for r in results)
    total_rewards = sum(r['total_reward'] for r in results)
    
    print(f"\n📊 TEST RESULTS:")
    print(f"   Total Episodes: {len(results)}")
    print(f"   Total Steps: {total_steps:,}")
    print(f"   Total PnL: ${total_pnl:.2f}")
    print(f"   Total Rewards: {total_rewards:.2f}")
    print(f"   Test Duration: {test_duration:.2f}s")
    print(f"   Steps per Second: {total_steps/test_duration:.0f}")
    
    # Estimate full scale
    full_scale_steps = MassiveScaleConfig.get_total_steps()
    estimated_hours = (full_scale_steps / total_steps) * (test_duration / 3600)
    
    print(f"\n🎯 FULL SCALE ESTIMATION:")
    print(f"   Full Scale Steps: {full_scale_steps:,}")
    print(f"   Estimated Time: {estimated_hours:.1f} hours")
    print(f"   Estimated Time: {estimated_hours/24:.1f} days")
    
    if estimated_hours < 10:
        print("✅ System performance looks good for full scale training!")
    else:
        print("⚠️  Full scale training will take significant time")
    
    return results

if __name__ == "__main__":
    results = quick_test()
    
    if results:
        print(f"\n🎉 Test completed successfully!")
        print(f"The massive scale training system is ready to use.")
        
        response = input(f"\n▶️  Run full scale training now? (y/N): ")
        if response.lower() == 'y':
            print("🚀 Starting full scale training...")
            import subprocess
            subprocess.run(['python', 'massive_scale_distributed_training.py'])
        else:
            print("💡 Use 'python launch_massive_training.py' when ready to start")
    else:
        print("❌ Test failed - check system configuration")
