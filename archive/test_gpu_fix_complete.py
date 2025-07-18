#!/usr/bin/env python3
"""
🧪 COMPLETE GPU FIX VERIFICATION
Test all aspects of the distributed GPU fix
"""

import os
import sys
import subprocess
import time

def run_test_sequence():
    """Run complete test sequence"""
    print("🧪 COMPLETE GPU FIX VERIFICATION")
    print("=" * 60)
    
    tests = [
        {
            'name': '🔍 Step 1: Test GPU Detection Fix',
            'command': 'python fix_distributed_gpu_detection.py',
            'expected': 'ALL WORKERS SUCCESSFUL',
            'timeout': 60
        },
        {
            'name': '🚀 Step 2: Test Fixed Distributed Trainer',
            'command': 'python fixed_distributed_trainer.py',
            'expected': 'Training successful',
            'timeout': 120
        },
        {
            'name': '📊 Step 3: Test Original Trainer (Should show no VRAM errors)',
            'command': 'python test_progress_monitoring.py',
            'expected': 'ALL TESTS PASSED',
            'timeout': 90,
            'input': 'y\n'  # Auto-answer the training question
        }
    ]
    
    results = []
    
    for i, test in enumerate(tests, 1):
        print(f"\n{test['name']}")
        print("-" * 50)
        
        try:
            # Prepare command
            cmd = test['command'].split()
            
            # Run test with timeout
            if test.get('input'):
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                # Send input
                stdout, stderr = process.communicate(
                    input=test['input'],
                    timeout=test['timeout']
                )
            else:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=test['timeout']
                )
                stdout = result.stdout
                stderr = result.stderr
                process = result
            
            # Check results with more flexible matching
            success_indicators = {
                'ALL WORKERS SUCCESSFUL': ['ALL WORKERS SUCCESSFUL', 'GPU distribution fix successful'],
                'Training successful': ['Training successful', 'training completed successfully'],  
                'ALL TESTS PASSED': ['ALL TESTS PASSED', 'ALL TESTS PASSED!']
            }
            
            expected = test['expected']
            indicators = success_indicators.get(expected, [expected])
            
            if any(indicator in stdout for indicator in indicators):
                print(f"✅ {test['name']} - PASSED")
                results.append({'test': test['name'], 'status': 'PASSED', 'output': stdout})
            else:
                print(f"❌ {test['name']} - FAILED")
                print(f"Expected any of: {indicators}")
                print(f"Output: {stdout[-200:]}")  # Last 200 chars
                results.append({'test': test['name'], 'status': 'FAILED', 'output': stdout, 'error': stderr})
            
        except subprocess.TimeoutExpired:
            print(f"⏰ {test['name']} - TIMEOUT")
            results.append({'test': test['name'], 'status': 'TIMEOUT'})
            
        except Exception as e:
            print(f"💥 {test['name']} - ERROR: {e}")
            results.append({'test': test['name'], 'status': 'ERROR', 'error': str(e)})
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for r in results if r['status'] == 'PASSED')
    total = len(results)
    
    for result in results:
        status_emoji = {
            'PASSED': '✅',
            'FAILED': '❌', 
            'TIMEOUT': '⏰',
            'ERROR': '💥'
        }
        emoji = status_emoji.get(result['status'], '❓')
        print(f"{emoji} {result['test']}: {result['status']}")
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎊 ALL TESTS PASSED!")
        print("🔧 GPU distribution issue is FIXED!")
        print("\n📋 What this means:")
        print("✅ No more 'Invalid device id' errors")
        print("✅ Both RTX 3090 and RTX 3070 working properly")
        print("✅ Distributed training fully functional")
        print("\n🚀 You can now run your full training with:")
        print("   python fixed_distributed_trainer.py")
        return True
    else:
        print(f"\n💥 {total-passed} tests failed!")
        print("🔍 Check the output above for details")
        print("\n🛠️  Possible solutions:")
        print("1. Restart Ray cluster: ray stop && ray start --head")
        print("2. Check GPU drivers on both PCs")
        print("3. Verify CUDA installation")
        return False

def quick_ray_check():
    """Quick Ray cluster health check"""
    print("🔍 Quick Ray Cluster Check...")
    
    try:
        import ray
        
        if not ray.is_initialized():
            ray.init(address='auto')
        
        resources = ray.cluster_resources()
        available = ray.available_resources()
        
        print(f"📊 Cluster resources: {resources}")
        print(f"💡 Available resources: {available}")
        
        gpu_count = resources.get('GPU', 0)
        cpu_count = resources.get('CPU', 0)
        
        if gpu_count >= 2 and cpu_count >= 50:
            print("✅ Ray cluster looks healthy!")
            return True
        else:
            print(f"⚠️  Ray cluster may have issues:")
            print(f"   GPUs: {gpu_count} (expected: 2+)")
            print(f"   CPUs: {cpu_count} (expected: 50+)")
            return False
            
    except Exception as e:
        print(f"❌ Ray cluster check failed: {e}")
        return False

if __name__ == "__main__":
    print("🏁 Starting GPU Fix Verification...")
    
    # Check Ray first
    if not quick_ray_check():
        print("\n💥 Ray cluster issues detected!")
        print("🔧 Please fix Ray cluster first, then run this test again")
        sys.exit(1)
    
    # Run main tests
    success = run_test_sequence()
    
    if success:
        print("\n🎉 GPU fix verification completed successfully!")
        print("💪 Your forex trading system is ready for full-scale training!")
    else:
        print("\n💔 GPU fix verification failed")
        print("🔍 Please check the errors above and try again")
        sys.exit(1) 