#!/usr/bin/env python3
"""
RESTORATION VERIFICATION SCRIPT
Run this after OS reinstall to verify the system is working correctly
"""

import sys
import os
import traceback

def check_dependencies():
    """Check all required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'torch', 'ray', 'gymnasium', 'talib', 
        'pandas', 'numpy', 'psutil'
    ]
    
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n🚨 Install missing packages: pip install {' '.join(missing)}")
        return False
    
    print("✅ All dependencies available")
    return True

def check_gpu():
    """Check GPU availability"""
    print("\n🔍 Checking GPU...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"  ✅ CUDA available: {gpu_count} GPU(s)")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                print(f"    GPU {i}: {gpu_name}")
            return True
        else:
            print("  ❌ CUDA not available")
            return False
    except Exception as e:
        print(f"  ❌ GPU check failed: {e}")
        return False

def check_ray():
    """Check Ray cluster"""
    print("\n🔍 Checking Ray...")
    
    try:
        import ray
        
        # Try to initialize Ray
        if not ray.is_initialized():
            ray.init()
        
        resources = ray.cluster_resources()
        cpu_count = int(resources.get('CPU', 0))
        gpu_count = int(resources.get('GPU', 0))
        
        print(f"  ✅ Ray cluster: {cpu_count} CPUs, {gpu_count} GPUs")
        
        if ray.is_initialized():
            ray.shutdown()
        
        return True
        
    except Exception as e:
        print(f"  ❌ Ray check failed: {e}")
        return False

def check_data_files():
    """Check forex data files"""
    print("\n🔍 Checking data files...")
    
    data_dir = "data"
    if not os.path.exists(data_dir):
        print(f"  ❌ Data directory '{data_dir}' not found")
        return False
    
    expected_files = [
        "EURUSD_H1.csv", "EURUSD_H4.csv", "EURUSD_D1.csv"
    ]
    
    found_files = []
    for file in expected_files:
        file_path = os.path.join(data_dir, file)
        if os.path.exists(file_path):
            print(f"  ✅ {file}")
            found_files.append(file)
        else:
            print(f"  ❌ {file} - Missing")
    
    if found_files:
        print(f"✅ Data files available: {len(found_files)}")
        return True
    else:
        print("❌ No data files found")
        return False

def test_indicator_system():
    """Test the core indicator system"""
    print("\n🔍 Testing indicator system...")
    
    try:
        # Test importing main components
        from run_production_forex_trainer import (
            ComprehensiveTechnicalIndicators,
            ProductionForexEnvironment,
            ProductionTradingBot
        )
        print("  ✅ Main components imported")
        
        # Test environment creation
        env = ProductionForexEnvironment()
        print(f"  ✅ Environment created: {env.observation_space.shape[0]} features")
        
        # Test indicator calculation
        indicators = env.indicators
        print(f"  ✅ Indicators calculated: {len(indicators)} indicators")
        
        # Test bot creation
        bot = ProductionTradingBot(
            input_size=env.observation_space.shape[0],
            strategy_type="test_bot"
        )
        print("  ✅ Bot created successfully")
        
        # Test observation
        obs = env._get_observation()
        print(f"  ✅ Observation generated: shape {obs.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Indicator system test failed: {e}")
        traceback.print_exc()
        return False

def test_indicator_mapping():
    """Test if indicator mapping fix is implemented"""
    print("\n🔍 Testing indicator mapping...")
    
    try:
        from run_production_forex_trainer import ProductionForexEnvironment
        
        env = ProductionForexEnvironment()
        
        # Check if indicator mapping exists
        if hasattr(env, 'indicator_mapping'):
            mapping_count = len(env.indicator_mapping)
            print(f"  ✅ Indicator mapping exists: {mapping_count} mappings")
            
            # Show first few mappings
            for i, (name, idx) in enumerate(list(env.indicator_mapping.items())[:5]):
                print(f"    {name} -> position {idx}")
            
            return True
        else:
            print("  ❌ Indicator mapping NOT IMPLEMENTED yet")
            print("    This is expected if you haven't applied the fixes yet")
            return False
            
    except Exception as e:
        print(f"  ❌ Indicator mapping test failed: {e}")
        return False

def main():
    """Run all verification checks"""
    print("🚀 PROJECT RESTORATION VERIFICATION")
    print("=" * 50)
    
    checks = [
        ("Dependencies", check_dependencies),
        ("GPU", check_gpu),
        ("Ray Cluster", check_ray),
        ("Data Files", check_data_files),
        ("Indicator System", test_indicator_system),
        ("Indicator Mapping", test_indicator_mapping),
    ]
    
    results = {}
    
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"  ❌ {name} check crashed: {e}")
            results[name] = False
    
    print("\n" + "=" * 50)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(checks)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nOVERALL: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 SYSTEM FULLY RESTORED AND READY!")
    elif passed >= total - 1:  # Allow for indicator mapping to be missing
        print("🔧 SYSTEM MOSTLY READY - implement the 4 critical fixes")
    else:
        print("🚨 SYSTEM NEEDS ATTENTION - check failed components")
    
    print("\n📋 NEXT STEPS:")
    if not results.get("Dependencies", False):
        print("1. Install missing dependencies: pip install -r requirements.txt")
    if not results.get("Data Files", False):
        print("2. Restore data files to data/ directory")
    if not results.get("Indicator Mapping", False):
        print("3. Implement the 4 critical fixes from PROJECT_STATE_BACKUP.md")
    if results.get("Indicator System", False):
        print("4. Ready to run: python run_production_forex_trainer.py")

if __name__ == "__main__":
    main() 