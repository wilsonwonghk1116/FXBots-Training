#!/usr/bin/env python3
"""
Test Worker Environment Setup
Run this on Worker PC to verify all modules can be imported
"""

import sys
import os
import traceback

def test_basic_imports():
    """Test basic Python module imports"""
    print("🔍 Testing Basic Imports...")
    
    # Test basic modules
    basic_modules = ['os', 'sys', 'time', 'logging', 'subprocess', 'json']
    
    for module in basic_modules:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except ImportError as e:
            print(f"  ❌ {module}: {e}")

def test_scientific_imports():
    """Test scientific computing imports"""
    print("\n🔬 Testing Scientific Computing Imports...")
    
    try:
        import numpy as np
        print(f"  ✅ numpy {np.__version__}")
    except ImportError as e:
        print(f"  ❌ numpy: {e}")
    
    try:
        import pandas as pd
        print(f"  ✅ pandas {pd.__version__}")
    except ImportError as e:
        print(f"  ❌ pandas: {e}")
    
    try:
        import torch
        print(f"  ✅ torch {torch.__version__}")
        print(f"      CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"      CUDA devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"        Device {i}: {torch.cuda.get_device_name(i)}")
    except ImportError as e:
        print(f"  ❌ torch: {e}")

def test_ray_import():
    """Test Ray import and functionality"""
    print("\n⚡ Testing Ray Import...")
    
    try:
        import ray
        print(f"  ✅ ray {ray.__version__}")
        
        # Test Ray init (don't actually connect to avoid conflicts)
        print("  🔍 Ray import successful")
        
    except ImportError as e:
        print(f"  ❌ ray: {e}")

def test_project_imports():
    """Test project-specific imports"""
    print("\n📦 Testing Project Module Imports...")
    
    # Add project paths
    project_paths = [
        "/home/w1/cursor-to-copilot-backup/TaskmasterForexBots",
        "/home/w2/cursor-to-copilot-backup/TaskmasterForexBots"
    ]
    
    for path in project_paths:
        if os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)
            print(f"  📁 Added to path: {path}")
    
    # Test project modules
    project_modules = [
        'synthetic_env',
        'config',
        'trading_bot',
        'bot_population',
        'indicators',
        'predictors',
        'reward',
        'utils',
        'checkpoint_utils',
        'champion_analysis'
    ]
    
    for module in project_modules:
        try:
            imported_module = __import__(module)
            print(f"  ✅ {module}")
            
            # Special checks for key modules
            if module == 'synthetic_env':
                try:
                    from synthetic_env import SyntheticForexEnv
                    print(f"    ✅ SyntheticForexEnv class imported")
                except ImportError as e:
                    print(f"    ❌ SyntheticForexEnv class: {e}")
            
            elif module == 'bot_population':
                try:
                    from bot_population import EvaluationActor
                    print(f"    ✅ EvaluationActor class imported")
                except ImportError as e:
                    print(f"    ❌ EvaluationActor class: {e}")
                    
        except ImportError as e:
            print(f"  ❌ {module}: {e}")
            if module in ['synthetic_env', 'config']:
                print(f"    📋 This is a critical module - check file exists")

def test_data_file():
    """Test if data file is accessible"""
    print("\n📊 Testing Data File Access...")
    
    data_paths = [
        "/home/w1/cursor-to-copilot-backup/TaskmasterForexBots/data/EURUSD_H1.csv",
        "/home/w2/cursor-to-copilot-backup/TaskmasterForexBots/data/EURUSD_H1.csv"
    ]
    
    for path in data_paths:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  ✅ {path} ({size_mb:.1f} MB)")
            
            # Test reading first few lines
            try:
                import pandas as pd
                df = pd.read_csv(path, nrows=5)
                print(f"    📈 Columns: {list(df.columns)}")
                print(f"    📊 Sample shape: {df.shape}")
            except Exception as e:
                print(f"    ❌ Error reading: {e}")
        else:
            print(f"  ❌ {path} (not found)")

def test_environment_variables():
    """Test environment variables"""
    print("\n🌍 Testing Environment Variables...")
    
    # Check conda environment
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'Not set')
    print(f"  🐍 CONDA_DEFAULT_ENV: {conda_env}")
    
    # Check Python path
    pythonpath = os.environ.get('PYTHONPATH', 'Not set')
    print(f"  📁 PYTHONPATH: {pythonpath}")
    
    # Check Python executable
    print(f"  🐍 Python executable: {sys.executable}")
    print(f"  📁 Python sys.path (first 3):")
    for i, path in enumerate(sys.path[:3]):
        print(f"    {i+1}. {path}")

def test_gpu_access():
    """Test GPU access from worker environment"""
    print("\n🎮 Testing GPU Access...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print(f"  ✅ CUDA devices available: {device_count}")
            
            for i in range(device_count):
                device_name = torch.cuda.get_device_name(i)
                memory_total = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"    🎮 Device {i}: {device_name} ({memory_total:.1f} GB)")
                
                # Test basic GPU operation
                try:
                    test_tensor = torch.randn(100, 100).cuda(i)
                    result = torch.matmul(test_tensor, test_tensor)
                    print(f"    ✅ GPU {i} computation test passed")
                    del test_tensor, result
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"    ❌ GPU {i} computation test failed: {e}")
        else:
            print(f"  ❌ No CUDA devices available")
            
    except ImportError as e:
        print(f"  ❌ PyTorch not available: {e}")

def main():
    print("🧪 Worker Environment Test Suite")
    print("=" * 50)
    print(f"Running on: {os.uname().nodename}")
    print(f"Working directory: {os.getcwd()}")
    print(f"User: {os.environ.get('USER', 'unknown')}")
    print("=" * 50)
    
    try:
        test_environment_variables()
        test_basic_imports()
        test_scientific_imports()
        test_ray_import()
        test_project_imports()
        test_data_file()
        test_gpu_access()
        
        print("\n🎯 Test Summary:")
        print("✅ If all critical modules show ✅, the environment is ready")
        print("❌ If any ❌ appear, those need to be fixed before training")
        print("\n📋 Critical modules for training:")
        print("  - synthetic_env (must work)")
        print("  - ray (must work)")  
        print("  - torch with CUDA (must work)")
        print("  - bot_population.EvaluationActor (must work)")
        
    except Exception as e:
        print(f"\n💥 Test suite crashed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 