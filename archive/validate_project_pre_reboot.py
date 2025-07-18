#!/usr/bin/env python3
"""
Project Validation Script - Pre-Reboot Final Check
Validates all components are ready for 75% utilization testing
"""

import os
import sys
from pathlib import Path
import subprocess
import json
from datetime import datetime

def check_file_exists(filepath, description):
    """Check if a file exists and is readable"""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        print(f"✅ {description}: {filepath} ({size:,} bytes)")
        return True
    else:
        print(f"❌ {description}: {filepath} - NOT FOUND")
        return False

def check_executable(filepath, description):
    """Check if a file is executable"""
    if os.path.exists(filepath) and os.access(filepath, os.X_OK):
        print(f"✅ {description}: {filepath} - EXECUTABLE")
        return True
    else:
        print(f"❌ {description}: {filepath} - NOT EXECUTABLE")
        return False

def check_python_imports():
    """Check if all required Python packages can be imported"""
    required_packages = [
        'ray', 'torch', 'numpy', 'pandas', 'psutil'
    ]
    
    optional_packages = [
        'GPUtil', 'numba', 'matplotlib'
    ]
    
    print("\n🐍 Python Package Validation:")
    all_required_ok = True
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}: Available")
        except ImportError:
            print(f"❌ {package}: MISSING (REQUIRED)")
            all_required_ok = False
    
    for package in optional_packages:
        try:
            __import__(package)
            print(f"✅ {package}: Available")
        except ImportError:
            print(f"⚠️  {package}: Missing (optional)")
    
    return all_required_ok

def validate_ray_functionality():
    """Test basic Ray functionality"""
    print("\n🌟 Ray Functionality Test:")
    try:
        import ray
        print(f"✅ Ray version: {ray.__version__}")
        
        # Test basic Ray operations
        if ray.is_initialized():
            ray.shutdown()
        
        ray.init(local_mode=True, ignore_reinit_error=True)
        
        @ray.remote
        def test_function(x):
            return x * 2
        
        result = ray.get(test_function.remote(21))
        if result == 42:
            print("✅ Ray remote functions: Working")
        else:
            print("❌ Ray remote functions: Failed")
            return False
        
        ray.shutdown()
        print("✅ Ray initialization/shutdown: Working")
        return True
        
    except Exception as e:
        print(f"❌ Ray test failed: {e}")
        return False

def check_gpu_availability():
    """Check GPU availability and CUDA"""
    print("\n🎮 GPU Validation:")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"✅ CUDA available: {gpu_count} GPU(s)")
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"   • GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
            
            return True
        else:
            print("⚠️  CUDA not available - will use CPU only")
            return False
    except Exception as e:
        print(f"❌ GPU check failed: {e}")
        return False

def validate_project_structure():
    """Validate the complete project structure"""
    print("📁 Project Structure Validation:")
    
    critical_files = [
        ("kelly_monte_bot.py", "Main bot implementation"),
        ("ray_kelly_ultimate_75_percent.py", "Ultimate performance script"),
        ("setup_ray_cluster_75_percent.sh", "Cluster setup script"),
        ("ray_cluster_monitor_75_percent.py", "Resource monitoring script"),
        ("post_reboot_quick_start.sh", "Quick start script"),
        ("PROJECT_STATE_BACKUP_REBOOT.md", "Project backup documentation"),
        ("RAY_75_PERCENT_UTILIZATION_GUIDE.md", "Setup guide"),
        ("requirements.txt", "Dependencies list")
    ]
    
    all_files_ok = True
    for filepath, description in critical_files:
        if not check_file_exists(filepath, description):
            all_files_ok = False
    
    return all_files_ok

def validate_executable_permissions():
    """Check executable permissions on scripts"""
    print("\n🔧 Executable Permissions:")
    
    executable_files = [
        ("setup_ray_cluster_75_percent.sh", "Cluster setup script"),
        ("ray_cluster_monitor_75_percent.py", "Monitor script"),
        ("post_reboot_quick_start.sh", "Quick start script")
    ]
    
    all_exec_ok = True
    for filepath, description in executable_files:
        if not check_executable(filepath, description):
            all_exec_ok = False
    
    return all_exec_ok

def generate_validation_report():
    """Generate comprehensive validation report"""
    print("\n" + "="*80)
    print("📋 PROJECT VALIDATION REPORT")
    print("="*80)
    
    # Run all validations
    structure_ok = validate_project_structure()
    permissions_ok = validate_executable_permissions()
    python_ok = check_python_imports()
    ray_ok = validate_ray_functionality()
    gpu_ok = check_gpu_availability()
    
    # System info
    print(f"\n🖥️  System Information:")
    try:
        print(f"   • Python: {sys.version.split()[0]}")
        print(f"   • Platform: {sys.platform}")
        print(f"   • CPU cores: {os.cpu_count()}")
        
        # Check conda environment
        conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'None')
        print(f"   • Conda env: {conda_env}")
        
    except Exception as e:
        print(f"   • System info error: {e}")
    
    # Overall assessment
    print(f"\n🎯 VALIDATION SUMMARY:")
    print(f"   • Project structure: {'✅ PASS' if structure_ok else '❌ FAIL'}")
    print(f"   • File permissions: {'✅ PASS' if permissions_ok else '❌ FAIL'}")
    print(f"   • Python packages: {'✅ PASS' if python_ok else '❌ FAIL'}")
    print(f"   • Ray functionality: {'✅ PASS' if ray_ok else '❌ FAIL'}")
    print(f"   • GPU availability: {'✅ PASS' if gpu_ok else '⚠️  CPU ONLY'}")
    
    # Overall status
    critical_ok = structure_ok and permissions_ok and python_ok and ray_ok
    
    if critical_ok:
        print(f"\n🎉 PROJECT STATUS: ✅ READY FOR 75% UTILIZATION TESTING")
        print(f"   • All critical components validated")
        print(f"   • Ready for post-reboot execution")
        print(f"   • Expected time to results: 7 minutes")
    else:
        print(f"\n⚠️  PROJECT STATUS: ❌ ISSUES DETECTED")
        print(f"   • Please resolve the issues above before reboot")
        print(f"   • Run this script again after fixes")
    
    # Save validation report
    report = {
        'validation_timestamp': datetime.now().isoformat(),
        'project_structure': structure_ok,
        'file_permissions': permissions_ok,
        'python_packages': python_ok,
        'ray_functionality': ray_ok,
        'gpu_availability': gpu_ok,
        'overall_status': 'READY' if critical_ok else 'ISSUES_DETECTED',
        'python_version': sys.version.split()[0],
        'conda_environment': os.environ.get('CONDA_DEFAULT_ENV', 'None')
    }
    
    with open('pre_reboot_validation_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Validation report saved: pre_reboot_validation_report.json")
    
    return critical_ok

def main():
    """Main validation function"""
    print("🔍 PRE-REBOOT PROJECT VALIDATION")
    print("Kelly Monte Carlo Trading Bot - 75% Utilization System")
    print("="*80)
    
    # Change to project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    print(f"📁 Working directory: {os.getcwd()}")
    
    # Run comprehensive validation
    validation_passed = generate_validation_report()
    
    if validation_passed:
        print(f"\n🚀 POST-REBOOT COMMAND:")
        print(f"   ./post_reboot_quick_start.sh")
        print(f"\n🎯 EXPECTED RESULTS:")
        print(f"   • 75% CPU utilization across all cores")
        print(f"   • 75% GPU utilization on all GPUs")
        print(f"   • 75% vRAM usage on all GPUs")
        print(f"   • 10,000+ Monte Carlo scenarios/second")
        return 0
    else:
        print(f"\n🔧 REQUIRED ACTIONS:")
        print(f"   1. Fix the issues listed above")
        print(f"   2. Run this validation script again")
        print(f"   3. Proceed with reboot only after all issues resolved")
        return 1

if __name__ == "__main__":
    sys.exit(main())
