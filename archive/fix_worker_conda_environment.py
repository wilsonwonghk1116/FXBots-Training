#!/usr/bin/env python3
"""
Worker PC Conda Environment Fixer
This script ensures that Ray workers on the Worker PC use the correct conda environment
"""

import subprocess
import sys
import os

def run_ssh_command(command):
    """Execute command on Worker PC via SSH"""
    ssh_cmd = f'sshpass -p "w" ssh w2@192.168.1.11 "{command}"'
    try:
        result = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def check_worker_conda_setup():
    """Check Worker PC conda environment setup"""
    print("🔍 Checking Worker PC conda environment...")
    
    # Check if conda is accessible
    success, stdout, stderr = run_ssh_command("~/miniconda3/bin/conda env list")
    if success:
        print("✅ Conda environments found on Worker PC:")
        print(stdout)
    else:
        print("❌ Failed to access conda on Worker PC:")
        print(stderr)
        return False
    
    # Check if BotsTraining_env has required packages
    success, stdout, stderr = run_ssh_command(
        "source ~/miniconda3/bin/activate BotsTraining_env && pip list | grep -E '(ray|torch|pandas|numpy)'"
    )
    if success:
        print("✅ Required packages in BotsTraining_env:")
        print(stdout)
    else:
        print("❌ Failed to check packages in BotsTraining_env:")
        print(stderr)
        return False
    
    return True

def setup_ray_worker_environment():
    """Setup Ray worker environment configuration on Worker PC"""
    print("🛠️ Setting up Ray worker environment configuration...")
    
    # Create a startup script for Ray workers
    startup_script = '''#!/bin/bash
# Ray Worker Startup Script for Worker PC
# Ensures workers use correct conda environment

export PATH="/home/w2/miniconda3/bin:$PATH"
source /home/w2/miniconda3/bin/activate BotsTraining_env

# Set Python path for our project
export PYTHONPATH="/home/w2/cursor-to-copilot-backup/TaskmasterForexBots:$PYTHONPATH"

# Change to project directory
cd /home/w2/cursor-to-copilot-backup/TaskmasterForexBots

# Execute the original command
exec "$@"
'''
    
    # Create the startup script on Worker PC
    script_path = "/home/w2/ray_worker_startup.sh"
    success, stdout, stderr = run_ssh_command(f'cat > {script_path} << "EOF"\n{startup_script}EOF')
    if success:
        print(f"✅ Created startup script: {script_path}")
    else:
        print(f"❌ Failed to create startup script: {stderr}")
        return False
    
    # Make it executable
    success, stdout, stderr = run_ssh_command(f"chmod +x {script_path}")
    if success:
        print("✅ Made startup script executable")
    else:
        print(f"❌ Failed to make script executable: {stderr}")
        return False
    
    return True

def stop_worker_ray_processes():
    """Stop any existing Ray processes on Worker PC"""
    print("🛑 Stopping existing Ray processes on Worker PC...")
    
    success, stdout, stderr = run_ssh_command("pkill -f ray")
    if success or "no process found" in stderr.lower():
        print("✅ Ray processes stopped on Worker PC")
    else:
        print(f"⚠️ Warning: Could not stop Ray processes: {stderr}")
    
    return True

def restart_worker_with_environment():
    """Restart Ray worker with proper environment"""
    print("🚀 Starting Ray worker with proper conda environment...")
    
    # Start Ray worker with custom Python executable
    ray_start_cmd = (
        "source ~/miniconda3/bin/activate BotsTraining_env && "
        "cd /home/w2/cursor-to-copilot-backup/TaskmasterForexBots && "
        "ray start --address='192.168.1.10:6379' --runtime-env='{\"env_vars\": {\"PYTHONPATH\": \"/home/w2/cursor-to-copilot-backup/TaskmasterForexBots\"}}'"
    )
    
    success, stdout, stderr = run_ssh_command(ray_start_cmd)
    if success:
        print("✅ Ray worker started successfully with conda environment")
        print(stdout)
    else:
        print(f"❌ Failed to start Ray worker: {stderr}")
        return False
    
    return True

def main():
    """Main function to fix Worker PC conda environment"""
    print("🎯 === Worker PC Conda Environment Fixer ===")
    print("This script will ensure Ray workers use the correct conda environment\n")
    
    # Step 1: Check current setup
    if not check_worker_conda_setup():
        print("❌ Worker PC conda setup check failed")
        return False
    
    # Step 2: Stop existing Ray processes
    stop_worker_ray_processes()
    
    # Step 3: Setup environment configuration
    if not setup_ray_worker_environment():
        print("❌ Failed to setup Ray worker environment")
        return False
    
    # Step 4: Restart Ray worker with proper environment
    if not restart_worker_with_environment():
        print("❌ Failed to restart Ray worker")
        return False
    
    print("\n🎉 === WORKER PC CONDA ENVIRONMENT FIXED ===")
    print("✅ Ray workers on Worker PC should now use BotsTraining_env conda environment")
    print("✅ Python path configured for project modules")
    print("✅ Ready for distributed training!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 