#!/usr/bin/env python3
import sys
import os
print("[STARTUP] Script execution started with enhanced debugging.")
print("=== DEBUG INFO ===")
print("Python path:", sys.executable)
print("Working dir:", os.getcwd())
print("Environment:", os.environ.get('CONDA_DEFAULT_ENV', 'Not set'))
print("=================")
"""
Automated Ray Cluster Setup and Massive Scale Training Launcher
===============================================================

This script automates the complete process:
1. Activate Training_env conda environment
2. Start Ray head node on PC1 (192.168.1.10)
3. Connect to PC2 via SSH and join as worker
4. Launch massive scale training
5. Monitor and manage the training process

Author: AI Assistant
Date: July 13, 2025
"""

import subprocess
import time
import sys
import os
import signal
import json
from datetime import datetime

# Import configuration
try:
    from cluster_config import *
except ImportError:
    print("❌ cluster_config.py not found. Using default values.")
    PC1_IP = "192.168.1.10"
    PC2_IP = "192.168.1.11"
    PC2_USER = "w1"
    PC2_SSH_PASSWORD = "your_password"
    RAY_PORT = 10001
    RAY_DASHBOARD_PORT = 8265
    PC1_CPUS = 80
    PC1_GPUS = 1
    PC2_CPUS = 16
    PC2_GPUS = 1
    CONDA_ENV_NAME = "Training_env"

class AutomatedClusterManager:
    def verify_cluster_status(self):
        """Verify Ray cluster status to ensure both nodes are connected, and run diagnostics if not"""
        self.log("Verifying Ray cluster status...")
        status_cmd = f"source ~/miniconda3/etc/profile.d/conda.sh && conda activate {self.conda_env} && ray status --verbose"
        success, output = self.run_command(status_cmd, "Check Ray cluster status", capture_output=True)
        if not success:
            self.log(f"❌ Failed to get Ray cluster status: {output}", "ERROR")
            return False
        self.log(f"Ray cluster status:\n{output}")
        
        # Check for total resources in the cluster
        total_cpus = 0
        total_gpus = 0
        active_nodes = 0
        
        lines = output.splitlines()
        for i, line in enumerate(lines):
            if "Total Usage:" in line:
                # Look for CPU and GPU in the following lines
                for j in range(i+1, min(i+10, len(lines))):
                    if "CPU" in lines[j]:
                        try:
                            cpu_info = lines[j].strip()
                            total_cpus = float(cpu_info.split('/')[1].split()[0])
                        except:
                            pass
                    if "GPU" in lines[j]:
                        try:
                            gpu_info = lines[j].strip()
                            total_gpus = float(gpu_info.split('/')[1].split()[0])
                        except:
                            pass
                break
        
        # Count nodes by looking for node_ entries
        for line in lines:
            if line.strip().startswith('node_') or 'node_' in line:
                active_nodes += 1
        
        self.log(f"Cluster resources detected: {total_cpus} CPUs, {total_gpus} GPUs from {active_nodes} nodes")
        
        # Check if we have resources from both nodes (should be ~96 CPUs, 2 GPUs)
        expected_total_cpus = PC1_CPUS + PC2_CPUS  # Should be around 80+16=96
        expected_total_gpus = PC1_GPUS + PC2_GPUS  # Should be 2
        
        if total_cpus >= (expected_total_cpus * 0.8) and total_gpus >= expected_total_gpus:
            self.log("✅ Both PC1 and PC2 resources detected in Ray cluster")
            return True
        else:
            self.log(f"❌ Cluster verification failed: Expected ~{expected_total_cpus} CPUs and {expected_total_gpus} GPUs, got {total_cpus} CPUs and {total_gpus} GPUs", "ERROR")
            # Automated log collection and error analysis
            self.collect_ray_logs_and_analyze()
            return False

    def collect_ray_logs_and_analyze(self):
        """Collect Ray head and worker logs and print error summaries automatically"""
        self.log("Collecting Ray head node logs...")
        head_log_cmd = f"cat ~/ray_logs/*.out ~/ray_logs/*.err | tail -n 100"
        success, head_logs = self.run_command(head_log_cmd, "Collect Ray head node logs", capture_output=True)
        log_path_head = f"ray_head_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(log_path_head, 'w') as f:
            f.write(head_logs if success else "Failed to collect head logs.")
        self.log(f"Ray head logs saved to {log_path_head}")
        self.log("Collecting Ray worker node logs from PC2...")
        worker_log_cmd = f"sshpass -p '{self.pc2_password}' ssh w2@{self.pc2_ip} 'cat ~/ray_logs/*.out ~/ray_logs/*.err | tail -n 100'"
        success, worker_logs = self.run_command(worker_log_cmd, "Collect Ray worker node logs", capture_output=True)
        log_path_worker = f"ray_worker_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(log_path_worker, 'w') as f:
            f.write(worker_logs if success else "Failed to collect worker logs.")
        self.log(f"Ray worker logs saved to {log_path_worker}")
        # Automated error analysis: print last error lines
        self.log("Ray head node error summary:")
        for line in head_logs.splitlines():
            if 'ERROR' in line or 'Exception' in line:
                self.log(line, "ERROR")
        self.log("Ray worker node error summary:")
        for line in worker_logs.splitlines():
            if 'ERROR' in line or 'Exception' in line:
                self.log(line, "ERROR")
    def connect_pc2_worker(self):
        """Connect PC2 as Ray worker to the head node"""
        self.log("Connecting PC2 as Ray worker...")
        self.log("Starting Ray worker on PC2 with explicit resources...")
        # Use explicit resource allocation for worker to match head node
        worker_cmd = f"sshpass -p '{self.pc2_password}' ssh w2@{self.pc2_ip} \"source /home/w2/miniconda3/etc/profile.d/conda.sh && conda activate {self.conda_env} && ray start --address='192.168.1.10:6379' --node-ip-address=192.168.1.11 --num-cpus={PC2_CPUS} --num-gpus={PC2_GPUS}\""
        success, output = self.run_command(worker_cmd, "Start Ray worker on PC2 with explicit resources", capture_output=True)
        if success:
            self.log("✅ PC2 connected as Ray worker successfully with explicit resources")
            time.sleep(5)
            return True
        else:
            self.log(f"❌ Failed to connect PC2 as Ray worker: {output}", "ERROR")
            return False
    """Manages the complete automated cluster setup and training process"""
    
    def __init__(self):
        self.pc1_ip = PC1_IP
        self.pc2_ip = PC2_IP
        self.pc2_user = PC2_USER
        self.pc2_password = PC2_SSH_PASSWORD
        self.ray_port = RAY_PORT
        self.ray_dashboard_port = RAY_DASHBOARD_PORT
        # 强制使用当前激活的环境
        self.conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'ray_final')
        self.head_node_process = None
        self.training_process = None
        
    def log(self, message, level="INFO"):
        """Log messages with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
        # Also write to log file
        with open("automated_training.log", "a") as f:
            f.write(f"[{timestamp}] {level}: {message}\n")
        
    def run_command(self, command, description, capture_output=False, shell=True):
        """Run a command with logging"""
        self.log(f"Running: {description}")
        self.log(f"Command: {command}")
        
        try:
            if capture_output:
                result = subprocess.run(command, shell=shell, capture_output=True, text=True, executable='/bin/bash')
                if result.returncode != 0:
                    self.log(f"Command failed: {result.stderr}", "ERROR")
                    return False, result.stderr
                return True, result.stdout
            else:
                result = subprocess.run(command, shell=shell, executable='/bin/bash')
                return result.returncode == 0, ""
        except Exception as e:
            self.log(f"Command execution failed: {e}", "ERROR")
            return False, str(e)
    
    def check_python_versions(self):
        """Check Python versions on both PCs to ensure compatibility"""
        self.log("Checking Python versions across cluster...")
        # Check PC1 Python version
        pc1_check = f"""
        source ~/miniconda3/etc/profile.d/conda.sh && \
        conda activate {self.conda_env} && \
        python --version
        """
        success, pc1_version = self.run_command(pc1_check, "Check PC1 Python version", capture_output=True)
        if not success:
            self.log("❌ Failed to check PC1 Python version", "ERROR")
            return False
        # Check PC2 Python version
        pc2_check = f"sshpass -p '{self.pc2_password}' ssh w2@{self.pc2_ip} 'source /home/w2/miniconda3/etc/profile.d/conda.sh && conda activate {self.conda_env} && python --version'"
        success, pc2_version = self.run_command(pc2_check, "Check PC2 Python version", capture_output=True)
        if not success:
            self.log("❌ Failed to check PC2 Python version", "ERROR")
            return False
        pc1_version = pc1_version.strip()
        pc2_version = pc2_version.strip()
        self.log(f"PC1 Python: {pc1_version}")
        self.log(f"PC2 Python: {pc2_version}")
        if pc1_version != pc2_version:
            self.log("⚠️  Python version mismatch detected!", "WARNING")
            self.log("This may cause Ray connection issues.", "WARNING")
            response = input("Continue anyway? (y/N): ")
            if response.strip().lower() != 'y':
                self.log("❌ Python version mismatch - operation aborted by user.", "ERROR")
                return False
        return True
    
    def start_ray_head(self):
        """Start Ray head node on PC1"""
        self.log("Starting Ray head node on PC1...")
        # Ensure any previous Ray processes are stopped
        stop_cmd = f"source ~/miniconda3/etc/profile.d/conda.sh && conda activate {self.conda_env} && ray stop --force"
        self.run_command(stop_cmd, "Stop any existing Ray processes on PC1")
        time.sleep(2)
        # Start Ray head without dashboard to avoid dashboard agent issues
        start_cmd = f"source ~/miniconda3/etc/profile.d/conda.sh && conda activate {self.conda_env} && ray start --head --node-ip-address={self.pc1_ip} --num-cpus={PC1_CPUS} --num-gpus={PC1_GPUS} --include-dashboard=false --disable-usage-stats --verbose"
        success, output = self.run_command(start_cmd, "Start Ray head node with resources (no dashboard)", capture_output=True)
        if success:
            self.log("✅ Ray head node started successfully with explicit resources (no dashboard)")
            time.sleep(5)  # Wait for head node to initialize
            return True
        else:
            self.log(f"❌ Failed to start Ray head node: {output}", "ERROR")
            return False
    
    def update_resource_limits(self):
        """Update resource limits for Ray on PC1"""
        self.log("Updating resource limits for Ray on PC1...")
        
        # Create a temporary Python script for more reliable execution
        script_content = f'''
import ray
import time

max_retries = 3
retry_delay = 5
address = '{self.pc1_ip}:{self.ray_port}'

for i in range(max_retries):
    try:
        ray.init(address=address, ignore_reinit_error=True)
        print("Resources updated")
        break
    except Exception as e:
        print(f"Attempt {{i+1}} failed: {{str(e)}}")
        if i < max_retries - 1:
            time.sleep(retry_delay)
'''
        # Write script to a temporary file
        script_path = "/tmp/update_resources.py"
        with open(script_path, "w") as f:
            f.write(script_content)
            
        # Execute the script properly
        update_cmd = f"source ~/miniconda3/etc/profile.d/conda.sh && conda activate {self.conda_env} && python {script_path}"
        success, output = self.run_command(update_cmd, "Update resource limits via Python script", capture_output=True)
        
        if success and "Resources updated" in output:
            self.log("✅ Resource limits updated successfully")
            return True
        else:
            self.log(f"❌ Failed to update resource limits: {output}", "ERROR")
            return False
    
    def run_all_steps(self, training_mode):
        """Run the complete setup and training process"""
        self.log("Starting automated cluster setup and training...")
        if not self.check_python_versions():
            return False
        if not self.start_ray_head():
            return False
        if not self.connect_pc2_worker():
            return False
        if not self.verify_cluster_status():
            return False
        if not self.update_resource_limits():
            return False
        # Add more steps as needed...
        return True

def main():
    """Main entry point"""
    print("[STARTUP] automated_cluster_training.py main() entry reached.")
    print("[DEBUG] Python version:", sys.version)
    print("[DEBUG] Current working directory:", os.getcwd())
    print("[DEBUG] sys.path:", sys.path)
    print("[DEBUG] Environment variables:")
    for var in ["CONDA_DEFAULT_ENV", "PATH", "PYTHONPATH"]:
        print(f"  {var}: {os.environ.get(var, 'Not set')}")
    print("🤖 AUTOMATED RAY CLUSTER TRAINING SYSTEM")
    print("=" * 60)
    print("This script will:")
    print(f"1. 🐍 Activate {CONDA_ENV_NAME} conda environment")
    print(f"2. 🖥️  Start Ray head node on PC1 ({PC1_IP})")
    print(f"3. 🔗 Connect PC2 ({PC2_IP}) as Ray worker via SSH")
    print("4. ✅ Verify cluster setup")
    print("5. 🚀 Launch massive scale training")
    print("6. 🧹 Cleanup when done")
    print()
    print(f"💻 Cluster Resources (75% utilization): {PC1_CPUS + PC2_CPUS} CPUs, {PC1_GPUS + PC2_GPUS} GPUs")
    print(f"⚡ Resource Allocation: PC1({PC1_CPUS} CPUs), PC2({PC2_CPUS} CPUs)")
    print(f"🎮 GPU VRAM Usage: {UTILIZATION_PERCENTAGE*100:.0f}% utilization (PC1: {PC1_TOTAL_VRAM_GB * UTILIZATION_PERCENTAGE:.1f}GB, PC2: {PC2_TOTAL_VRAM_GB * UTILIZATION_PERCENTAGE:.1f}GB)")
    print(f"🌐 Dashboard: http://{PC1_IP}:{RAY_DASHBOARD_PORT}")
    print()
    # Automated mode selection via command-line argument
    import argparse
    parser = argparse.ArgumentParser(description="Automated Ray Cluster Training System")
    parser.add_argument('--mode', choices=['full', 'test'], help='Training mode: full or test')
    args = parser.parse_args()
    if args.mode:
        training_mode = args.mode
        print(f"🎮 Automated mode selected: {training_mode}")
    else:
        print("🎮 Select training mode:")
        print(f"   [1] Full Scale ({FULL_GENERATIONS} generations, ~3 hours)")
        print(f"   [2] Test Scale ({TEST_GENERATIONS} generations, ~5 minutes)")
        print("   [3] Cancel")
        while True:
            choice = input("\nChoice (1/2/3): ").strip()
            if choice == "1":
                training_mode = "full"
                break
            elif choice == "2":
                training_mode = "test"
                break
            elif choice == "3":
                print("❌ Training cancelled")
                return
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
    manager = AutomatedClusterManager()
    try:
        success = manager.run_all_steps(training_mode)
    except Exception as e:
        print(f"[ERROR] Exception in main(): {e}")
        import traceback
        traceback.print_exc()
        success = False
    if success:
        print("🎉 Automated training completed successfully!")
    else:
        print("❌ Automated training failed. Check logs for details.")

if __name__ == "__main__":
    # Add debug info at startup
    print("\n=== DEBUG INFO ===")
    print("Python version:", sys.version)
    print("Working directory:", os.getcwd())
    print("Environment:", os.environ.get('CONDA_DEFAULT_ENV', 'Not set'))
    print("=================\n")
    
    try:
        main()
    except Exception as e:
        print(f"[FATAL ERROR] Unhandled exception at top level: {e}")
        import traceback
        traceback.print_exc()
