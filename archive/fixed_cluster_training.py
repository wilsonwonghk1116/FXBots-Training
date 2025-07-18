#!/usr/bin/env python3
"""
Fixed Cluster Training Script
============================
This script fixes the shell and SSH authentication issues.

Usage:
    python fixed_cluster_training.py

Author: AI Assistant
Date: July 13, 2025
"""

import subprocess
import time
import sys
import os
import signal
from datetime import datetime

# Configuration
PC1_IP = "192.168.1.10"
PC2_IP = "192.168.1.11"
PC2_USER = "w1"
PC2_PASSWORD = "w"
RAY_PORT = 10001
RAY_DASHBOARD_PORT = 8265
CONDA_ENV = "Training_env"

class FixedClusterTrainer:
    """Fixed cluster trainer with proper shell and SSH handling"""
    
    def __init__(self):
        self.pc1_ip = PC1_IP
        self.pc2_ip = PC2_IP
        self.pc2_user = PC2_USER
        self.pc2_password = PC2_PASSWORD
        self.ray_port = RAY_PORT
        self.conda_env = CONDA_ENV
        
    def log(self, message, level="INFO"):
        """Log with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
        
    def run_bash_command(self, command, description, capture_output=False, timeout=60):
        """Run command with bash shell"""
        self.log(f"🔧 {description}")
        try:
            if capture_output:
                result = subprocess.run(
                    command, 
                    shell=True, 
                    capture_output=True, 
                    text=True, 
                    timeout=timeout,
                    executable='/bin/bash'
                )
                if result.returncode == 0:
                    return True, result.stdout
                else:
                    self.log(f"❌ Command failed: {result.stderr}", "ERROR")
                    return False, result.stderr
            else:
                result = subprocess.run(
                    command, 
                    shell=True, 
                    timeout=timeout,
                    executable='/bin/bash'
                )
                return result.returncode == 0, ""
        except subprocess.TimeoutExpired:
            self.log(f"⏱️ Timeout: {description}", "WARNING")
            return False, "Timeout"
        except Exception as e:
            self.log(f"❌ Error: {e}", "ERROR")
            return False, str(e)
    
    def test_ssh_connection(self):
        """Test SSH connection to PC2"""
        self.log("🔐 Testing SSH connection to PC2...")
        
        test_cmd = f"sshpass -p '{self.pc2_password}' ssh w2@{self.pc2_ip} 'echo SSH_SUCCESS'"
        
        success, output = self.run_bash_command(test_cmd, "Test SSH to PC2", capture_output=True, timeout=15)
        
        if success and "SSH_SUCCESS" in output:
            self.log("✅ SSH connection to PC2 successful")
            return True
        else:
            self.log(f"❌ SSH connection failed: {output}", "ERROR")
            self.log("💡 Check PC2 password, IP address, and SSH service", "INFO")
            return False
    
    def check_ray_status(self):
        """Check current Ray cluster status"""
        self.log("🔍 Checking current Ray status...")
        
        cmd = f"source ~/miniconda3/etc/profile.d/conda.sh && conda activate {self.conda_env} && ray status"
        success, output = self.run_bash_command(cmd, "Check Ray status", capture_output=True)
        
        if success and ("2 node" in output.lower() or "nodes: 2" in output.lower()):
            self.log("✅ Ray cluster already running with 2 nodes!")
            self.log(f"Cluster info:\n{output}")
            return True
        else:
            self.log("⚠️  No 2-node cluster detected")
            return False
    
    def stop_ray_everywhere(self):
        """Stop Ray on both PCs"""
        self.log("🛑 Stopping Ray on all nodes...")
        
        # Stop on PC1
        pc1_cmd = f"source ~/miniconda3/etc/profile.d/conda.sh && conda activate {self.conda_env} && ray stop --force"
        self.run_bash_command(pc1_cmd, "Stop Ray on PC1")
        
        # Stop on PC2 (if SSH works)
        if self.test_ssh_connection():
            pc2_cmd = f"sshpass -p '{self.pc2_password}' ssh w2@{self.pc2_ip} 'source /home/w2/miniconda3/etc/profile.d/conda.sh && conda activate {self.conda_env} && ray stop --force'"
            self.run_bash_command(pc2_cmd, "Stop Ray on PC2")
        
        time.sleep(3)
    
    def start_ray_head(self):
        """Start Ray head node on PC1"""
        self.log("🖥️  Starting Ray head node...")
        
        cmd = f"""
        source ~/miniconda3/etc/profile.d/conda.sh && \\
        conda activate {self.conda_env} && \\
        ray start --head --node-ip-address={self.pc1_ip} \\
            --dashboard-host=0.0.0.0 \\
            --dashboard-port={RAY_DASHBOARD_PORT} \\
            --num-cpus=80 \\
            --num-gpus=1 \\
            --object-store-memory=19000000000
        """
        
        success, output = self.run_bash_command(cmd, "Start Ray head node", capture_output=True, timeout=120)
        
        if success and "Ray runtime started" in output:
            self.log("✅ Ray head node started successfully")
            self.log(f"🌐 Dashboard: http://{self.pc1_ip}:{RAY_DASHBOARD_PORT}")
            time.sleep(5)
            return True
        else:
            self.log(f"❌ Failed to start Ray head: {output}", "ERROR")
            return False
    
    def connect_pc2_worker(self):
        """Connect PC2 as Ray worker"""
        self.log("🔗 Connecting PC2 as worker...")
        
        if not self.test_ssh_connection():
            return False
        
        cmd = f"sshpass -p '{self.pc2_password}' ssh w2@{self.pc2_ip} \"source /home/w2/miniconda3/etc/profile.d/conda.sh && conda activate {self.conda_env} && ray start --address='{self.pc1_ip}:6379' --node-ip-address={self.pc2_ip}\""
        
        success, output = self.run_bash_command(cmd, "Connect PC2 worker", capture_output=True, timeout=120)
        
        if success and "Ray runtime started" in output:
            self.log("✅ PC2 connected as worker")
            time.sleep(5)
            return True
        else:
            self.log(f"❌ Failed to connect PC2: {output}", "ERROR")
            return False
    
    def verify_cluster(self):
        """Verify cluster is ready"""
        self.log("✅ Verifying cluster...")
        
        cmd = f"""
        source ~/miniconda3/etc/profile.d/conda.sh && \\
        conda activate {self.conda_env} && \\
        python -c "
import ray
try:
    ray.init(address='ray://{self.pc1_ip}:10001')
    resources = ray.cluster_resources()
    total_cpus = resources.get('CPU', 0)
    total_gpus = resources.get('GPU', 0)
    print(f'Resources: {{total_cpus}} CPUs, {{total_gpus}} GPUs')
    
    if total_cpus >= 90 and total_gpus >= 2:
        print('✅ CLUSTER_READY')
    else:
        print(f'❌ INSUFFICIENT: {{total_cpus}}/96 CPUs, {{total_gpus}}/2 GPUs')
    ray.shutdown()
except Exception as e:
    print(f'❌ ERROR: {{e}}')
"
        """
        
        success, output = self.run_bash_command(cmd, "Verify cluster", capture_output=True)
        
        if success and "✅ CLUSTER_READY" in output:
            self.log("✅ Cluster ready for training!")
            self.log(f"📊 {output.strip()}")
            return True
        else:
            self.log(f"❌ Cluster verification failed: {output}", "ERROR")
            return False
    
    def setup_cluster(self):
        """Complete cluster setup"""
        self.log("🚀 Setting up cluster...")
        
        # Test SSH first
        if not self.test_ssh_connection():
            self.log("❌ Cannot proceed without SSH access to PC2", "ERROR")
            return False
        
        # Stop any existing Ray
        self.stop_ray_everywhere()
        
        # Start head node
        if not self.start_ray_head():
            return False
        
        # Connect worker
        if not self.connect_pc2_worker():
            return False
        
        # Verify cluster
        return self.verify_cluster()
    
    def launch_test_training(self):
        """Launch quick test training"""
        self.log("🧪 Launching test training...")
        
        cmd = f"""
        source ~/miniconda3/etc/profile.d/conda.sh && \\
        conda activate {self.conda_env} && \\
        python -c "
import sys
import os
sys.path.append('/home/w1/cursor-to-copilot-backup/TaskmasterForexBots')

print('🧪 TEST MODE: Quick training validation')
print('📡 Connecting to Ray cluster...')

import ray
ray.init(address='ray://192.168.1.10:10001')

print('✅ Connected to cluster')
print('📊 Resources:', ray.cluster_resources())

# Simple distributed test
@ray.remote
def test_task(x):
    import time
    time.sleep(1)
    return x * x

print('🎯 Running distributed test...')
futures = [test_task.remote(i) for i in range(10)]
results = ray.get(futures)
print(f'✅ Test completed: {{results}}')

print('🎉 CLUSTER TEST SUCCESSFUL!')
ray.shutdown()
"
        """
        
        success, output = self.run_bash_command(cmd, "Test training", capture_output=True, timeout=300)
        
        if success and "CLUSTER TEST SUCCESSFUL" in output:
            self.log("✅ Test training completed!")
            return True
        else:
            self.log(f"❌ Test training failed: {output}", "ERROR")
            return False
    
    def cleanup(self):
        """Clean up cluster"""
        self.log("🧹 Cleaning up...")
        self.stop_ray_everywhere()

def main():
    """Main execution"""
    print("🛠️  FIXED CLUSTER TRAINING SCRIPT")
    print("=" * 40)
    print(f"🖥️  PC1 (Head): {PC1_IP}")
    print(f"🔗 PC2 (Worker): {PC2_IP}")
    print(f"🌐 Dashboard: http://{PC1_IP}:{RAY_DASHBOARD_PORT}")
    print("=" * 40)
    
    trainer = FixedClusterTrainer()
    
    # Setup signal handler
    def signal_handler(signum, frame):
        print("\n🛑 Interrupted! Cleaning up...")
        trainer.cleanup()
        sys.exit(1)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Check if cluster already running
        if trainer.check_ray_status():
            print("\n✅ Cluster already running!")
            choice = input("Run test training? (y/N): ").strip().lower()
            if choice == 'y':
                trainer.launch_test_training()
        else:
            # Setup cluster
            if trainer.setup_cluster():
                print("\n🎉 Cluster setup successful!")
                choice = input("Run test training? (y/N): ").strip().lower()
                if choice == 'y':
                    trainer.launch_test_training()
            else:
                print("❌ Cluster setup failed")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    finally:
        trainer.cleanup()

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 SCRIPT COMPLETED!")
    else:
        print("\n❌ Script failed")
