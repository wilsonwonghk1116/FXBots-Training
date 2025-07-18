#!/usr/bin/env python3
"""
Launch Real Training Program
===========================
Streamlined script to launch the actual forex bot training with full cluster setup.
This script assumes cluster connectivity has been verified.

Usage:
    python launch_real_training.py

Author: AI Assistant  
Date: July 13, 2025
"""

import subprocess
import time
import sys
import os
import signal
from datetime import datetime

# Configuration from cluster_config.py
PC1_IP = "192.168.1.10"
PC2_IP = "192.168.1.11"
PC2_USER = "w1"
PC2_PASSWORD = "w"
RAY_PORT = 10001
RAY_DASHBOARD_PORT = 8265
CONDA_ENV = "Training_env"

class RealTrainingLauncher:
    """Launches the real forex bot training program"""
    
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
        
    def run_command(self, command, description, capture_output=False):
        """Execute command with logging"""
        self.log(f"🔧 {description}")
        try:
            if capture_output:
                result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=60, executable='/bin/bash')
                return result.returncode == 0, result.stdout if result.returncode == 0 else result.stderr
            else:
                result = subprocess.run(command, shell=True, timeout=300, executable='/bin/bash')
                return result.returncode == 0, ""
        except subprocess.TimeoutExpired:
            self.log(f"⏱️ Timeout: {description}", "WARNING")
            return False, "Timeout"
        except Exception as e:
            self.log(f"❌ Error: {e}", "ERROR")
            return False, str(e)
    
    def quick_cluster_check(self):
        """Quick check if Ray cluster is already running"""
        self.log("🔍 Checking existing Ray cluster...")
        
        check_cmd = f"/bin/bash -c 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate {self.conda_env} && ray status'"
        success, output = self.run_command(check_cmd, "Check Ray status", capture_output=True)
        
        if success and ("2 node" in output.lower() or "nodes: 2" in output.lower()):
            self.log("✅ Ray cluster already running with 2 nodes!")
            self.log(f"Cluster info:\n{output}")
            return True
        else:
            self.log("⚠️  No active 2-node cluster detected")
            return False
    
    def setup_cluster(self):
        """Set up the Ray cluster if not already running"""
        self.log("🚀 Setting up Ray cluster...")
        
        # Stop any existing Ray processes
        self.log("Cleaning existing Ray processes...")
        self.run_command("/bin/bash -c 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate Training_env && ray stop --force'", "Stop Ray on PC1")
        
        pc2_stop = f"sshpass -p '{self.pc2_password}' ssh w2@{self.pc2_ip} 'source /home/w2/miniconda3/etc/profile.d/conda.sh && conda activate Training_env && ray stop --force'"
        self.run_command(pc2_stop, "Stop Ray on PC2")
        
        time.sleep(3)
        
        # Start Ray head node
        self.log("🖥️  Starting Ray head node on PC1...")
        head_cmd = f"""
        /bin/bash -c '
        source ~/miniconda3/etc/profile.d/conda.sh && \\
        conda activate {self.conda_env} && \\
        ray start --head --node-ip-address={self.pc1_ip} \\
            --dashboard-host=0.0.0.0 \\
            --dashboard-port={RAY_DASHBOARD_PORT} \\
            --num-cpus=80 \\
            --num-gpus=1 \\
            --object-store-memory=19000000000
        '
        """
        
        success, output = self.run_command(head_cmd, "Start Ray head node", capture_output=True)
        if not success:
            self.log(f"❌ Failed to start head node: {output}", "ERROR")
            return False
            
        time.sleep(5)
        
        # Connect PC2 worker
        self.log("🔗 Connecting PC2 as worker...")
        worker_cmd = f"sshpass -p '{self.pc2_password}' ssh w2@{self.pc2_ip} \"source /home/w2/miniconda3/etc/profile.d/conda.sh && conda activate {self.conda_env} && ray start --address='{self.pc1_ip}:6379' --node-ip-address={self.pc2_ip}\""
        
        success, output = self.run_command(worker_cmd, "Connect PC2 worker", capture_output=True)
        if not success:
            self.log(f"❌ Failed to connect PC2: {output}", "ERROR")
            return False
            
        time.sleep(5)
        
        # Verify cluster
        return self.verify_cluster()
    
    def verify_cluster(self):
        """Verify the cluster is ready for training"""
        self.log("✅ Verifying cluster status...")
        
        verify_cmd = f"""
        /bin/bash -c '
        source ~/miniconda3/etc/profile.d/conda.sh && \\
        conda activate {self.conda_env} && \\
        python -c "
import ray
try:
    ray.init(address=\"ray://{self.pc1_ip}:10001\")
    resources = ray.cluster_resources()
    total_cpus = resources.get(\"CPU\", 0)
    total_gpus = resources.get(\"GPU\", 0)
    print(f\"Cluster Resources: {{total_cpus}} CPUs, {{total_gpus}} GPUs\")
    
    if total_cpus >= 90 and total_gpus >= 2:
        print(\"✅ CLUSTER_READY_FOR_TRAINING\")
    else:
        print(f\"❌ INSUFFICIENT_RESOURCES: {{total_cpus}} CPUs, {{total_gpus}} GPUs\")
    ray.shutdown()
except Exception as e:
    print(f\"❌ CLUSTER_ERROR: {{e}}\")
"
        '
        """
        
        success, output = self.run_command(verify_cmd, "Verify cluster resources", capture_output=True)
        
        if success and "✅ CLUSTER_READY_FOR_TRAINING" in output:
            self.log("✅ Cluster verified and ready for training!")
            self.log(f"📊 {output.strip()}")
            return True
        else:
            self.log(f"❌ Cluster verification failed: {output}", "ERROR")
            return False
    
    def launch_training(self, mode="test"):
        """Launch the actual forex bot training"""
        self.log(f"🚀 Launching {mode.upper()} mode training...")
        
        if mode == "test":
            # Quick test training (5 generations)
            training_cmd = f"""
            source ~/miniconda3/etc/profile.d/conda.sh && \\
            conda activate {self.conda_env} && \\
            python -c "
import sys
import os
sys.path.append('/home/w1/cursor-to-copilot-backup/TaskmasterForexBots')

# Set environment for optimal performance
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
os.environ['OMP_NUM_THREADS'] = '40'
os.environ['MKL_NUM_THREADS'] = '40'

print('🧪 TEST MODE: 5 generations, ~5-10 minutes')
print('📡 Connecting to Ray cluster...')

import ray
ray.init(address='ray://192.168.1.10:10001')

print('✅ Connected to cluster')
print('📊 Cluster resources:', ray.cluster_resources())

try:
    # Import the training system
    from comprehensive_trading_system import ComprehensiveTradingSystem
    
    print('🎯 Starting TEST training...')
    
    # Create trading system with test parameters
    trading_system = ComprehensiveTradingSystem()
    
    # Run training with reduced parameters for testing
    results = trading_system.run_training(
        generations=5,           # Quick test
        population_size=10,      # Smaller population
        episodes_per_gen=50,     # Fewer episodes
        steps_per_episode=100    # Shorter episodes
    )
    
    print('✅ TEST training completed!')
    print('📈 Results:', results)
    
except Exception as e:
    print(f'❌ Training error: {{e}}')
    import traceback
    traceback.print_exc()
finally:
    ray.shutdown()
"
            """
        else:
            # Full scale training (200 generations)
            training_cmd = f"""
            source ~/miniconda3/etc/profile.d/conda.sh && \\
            conda activate {self.conda_env} && \\
            python -c "
import sys
import os
sys.path.append('/home/w1/cursor-to-copilot-backup/TaskmasterForexBots')

# Set environment for maximum performance
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
os.environ['OMP_NUM_THREADS'] = '40'
os.environ['MKL_NUM_THREADS'] = '40'

print('🚀 FULL MODE: 200 generations, ~3-4 hours')
print('📡 Connecting to Ray cluster...')

import ray
ray.init(address='ray://192.168.1.10:10001')

print('✅ Connected to cluster')
print('📊 Cluster resources:', ray.cluster_resources())

try:
    # Import the comprehensive training system
    from comprehensive_trading_system import ComprehensiveTradingSystem
    
    print('🎯 Starting FULL SCALE training...')
    print('⚡ Expected duration: 3-4 hours')
    print('🌐 Monitor progress at: http://{self.pc1_ip}:{RAY_DASHBOARD_PORT}')
    
    # Create trading system with full parameters
    trading_system = ComprehensiveTradingSystem()
    
    # Run full training
    results = trading_system.run_training(
        generations=200,         # Full training
        population_size=20,      # Full population
        episodes_per_gen=1000,   # Full episodes
        steps_per_episode=1000   # Full steps
    )
    
    print('🎉 FULL SCALE training completed!')
    print('📈 Final results:', results)
    
except Exception as e:
    print(f'❌ Training error: {{e}}')
    import traceback
    traceback.print_exc()
finally:
    ray.shutdown()
"
            """
        
        # Execute training
        self.log("⚡ Starting training execution...")
        
        try:
            process = subprocess.Popen(
                training_cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Monitor output in real-time
            for line in iter(process.stdout.readline, ''):
                if line:
                    print(line.rstrip())
            
            process.wait()
            
            if process.returncode == 0:
                self.log("🎉 Training completed successfully!", "SUCCESS")
                return True
            else:
                self.log(f"❌ Training failed with code: {process.returncode}", "ERROR")
                return False
                
        except KeyboardInterrupt:
            self.log("🛑 Training interrupted by user", "WARNING")
            process.terminate()
            return False
    
    def cleanup(self):
        """Clean up Ray cluster"""
        self.log("🧹 Cleaning up cluster...")
        
        # Stop Ray on both PCs
        self.run_command("/bin/bash -c 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate Training_env && ray stop --force'", "Stop Ray on PC1")
        
        pc2_stop = f"sshpass -p '{self.pc2_password}' ssh w2@{self.pc2_ip} 'source /home/w2/miniconda3/etc/profile.d/conda.sh && conda activate Training_env && ray stop --force'"
        self.run_command(pc2_stop, "Stop Ray on PC2")
        
        self.log("✅ Cleanup completed")

def main():
    """Main execution function"""
    print("🤖 REAL FOREX BOT TRAINING LAUNCHER")
    print("=" * 50)
    print(f"🖥️  PC1 (Head): {PC1_IP}")
    print(f"🔗 PC2 (Worker): {PC2_IP}")
    print(f"🌐 Dashboard: http://{PC1_IP}:{RAY_DASHBOARD_PORT}")
    print("=" * 50)
    
    launcher = RealTrainingLauncher()
    
    # Setup signal handler for clean exit
    def signal_handler(signum, frame):
        print("\n🛑 Interrupted! Cleaning up...")
        launcher.cleanup()
        sys.exit(1)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Check if cluster is already running
        if not launcher.quick_cluster_check():
            # Set up cluster if not running
            if not launcher.setup_cluster():
                print("❌ Failed to set up cluster")
                return False
        
        # Choose training mode
        print("\n🎮 Select training mode:")
        print("   [1] Test Mode (5 generations, ~5-10 minutes)")
        print("   [2] Full Mode (200 generations, ~3-4 hours)")
        print("   [3] Cancel")
        
        while True:
            choice = input("\nChoice (1/2/3): ").strip()
            if choice == "1":
                mode = "test"
                break
            elif choice == "2":
                mode = "full"
                break
            elif choice == "3":
                print("❌ Cancelled")
                return False
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
        
        # Confirm training
        print(f"\n🚀 Ready to launch {mode.upper()} mode training")
        if mode == "full":
            print("⚠️  Full mode will take 3-4 hours to complete")
        
        confirm = input("Proceed? (y/N): ").strip().lower()
        if confirm != 'y':
            print("❌ Training cancelled")
            return False
        
        # Launch training
        success = launcher.launch_training(mode)
        
        return success
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    finally:
        # Always cleanup
        launcher.cleanup()

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 TRAINING SESSION COMPLETED SUCCESSFULLY!")
    else:
        print("\n❌ Training session failed or was cancelled")
