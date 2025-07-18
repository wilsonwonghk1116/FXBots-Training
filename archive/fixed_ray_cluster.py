#!/usr/bin/env python3
"""
FIXED RAY CLUSTER SETUP - HYBRID APPROACH
Uses programmatic initialization for head node and shell commands for worker
"""
import subprocess
import time
import sys
import ray

def log(message):
def sync_code_to_worker():
    """Sync all Python files to PC2 using scp"""
    log("Syncing code to PC2 via scp...")
    # Copy all .py files in project root to PC2 ~/TaskmasterForexBots/
    scp_cmd = "sshpass -p 'w' scp -o StrictHostKeyChecking=no -r /home/w1/cursor-to-copilot-backup/TaskmasterForexBots/*.py w2@192.168.1.11:/home/w2/TaskmasterForexBots/"
    success, output = run_command(scp_cmd, "SCP Python files to PC2")
    if success:
        log("✅ Code sync to PC2 complete")
        return True
    else:
        log(f"❌ Code sync to PC2 failed: {output}")
        return False
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")

def run_command(cmd, description):
    """Run shell command with proper error handling"""
    log(f"Running: {description}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable='/bin/bash')
        if result.returncode == 0:
            log(f"✅ {description} - SUCCESS")
            return True, result.stdout.strip()
        else:
            log(f"❌ {description} - FAILED: {result.stderr.strip()}")
            return False, result.stderr.strip()
    except Exception as e:
        log(f"❌ {description} - EXCEPTION: {e}")
        return False, str(e)

def start_ray_head_programmatic():
    """Start Ray head node programmatically"""
    log("Starting Ray head node programmatically...")
    
    try:
        # Shutdown any existing Ray instance
        try:
            ray.shutdown()
        except:
            pass
        
        # Initialize Ray with explicit resources and port
        ray.init(
            address=None,  # Start new cluster
            num_cpus=60,
            num_gpus=1,
            _node_ip_address="192.168.1.10",
            include_dashboard=False,
            configure_logging=True,
            log_to_driver=False
        )
        
        log("✅ Ray head node started programmatically")
        log(f"Cluster resources: {ray.cluster_resources()}")
        
        # Get the actual Ray address for workers to connect
        context = ray.runtime_context.get_runtime_context()
        gcs_address = context.gcs_address
        log(f"Ray GCS address: {gcs_address}")
        
        return True, gcs_address
        
    except Exception as e:
        log(f"❌ Failed to start Ray head programmatically: {e}")
        return False, None

def connect_worker_node(ray_address):
    """Connect PC2 as worker node via SSH"""
    log(f"Connecting PC2 as Ray worker to {ray_address}...")
    
    # Stop any existing Ray on PC2
    stop_cmd = "sshpass -p 'w' ssh w2@192.168.1.11 'source /home/w2/miniconda3/etc/profile.d/conda.sh && conda activate Training_env && ray stop --force'"
    run_command(stop_cmd, "Stop any existing Ray processes on PC2")
    time.sleep(2)
    
    # Start Ray worker on PC2 with the correct address
    worker_cmd = f"sshpass -p 'w' ssh w2@192.168.1.11 \"source /home/w2/miniconda3/etc/profile.d/conda.sh && conda activate Training_env && ray start --address='{ray_address}' --num-cpus=12 --num-gpus=1\""
    success, output = run_command(worker_cmd, "Start Ray worker on PC2")
    
    if success:
        log("✅ PC2 connected as Ray worker")
        return True
    else:
        log(f"❌ Failed to connect PC2: {output}")
        return False

def verify_cluster():
    """Verify cluster status"""
    log("Verifying cluster status...")
    
    try:
        resources = ray.cluster_resources()
        log(f"Total cluster resources: {resources}")
        
        cpus = resources.get('CPU', 0)
        gpus = resources.get('GPU', 0)
        
        log(f"Found {cpus} CPUs and {gpus} GPUs")
        
        if cpus >= 70 and gpus >= 2:
            log("✅ Cluster verification successful!")
            return True
        else:
            log(f"❌ Cluster verification failed - Expected ~72 CPUs and 2 GPUs, got {cpus} CPUs and {gpus} GPUs")
            return False
            
    except Exception as e:
        log(f"❌ Failed to verify cluster: {e}")
        return False

def main():
    log("🤖 FIXED RAY CLUSTER SETUP")
    log("=" * 50)
    
    # Step 1: Start Ray head programmatically
    success, ray_address = start_ray_head_programmatic()
    if not success:
        log("❌ Failed to start Ray head node")
        return False

    time.sleep(3)

    # Step 2: Sync code to worker node
    if not sync_code_to_worker():
        log("❌ Failed to sync code to worker node")
        return False

    time.sleep(2)

    # Step 3: Connect worker node
    if not connect_worker_node(ray_address):
        log("❌ Failed to connect worker node")
        return False

    time.sleep(5)

    # Step 4: Verify cluster
    if verify_cluster():
        log("🎉 CLUSTER SETUP SUCCESSFUL!")
        return True
    else:
        log("❌ CLUSTER SETUP FAILED")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
