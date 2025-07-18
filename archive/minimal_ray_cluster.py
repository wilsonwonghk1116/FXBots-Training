#!/usr/bin/env python3
"""
MINIMAL RAY CLUSTER SETUP - NO DASHBOARD, FOCUS ON CONNECTIVITY
Simple approach: Shell commands only, no programmatic init
"""
import subprocess
import time
import sys
import os

def log(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")

def run_command(cmd, description):
    """Run shell command with proper error handling"""
    log(f"Running: {description}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable='/bin/bash')
        if result.returncode == 0:
            log(f"✅ {description} - SUCCESS")
            if result.stdout.strip():
                log(f"Output: {result.stdout.strip()}")
            return True, result.stdout.strip()
        else:
            log(f"❌ {description} - FAILED: {result.stderr.strip()}")
            return False, result.stderr.strip()
    except Exception as e:
        log(f"❌ {description} - EXCEPTION: {e}")
        return False, str(e)

def stop_all_ray():
    """Stop all Ray processes on both machines"""
    log("Stopping all Ray processes on both machines...")
    
    # Stop Ray on PC1
    stop_cmd1 = "source ~/miniconda3/etc/profile.d/conda.sh && conda activate Training_env && ray stop --force"
    run_command(stop_cmd1, "Stop Ray on PC1")
    
    # Stop Ray on PC2
    stop_cmd2 = "sshpass -p 'w' ssh w2@192.168.1.11 'source /home/w2/miniconda3/etc/profile.d/conda.sh && conda activate Training_env && ray stop --force'"
    run_command(stop_cmd2, "Stop Ray on PC2")
    
    time.sleep(3)

def start_ray_head_minimal():
    """Start Ray head with minimal configuration - NO DASHBOARD"""
    log("Starting Ray head node with minimal configuration...")
    
    # Use simplest possible configuration - let Ray handle its own agents
    head_cmd = """
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate Training_env && ray start --head \
    --num-cpus=60 \
    --num-gpus=1 \
    --port=6379 \
    --node-ip-address=192.168.1.10
    """
    
    success, output = run_command(head_cmd, "Start Ray head node (minimal config)")
    
    if success:
        log("✅ Ray head node started successfully")
        time.sleep(5)  # Wait for head node to stabilize
        return True
    else:
        log(f"❌ Failed to start Ray head node: {output}")
        return False

def connect_worker_minimal():
    """Connect PC2 as worker with minimal configuration"""
    log("Connecting PC2 as Ray worker...")
    
    worker_cmd = """
    sshpass -p 'w' ssh w2@192.168.1.11 '
    source /home/w2/miniconda3/etc/profile.d/conda.sh && 
    conda activate Training_env && 
    ray start --address="192.168.1.10:6379" \
    --num-cpus=12 \
    --num-gpus=1 \
    --node-ip-address=192.168.1.11
    '
    """
    
    success, output = run_command(worker_cmd, "Connect PC2 as Ray worker")
    
    if success:
        log("✅ PC2 connected as Ray worker")
        time.sleep(3)  # Wait for worker to register
        return True
    else:
        log(f"❌ Failed to connect PC2: {output}")
        return False

def verify_cluster_minimal():
    """Verify cluster status using ray status"""
    log("Verifying Ray cluster status...")
    
    # Wait a bit more for cluster to stabilize
    time.sleep(5)
    
    # First try ray status
    status_cmd = "source ~/miniconda3/etc/profile.d/conda.sh && conda activate Training_env && ray status"
    success, output = run_command(status_cmd, "Check Ray cluster status")
    
    if success and "No cluster status" not in output:
        log("Ray cluster status:")
        log(output)
        
        # Parse the output to check resources
        lines = output.splitlines()
        cpu_count = 0
        gpu_count = 0
        node_count = 0
        
        for line in lines:
            if "CPU" in line and "/" in line:
                try:
                    # Extract total CPUs from lines like "0.0/72.0 CPU"
                    parts = line.split("/")
                    if len(parts) > 1:
                        cpu_total = float(parts[1].split()[0])
                        cpu_count = max(cpu_count, cpu_total)
                except:
                    pass
            
            if "GPU" in line and "/" in line:
                try:
                    # Extract total GPUs
                    parts = line.split("/")
                    if len(parts) > 1:
                        gpu_total = float(parts[1].split()[0])
                        gpu_count = max(gpu_count, gpu_total)
                except:
                    pass
            
            if "node_" in line.lower() or "Active:" in line:
                node_count += 1
        
        log(f"Detected: {cpu_count} CPUs, {gpu_count} GPUs, {node_count} nodes")
        
        # Check if we have the expected resources
        if cpu_count >= 70 and gpu_count >= 2:
            log("✅ Cluster verification successful!")
            return True
        else:
            log(f"❌ Cluster verification failed - Expected ~72 CPUs and 2 GPUs")
            return False
    else:
        log("Ray status not available, trying programmatic verification...")
        # Try programmatic approach
        return test_simple_ray_task()

def test_simple_ray_task():
    """Test Ray cluster with a simple task"""
    log("Testing Ray cluster with simple task...")
    
    test_script = '''
import ray

@ray.remote
def test_task(x):
    return x * 2

# Test basic functionality
futures = [test_task.remote(i) for i in range(10)]
results = ray.get(futures)
print(f"Test results: {results}")
print(f"Cluster resources: {ray.cluster_resources()}")
    '''
    
    with open("/tmp/ray_test.py", "w") as f:
        f.write(test_script)
    
    test_cmd = "source ~/miniconda3/etc/profile.d/conda.sh && conda activate Training_env && python /tmp/ray_test.py"
    success, output = run_command(test_cmd, "Test Ray cluster functionality")
    
    if success:
        log("✅ Ray cluster test passed!")
        return True
    else:
        log(f"❌ Ray cluster test failed: {output}")
        return False

def main():
    log("🤖 MINIMAL RAY CLUSTER SETUP")
    log("=" * 60)
    log("Objective: Get Ray cluster working with NO dashboard complications")
    log("Configuration: PC1(60 CPUs, 1 GPU) + PC2(12 CPUs, 1 GPU)")
    log("=" * 60)
    
    # Step 1: Clean slate - stop everything
    stop_all_ray()
    
    # Step 2: Start head node with minimal config
    if not start_ray_head_minimal():
        log("❌ FAILED: Cannot start Ray head node")
        return False
    
    # Step 3: Connect worker node
    if not connect_worker_minimal():
        log("❌ FAILED: Cannot connect worker node")
        return False
    
    # Step 4: Verify cluster
    if not verify_cluster_minimal():
        log("❌ FAILED: Cluster verification failed")
        return False
    
    # Step 5: Test cluster functionality
    if not test_simple_ray_task():
        log("❌ FAILED: Cluster functionality test failed")
        return False
    
    log("🎉 SUCCESS: Ray cluster is working correctly!")
    log("Ready for training workloads")
    return True

if __name__ == "__main__":
    success = main()
    if success:
        log("✅ Cluster setup complete - you can now run training")
    else:
        log("❌ Cluster setup failed - check logs above")
    sys.exit(0 if success else 1)
