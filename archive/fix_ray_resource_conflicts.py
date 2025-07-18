#!/usr/bin/env python3
"""
🔧 RAY CLUSTER RESOURCE CONFLICT FIXER
Automatically fixes common Ray resource allocation issues
"""

import ray
import subprocess
import time
import logging
import os
import signal
import psutil

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RayClusterFixer:
    def __init__(self):
        self.fixed_issues = []
        
    def kill_ray_processes(self):
        """Kill all Ray processes to ensure clean restart"""
        logger.info("🧹 CLEANING UP RAY PROCESSES")
        logger.info("=" * 50)
        
        killed_processes = 0
        ray_processes = []
        
        # Find all Ray processes
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'ray' in proc.info['name'].lower() or any('ray' in str(cmd).lower() for cmd in proc.info['cmdline']):
                    ray_processes.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        logger.info(f"📋 Found {len(ray_processes)} Ray processes")
        
        # Kill processes gracefully first
        for proc in ray_processes:
            try:
                logger.info(f"🔄 Terminating PID {proc.pid}: {proc.name()}")
                proc.terminate()
                killed_processes += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        # Wait a bit for graceful shutdown
        time.sleep(3)
        
        # Force kill any remaining processes
        for proc in ray_processes:
            try:
                if proc.is_running():
                    logger.info(f"💥 Force killing PID {proc.pid}: {proc.name()}")
                    proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        logger.info(f"✅ Cleaned up {killed_processes} Ray processes")
        self.fixed_issues.append(f"Killed {killed_processes} Ray processes")
        
        return killed_processes > 0
    
    def shutdown_ray_gracefully(self):
        """Shutdown Ray gracefully if initialized"""
        logger.info("🛑 SHUTTING DOWN RAY GRACEFULLY")
        logger.info("=" * 50)
        
        try:
            if ray.is_initialized():
                logger.info("📤 Shutting down Ray connection...")
                ray.shutdown()
                logger.info("✅ Ray shutdown complete")
                self.fixed_issues.append("Ray shutdown gracefully")
                return True
            else:
                logger.info("ℹ️  Ray was not initialized")
                return False
        except Exception as e:
            logger.warning(f"⚠️  Error during Ray shutdown: {e}")
            return False
    
    def clean_ray_temp_files(self):
        """Clean up Ray temporary files"""
        logger.info("🗂️  CLEANING RAY TEMP FILES")
        logger.info("=" * 50)
        
        temp_dirs = [
            "/tmp/ray",
            "/tmp/ray_current_cluster",
            os.path.expanduser("~/ray_results"),
            "/dev/shm/ray*"
        ]
        
        cleaned_files = 0
        
        for temp_dir in temp_dirs:
            try:
                if '*' in temp_dir:
                    # Use shell for wildcard expansion
                    result = subprocess.run(f"rm -rf {temp_dir}", shell=True, capture_output=True, text=True)
                    if result.returncode == 0:
                        logger.info(f"🗑️  Cleaned wildcard: {temp_dir}")
                        cleaned_files += 1
                else:
                    if os.path.exists(temp_dir):
                        subprocess.run(f"rm -rf {temp_dir}", shell=True, check=True)
                        logger.info(f"🗑️  Cleaned: {temp_dir}")
                        cleaned_files += 1
            except subprocess.CalledProcessError as e:
                logger.warning(f"⚠️  Could not clean {temp_dir}: {e}")
            except Exception as e:
                logger.warning(f"⚠️  Error cleaning {temp_dir}: {e}")
        
        logger.info(f"✅ Cleaned {cleaned_files} temp locations")
        self.fixed_issues.append(f"Cleaned {cleaned_files} temp directories")
        
        return cleaned_files > 0
    
    def restart_ray_cluster(self, reduced_resources=True):
        """Restart Ray cluster with optimized resource configuration"""
        logger.info("🚀 RESTARTING RAY CLUSTER")
        logger.info("=" * 50)
        
        try:
            # Stop any existing Ray instance
            logger.info("🛑 Stopping Ray cluster...")
            subprocess.run("ray stop", shell=True, capture_output=True)
            time.sleep(2)
            
            # Start Ray head with conservative resource allocation
            if reduced_resources:
                # Reduce resource allocation to prevent conflicts
                ray_cmd = (
                    "ray start --head --port=6379 "
                    "--dashboard-host=0.0.0.0 --dashboard-port=8265 "
                    "--num-cpus=30 --num-gpus=1 "  # Conservative allocation
                    "--object-store-memory=4000000000"  # 4GB object store
                )
            else:
                ray_cmd = "ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265"
            
            logger.info(f"🔄 Starting Ray cluster: {ray_cmd}")
            result = subprocess.run(ray_cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ Ray cluster started successfully")
                logger.info(f"📊 Ray dashboard: http://localhost:8265")
                self.fixed_issues.append("Ray cluster restarted with optimized resources")
                
                # Wait for cluster to be ready
                time.sleep(3)
                
                # Verify cluster is working
                try:
                    ray.init(address='auto')
                    resources = ray.cluster_resources()
                    logger.info(f"🎯 Cluster resources: {resources}")
                    ray.shutdown()
                    return True
                except Exception as e:
                    logger.error(f"❌ Cluster verification failed: {e}")
                    return False
                    
            else:
                logger.error(f"❌ Failed to start Ray cluster: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error restarting Ray cluster: {e}")
            return False
    
    def fix_resource_configuration(self):
        """Create optimized resource configuration"""
        logger.info("⚙️  CREATING OPTIMIZED RESOURCE CONFIG")
        logger.info("=" * 50)
        
        config_content = '''
# Optimized Ray Cluster Configuration
# Conservative resource allocation to prevent conflicts

import ray

# Resource configuration for RTX 3070 + RTX 3090 setup
CONSERVATIVE_CONFIG = {
    "pc1_resources": {
        "num_cpus": 15,  # Reduced from 20
        "num_gpus": 0.8,  # Reduced from 1.0
        "memory": 12 * 1024 * 1024 * 1024,  # 12GB
        "object_store_memory": 4 * 1024 * 1024 * 1024  # 4GB
    },
    "pc2_resources": {
        "num_cpus": 12,  # Reduced from 20
        "num_gpus": 0.7,  # Reduced from 1.0  
        "memory": 8 * 1024 * 1024 * 1024,  # 8GB
        "object_store_memory": 3 * 1024 * 1024 * 1024  # 3GB
    }
}

# Actor configuration
ACTOR_CONFIG = {
    "cpu_per_actor": 15,  # Reduced from 20
    "gpu_per_actor": 0.8,  # Reduced from 1.0
    "max_actors": 2  # Ensure we only create 2 actors total
}

def get_conservative_resources():
    """Get conservative resource allocation"""
    return CONSERVATIVE_CONFIG

def get_actor_config():
    """Get actor configuration"""
    return ACTOR_CONFIG
'''
        
        try:
            with open("ray_conservative_config.py", "w") as f:
                f.write(config_content)
            
            logger.info("✅ Created ray_conservative_config.py")
            self.fixed_issues.append("Created optimized resource configuration")
            return True
            
        except Exception as e:
            logger.error(f"❌ Could not create config file: {e}")
            return False
    
    def create_fixed_trainer_config(self):
        """Create a corrected trainer configuration"""
        logger.info("🎯 CREATING FIXED TRAINER CONFIG")
        logger.info("=" * 50)
        
        config_content = '''#!/usr/bin/env python3
"""
🔧 FIXED RTX 3070 TRAINER CONFIGURATION
Conservative resource allocation to prevent Ray conflicts
"""

import ray
from ray_conservative_config import get_conservative_resources, get_actor_config

# Import the original trainer but with fixed configuration
class FixedRTX3070Trainer:
    def __init__(self):
        self.config = get_actor_config()
        
    def start_training(self, duration_minutes=1):
        """Start training with fixed resource allocation"""
        print("🚀 Starting FIXED RTX 3070 Trainer")
        print("=" * 50)
        
        # Conservative resource allocation
        actor_config = self.config
        
        try:
            # Connect to Ray
            if not ray.is_initialized():
                ray.init(address='auto')
            
            print(f"📊 Using conservative config: {actor_config}")
            
            # Create exactly 2 actors with reduced resources
            @ray.remote(
                num_cpus=actor_config['cpu_per_actor'],
                num_gpus=actor_config['gpu_per_actor']
            )
            class ConservativeWorker:
                def train(self, duration):
                    import time
                    import torch
                    
                    # Smaller batch size for reduced memory
                    batch_size = 256  # Reduced from 512
                    iterations = 0
                    
                    start_time = time.time()
                    end_time = start_time + duration * 60
                    
                    while time.time() < end_time:
                        # Simulate smaller training step
                        x = torch.randn(batch_size, 128).cuda()
                        y = torch.nn.functional.relu(x)
                        loss = y.mean()
                        
                        iterations += 1
                        
                        # Memory cleanup every 10 iterations
                        if iterations % 10 == 0:
                            torch.cuda.empty_cache()
                    
                    return {
                        "iterations": iterations,
                        "duration": time.time() - start_time,
                        "operations": iterations * batch_size
                    }
            
            # Create only 2 workers to prevent resource conflicts
            workers = [ConservativeWorker.remote() for _ in range(2)]
            print(f"✅ Created {len(workers)} conservative workers")
            
            # Start training
            futures = [worker.train.remote(duration_minutes) for worker in workers]
            
            # Monitor with simple progress
            print(f"🎯 Training for {duration_minutes} minute(s)...")
            
            # Use ray.wait for non-blocking monitoring
            remaining = futures[:]
            completed_count = 0
            
            while remaining:
                ready, remaining = ray.wait(remaining, timeout=10.0)
                
                for completed_future in ready:
                    completed_count += 1
                    print(f"✅ Worker {completed_count}/2 completed")
            
            # Collect results
            results = ray.get(futures)
            
            print("🎉 TRAINING COMPLETE!")
            for i, result in enumerate(results):
                print(f"📊 Worker {i+1}: {result['iterations']} iterations, {result['operations']} operations")
            
            return results
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return None
        finally:
            # Clean up workers
            try:
                for worker in workers:
                    ray.kill(worker)
            except:
                pass

if __name__ == "__main__":
    trainer = FixedRTX3070Trainer()
    trainer.start_training(1)  # 1 minute test
'''
        
        try:
            with open("fixed_rtx3070_trainer.py", "w") as f:
                f.write(config_content)
            
            logger.info("✅ Created fixed_rtx3070_trainer.py")
            self.fixed_issues.append("Created fixed trainer with conservative resources")
            return True
            
        except Exception as e:
            logger.error(f"❌ Could not create fixed trainer: {e}")
            return False
    
    def run_complete_fix(self):
        """Run the complete fix procedure"""
        logger.info("🔧 STARTING COMPLETE RAY CLUSTER FIX")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Step 1: Graceful shutdown
        self.shutdown_ray_gracefully()
        
        # Step 2: Kill processes
        self.kill_ray_processes()
        
        # Step 3: Clean temp files
        self.clean_ray_temp_files()
        
        # Step 4: Create optimized config
        self.fix_resource_configuration()
        
        # Step 5: Restart with conservative resources
        success = self.restart_ray_cluster(reduced_resources=True)
        
        # Step 6: Create fixed trainer
        if success:
            self.create_fixed_trainer_config()
        
        duration = time.time() - start_time
        
        logger.info("\n" + "=" * 60)
        logger.info("🎉 CLUSTER FIX COMPLETE!")
        logger.info(f"⏱️  Total time: {duration:.1f}s")
        logger.info("\n📋 ISSUES FIXED:")
        for i, issue in enumerate(self.fixed_issues, 1):
            logger.info(f"   {i}. {issue}")
        
        if success:
            logger.info("\n🚀 NEXT STEPS:")
            logger.info("   1. Test with: python fixed_rtx3070_trainer.py")
            logger.info("   2. Monitor with: python ray_cluster_deep_diagnostic.py")
            logger.info("   3. Check dashboard: http://localhost:8265")
        else:
            logger.error("\n❌ CLUSTER RESTART FAILED - Manual intervention required")
        
        return success

def main():
    """Main function"""
    fixer = RayClusterFixer()
    fixer.run_complete_fix()

if __name__ == "__main__":
    main() 