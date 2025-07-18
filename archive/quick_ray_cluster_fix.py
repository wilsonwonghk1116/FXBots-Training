#!/usr/bin/env python3
"""
🚀 QUICK RAY CLUSTER FIX & VERIFY
One-stop solution to diagnose, fix, and verify Ray cluster
"""

import subprocess
import time
import logging
import os
import psutil

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QuickRayFixer:
    def __init__(self):
        self.status = {}
        
    def check_ray_processes(self):
        """Check if Ray processes are running"""
        logger.info("🔍 CHECKING RAY PROCESSES")
        logger.info("=" * 50)
        
        ray_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'ray' in proc.info['name'].lower() or any('ray' in str(cmd).lower() for cmd in proc.info['cmdline']):
                    ray_processes.append({
                        'pid': proc.info['pid'],
                        'name': proc.info['name'],
                        'cmdline': ' '.join(proc.info['cmdline'][:3])
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        logger.info(f"📋 Found {len(ray_processes)} Ray processes:")
        for proc in ray_processes:
            logger.info(f"   PID {proc['pid']}: {proc['name']} - {proc['cmdline']}")
        
        self.status['ray_processes'] = len(ray_processes)
        return len(ray_processes) > 0
    
    def test_ray_connection(self):
        """Test if we can connect to Ray"""
        logger.info("\n🔌 TESTING RAY CONNECTION")
        logger.info("=" * 50)
        
        try:
            import ray
            if ray.is_initialized():
                logger.info("ℹ️  Ray already initialized, shutting down first...")
                ray.shutdown()
            
            logger.info("🔄 Attempting to connect to Ray cluster...")
            ray.init(address='auto', _temp_dir='/tmp/ray')
            
            # Test basic functionality
            cluster_resources = ray.cluster_resources()
            logger.info(f"✅ Ray connection successful! Resources: {cluster_resources}")
            
            ray.shutdown()
            self.status['ray_connection'] = "SUCCESS"
            return True
            
        except Exception as e:
            logger.error(f"❌ Ray connection failed: {e}")
            self.status['ray_connection'] = f"FAILED: {e}"
            return False
    
    def kill_all_ray_processes(self):
        """Kill all Ray processes"""
        logger.info("\n🧹 KILLING ALL RAY PROCESSES")
        logger.info("=" * 50)
        
        # Method 1: Use Ray's built-in stop
        try:
            logger.info("🛑 Using 'ray stop' command...")
            result = subprocess.run("ray stop", shell=True, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                logger.info("✅ Ray stop command successful")
            else:
                logger.warning(f"⚠️  Ray stop returned {result.returncode}: {result.stderr}")
        except subprocess.TimeoutExpired:
            logger.warning("⚠️  Ray stop command timed out")
        except Exception as e:
            logger.warning(f"⚠️  Ray stop failed: {e}")
        
        time.sleep(2)
        
        # Method 2: Kill processes manually
        killed_count = 0
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'ray' in proc.info['name'].lower() or any('ray' in str(cmd).lower() for cmd in proc.info['cmdline']):
                    logger.info(f"💥 Killing PID {proc.info['pid']}: {proc.info['name']}")
                    proc.kill()
                    killed_count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        logger.info(f"✅ Killed {killed_count} Ray processes")
        return killed_count
    
    def clean_ray_temp_files(self):
        """Clean Ray temporary files"""
        logger.info("\n🗑️  CLEANING RAY TEMP FILES")
        logger.info("=" * 50)
        
        temp_locations = [
            "/tmp/ray",
            "/tmp/ray_current_cluster", 
            "/dev/shm/ray*",
            os.path.expanduser("~/ray_results")
        ]
        
        cleaned_count = 0
        for location in temp_locations:
            try:
                if '*' in location:
                    result = subprocess.run(f"rm -rf {location}", shell=True, capture_output=True)
                    if result.returncode == 0:
                        logger.info(f"🗑️  Cleaned: {location}")
                        cleaned_count += 1
                else:
                    if os.path.exists(location):
                        subprocess.run(f"rm -rf {location}", shell=True, check=True)
                        logger.info(f"🗑️  Cleaned: {location}")
                        cleaned_count += 1
            except Exception as e:
                logger.warning(f"⚠️  Could not clean {location}: {e}")
        
        logger.info(f"✅ Cleaned {cleaned_count} temp locations")
        return cleaned_count
    
    def start_ray_cluster(self):
        """Start Ray cluster with conservative settings"""
        logger.info("\n🚀 STARTING RAY CLUSTER")
        logger.info("=" * 50)
        
        # Conservative Ray startup command
        ray_cmd = (
            "ray start --head --port=6379 "
            "--dashboard-host=0.0.0.0 --dashboard-port=8265 "
            "--num-cpus=25 --num-gpus=1 "  # Conservative allocation
            "--object-store-memory=3000000000 "  # 3GB object store
            "--temp-dir=/tmp/ray"
        )
        
        logger.info(f"🔄 Starting Ray: {ray_cmd}")
        
        try:
            result = subprocess.run(ray_cmd, shell=True, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                logger.info("✅ Ray cluster started successfully!")
                logger.info("📊 Ray dashboard: http://localhost:8265")
                
                # Wait for cluster to be ready
                time.sleep(5)
                
                # Verify it's working
                if self.test_ray_connection():
                    self.status['ray_startup'] = "SUCCESS"
                    return True
                else:
                    logger.error("❌ Ray started but connection test failed")
                    self.status['ray_startup'] = "FAILED: Connection test failed"
                    return False
            else:
                logger.error(f"❌ Ray startup failed: {result.stderr}")
                self.status['ray_startup'] = f"FAILED: {result.stderr}"
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("❌ Ray startup timed out")
            self.status['ray_startup'] = "FAILED: Timeout"
            return False
        except Exception as e:
            logger.error(f"❌ Ray startup error: {e}")
            self.status['ray_startup'] = f"FAILED: {e}"
            return False
    
    def verify_fix(self):
        """Verify the fix worked"""
        logger.info("\n🧪 VERIFYING FIX")
        logger.info("=" * 50)
        
        # Test 1: Ray connection
        if not self.test_ray_connection():
            return False
        
        # Test 2: Create test actors
        try:
            import ray
            ray.init(address='auto')
            
            @ray.remote(num_cpus=5, num_gpus=0.3)
            class TestActor:
                def test(self):
                    return "success"
            
            # Create 2 test actors
            actors = [TestActor.remote() for _ in range(2)]
            results = ray.get([actor.test.remote() for actor in actors], timeout=10.0)  # type: ignore
            
            if all(r == "success" for r in results):
                logger.info("✅ Test actors created and working!")
                success = True
            else:
                logger.error(f"❌ Test actor results: {results}")
                success = False
            
            # Clean up
            for actor in actors:
                ray.kill(actor)
            
            ray.shutdown()
            return success
            
        except Exception as e:
            logger.error(f"❌ Actor test failed: {e}")
            return False
    
    def run_complete_fix(self):
        """Run the complete fix process"""
        logger.info("🔧 STARTING COMPLETE RAY CLUSTER FIX")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Step 1: Check current status
        has_processes = self.check_ray_processes()
        can_connect = self.test_ray_connection()
        
        if can_connect:
            logger.info("🎉 Ray cluster is already working! No fix needed.")
            return True
        
        # Step 2: Clean up everything
        logger.info("\n🧹 CLEANING UP RAY CLUSTER...")
        self.kill_all_ray_processes()
        self.clean_ray_temp_files()
        
        # Step 3: Start fresh
        logger.info("\n🚀 STARTING FRESH RAY CLUSTER...")
        if not self.start_ray_cluster():
            logger.error("❌ Failed to start Ray cluster")
            return False
        
        # Step 4: Verify fix
        logger.info("\n🧪 VERIFYING THE FIX...")
        if not self.verify_fix():
            logger.error("❌ Fix verification failed")
            return False
        
        duration = time.time() - start_time
        
        logger.info("\n" + "=" * 60)
        logger.info("🎉 RAY CLUSTER FIX COMPLETE!")
        logger.info(f"⏱️  Total time: {duration:.1f}s")
        logger.info("\n📊 STATUS SUMMARY:")
        for key, value in self.status.items():
            logger.info(f"   {key}: {value}")
        
        logger.info("\n🚀 NEXT STEPS:")
        logger.info("   1. Run verification: python test_complete_fix.py")
        logger.info("   2. Test trainer: python fixed_rtx3070_trainer.py")
        logger.info("   3. Check dashboard: http://localhost:8265")
        
        return True

def main():
    """Main function"""
    fixer = QuickRayFixer()
    success = fixer.run_complete_fix()
    
    if success:
        logger.info("🎊 RAY CLUSTER IS READY!")
        exit(0)
    else:
        logger.error("💥 FIX FAILED - Manual intervention required")
        exit(1)

if __name__ == "__main__":
    main() 