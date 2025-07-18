#!/usr/bin/env python3
"""
🧪 COMPLETE RAY CLUSTER FIX VERIFICATION
Test all components of the Ray resource conflict solution
"""

import ray
import subprocess
import time
import logging
import os
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompleteFixTester:
    def __init__(self):
        self.test_results = {}
        
    def test_ray_cluster_status(self):
        """Test 1: Ray cluster is running properly"""
        logger.info("🧪 TEST 1: RAY CLUSTER STATUS")
        logger.info("=" * 50)
        
        try:
            if not ray.is_initialized():
                ray.init(address='auto')
            
            # Check basic cluster info
            cluster_resources = ray.cluster_resources()
            available_resources = ray.available_resources()
            
            logger.info(f"📊 Cluster resources: {cluster_resources}")
            logger.info(f"💡 Available resources: {available_resources}")
            
            # Verify minimum requirements
            total_cpu = cluster_resources.get('CPU', 0)
            total_gpu = cluster_resources.get('GPU', 0)
            available_cpu = available_resources.get('CPU', 0)
            available_gpu = available_resources.get('GPU', 0)
            
            success = True
            issues = []
            
            if total_cpu < 20:
                issues.append(f"Low CPU count: {total_cpu}")
                success = False
            
            if total_gpu < 1:
                issues.append(f"No GPU available: {total_gpu}")
                success = False
                
            if available_cpu < 15:
                issues.append(f"Insufficient available CPU: {available_cpu}")
                
            if available_gpu < 0.5:
                issues.append(f"Insufficient available GPU: {available_gpu}")
            
            if success and not issues:
                logger.info("✅ Ray cluster status: HEALTHY")
                self.test_results['cluster_status'] = "PASSED"
            else:
                logger.warning(f"⚠️  Ray cluster issues: {issues}")
                self.test_results['cluster_status'] = f"WARNING: {issues}"
                
        except Exception as e:
            logger.error(f"❌ Ray cluster test failed: {e}")
            self.test_results['cluster_status'] = f"FAILED: {e}"
    
    def test_conservative_config_exists(self):
        """Test 2: Conservative configuration files exist"""
        logger.info("\n🧪 TEST 2: CONFIGURATION FILES")
        logger.info("=" * 50)
        
        required_files = [
            "ray_conservative_config.py",
            "fixed_rtx3070_trainer.py"
        ]
        
        missing_files = []
        
        for file in required_files:
            if os.path.exists(file):
                logger.info(f"✅ Found: {file}")
            else:
                logger.error(f"❌ Missing: {file}")
                missing_files.append(file)
        
        if not missing_files:
            self.test_results['config_files'] = "PASSED"
        else:
            self.test_results['config_files'] = f"FAILED: Missing {missing_files}"
    
    def test_actor_creation(self):
        """Test 3: Conservative actor creation"""
        logger.info("\n🧪 TEST 3: CONSERVATIVE ACTOR CREATION")
        logger.info("=" * 50)
        
        try:
            # Try to create conservative actors
            @ray.remote(num_cpus=10, num_gpus=0.5)
            class TestWorker:
                def ping(self):
                    return "pong"
            
            # Create 2 test workers
            workers = []
            for i in range(2):
                try:
                    worker = TestWorker.remote()
                    workers.append(worker)
                    logger.info(f"✅ Created test worker {i+1}")
                except Exception as e:
                    logger.error(f"❌ Failed to create worker {i+1}: {e}")
                    self.test_results['actor_creation'] = f"FAILED: {e}"
                    return
            
            # Test worker communication
            futures = [worker.ping.remote() for worker in workers]
            results = ray.get(futures, timeout=10.0)
            
            if all(result == "pong" for result in results):
                logger.info("✅ All workers responding correctly")
                self.test_results['actor_creation'] = "PASSED"
            else:
                logger.error(f"❌ Worker communication failed: {results}")
                self.test_results['actor_creation'] = f"FAILED: Communication error"
            
            # Clean up workers
            for worker in workers:
                ray.kill(worker)
            
        except Exception as e:
            logger.error(f"❌ Actor creation test failed: {e}")
            self.test_results['actor_creation'] = f"FAILED: {e}"
    
    def test_resource_allocation(self):
        """Test 4: Resource allocation doesn't cause conflicts"""
        logger.info("\n🧪 TEST 4: RESOURCE ALLOCATION TEST")
        logger.info("=" * 50)
        
        try:
            # Check current resource usage
            cluster_resources = ray.cluster_resources()
            available_resources = ray.available_resources()
            
            # Calculate usage
            total_cpu = cluster_resources.get('CPU', 0)
            available_cpu = available_resources.get('CPU', 0)
            used_cpu = total_cpu - available_cpu
            
            total_gpu = cluster_resources.get('GPU', 0)
            available_gpu = available_resources.get('GPU', 0)
            used_gpu = total_gpu - available_gpu
            
            cpu_usage_percent = (used_cpu / total_cpu * 100) if total_cpu > 0 else 0
            gpu_usage_percent = (used_gpu / total_gpu * 100) if total_gpu > 0 else 0
            
            logger.info(f"📊 CPU usage: {used_cpu:.1f}/{total_cpu:.1f} ({cpu_usage_percent:.1f}%)")
            logger.info(f"🎮 GPU usage: {used_gpu:.1f}/{total_gpu:.1f} ({gpu_usage_percent:.1f}%)")
            
            # Check if usage is reasonable (not over-allocated)
            if cpu_usage_percent > 100:
                logger.error(f"❌ CPU over-allocated: {cpu_usage_percent:.1f}%")
                self.test_results['resource_allocation'] = "FAILED: CPU over-allocation"
            elif gpu_usage_percent > 100:
                logger.error(f"❌ GPU over-allocated: {gpu_usage_percent:.1f}%")
                self.test_results['resource_allocation'] = "FAILED: GPU over-allocation"
            elif cpu_usage_percent > 90 or gpu_usage_percent > 90:
                logger.warning(f"⚠️  High resource usage: CPU {cpu_usage_percent:.1f}%, GPU {gpu_usage_percent:.1f}%")
                self.test_results['resource_allocation'] = "WARNING: High usage"
            else:
                logger.info("✅ Resource allocation looks healthy")
                self.test_results['resource_allocation'] = "PASSED"
                
        except Exception as e:
            logger.error(f"❌ Resource allocation test failed: {e}")
            self.test_results['resource_allocation'] = f"FAILED: {e}"
    
    def test_fixed_trainer_import(self):
        """Test 5: Fixed trainer can be imported"""
        logger.info("\n🧪 TEST 5: FIXED TRAINER IMPORT")
        logger.info("=" * 50)
        
        try:
            # Try to import the conservative config dynamically
            if os.path.exists("ray_conservative_config.py"):
                try:
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("ray_conservative_config", "ray_conservative_config.py")
                    if spec is not None and spec.loader is not None:
                        ray_conservative_config = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(ray_conservative_config)
                        config = ray_conservative_config.get_actor_config()
                        logger.info(f"✅ Conservative config loaded: {config}")
                    else:
                        raise ImportError("Could not create module spec")
                except Exception as e:
                    logger.error(f"❌ Could not import ray_conservative_config: {e}")
                    self.test_results['trainer_import'] = "FAILED: Config import error"
                    return
            else:
                logger.error("❌ ray_conservative_config.py not found")
                self.test_results['trainer_import'] = "FAILED: Config file missing"
                return
            
            # Check if fixed trainer exists and has correct structure
            if os.path.exists("fixed_rtx3070_trainer.py"):
                with open("fixed_rtx3070_trainer.py", 'r') as f:
                    content = f.read()
                    
                # Check for key components
                required_components = [
                    "FixedRTX3070Trainer",
                    "ConservativeWorker",
                    "ray.remote",
                    "get_actor_config"
                ]
                
                missing_components = []
                for component in required_components:
                    if component not in content:
                        missing_components.append(component)
                
                if not missing_components:
                    logger.info("✅ Fixed trainer has all required components")
                    self.test_results['trainer_import'] = "PASSED"
                else:
                    logger.error(f"❌ Fixed trainer missing components: {missing_components}")
                    self.test_results['trainer_import'] = f"FAILED: Missing {missing_components}"
            else:
                logger.error("❌ fixed_rtx3070_trainer.py not found")
                self.test_results['trainer_import'] = "FAILED: Trainer file missing"
                
        except Exception as e:
            logger.error(f"❌ Trainer import test failed: {e}")
            self.test_results['trainer_import'] = f"FAILED: {e}"
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        logger.info("\n📋 GENERATING TEST REPORT")
        logger.info("=" * 60)
        
        passed_tests = sum(1 for result in self.test_results.values() if result == "PASSED")
        total_tests = len(self.test_results)
        warning_tests = sum(1 for result in self.test_results.values() if result.startswith("WARNING"))
        failed_tests = sum(1 for result in self.test_results.values() if result.startswith("FAILED"))
        
        logger.info(f"📊 TEST SUMMARY:")
        logger.info(f"   ✅ Passed: {passed_tests}/{total_tests}")
        logger.info(f"   ⚠️  Warnings: {warning_tests}/{total_tests}")
        logger.info(f"   ❌ Failed: {failed_tests}/{total_tests}")
        
        logger.info(f"\n📋 DETAILED RESULTS:")
        for test_name, result in self.test_results.items():
            status_icon = "✅" if result == "PASSED" else "⚠️" if result.startswith("WARNING") else "❌"
            logger.info(f"   {status_icon} {test_name}: {result}")
        
        # Overall assessment
        if failed_tests == 0:
            if warning_tests == 0:
                logger.info("\n🎉 ALL TESTS PASSED! Ray cluster fix is working perfectly!")
                overall_status = "SUCCESS"
            else:
                logger.info("\n🎯 MOSTLY WORKING! Some warnings but no critical failures.")
                overall_status = "WARNING"
        else:
            logger.error("\n💥 SOME TESTS FAILED! Manual intervention may be required.")
            overall_status = "FAILED"
        
        # Save report
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_file = f"fix_verification_report_{timestamp}.txt"
        
        try:
            with open(report_file, 'w') as f:
                f.write(f"Ray Cluster Fix Verification Report\n")
                f.write(f"Generated: {timestamp}\n")
                f.write(f"Overall Status: {overall_status}\n\n")
                f.write(f"Test Results:\n")
                for test_name, result in self.test_results.items():
                    f.write(f"{test_name}: {result}\n")
            
            logger.info(f"📄 Test report saved: {report_file}")
            
        except Exception as e:
            logger.warning(f"⚠️  Could not save report: {e}")
        
        return overall_status
    
    def run_complete_verification(self):
        """Run all verification tests"""
        logger.info("🧪 STARTING RAY CLUSTER FIX VERIFICATION")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Run all tests
        self.test_ray_cluster_status()
        self.test_conservative_config_exists()
        self.test_actor_creation()
        self.test_resource_allocation()
        self.test_fixed_trainer_import()
        
        # Generate report
        overall_status = self.generate_test_report()
        
        duration = time.time() - start_time
        logger.info(f"\n🎉 VERIFICATION COMPLETE ({duration:.1f}s)")
        logger.info("=" * 60)
        
        return overall_status

def main():
    """Run the verification"""
    
    # Initialize Ray if not already done
    try:
        if not ray.is_initialized():
            logger.info("🔌 Connecting to Ray cluster...")
            ray.init(address='auto')
    except Exception as e:
        logger.warning(f"⚠️  Could not connect to Ray: {e}")
        logger.info("ℹ️  Will run limited tests")
    
    # Run verification
    tester = CompleteFixTester()
    status = tester.run_complete_verification()
    
    # Cleanup
    if ray.is_initialized():
        ray.shutdown()
    
    # Exit with appropriate code
    if status == "SUCCESS":
        logger.info("🎊 Ready to run: python fixed_rtx3070_trainer.py")
        exit(0)
    elif status == "WARNING":
        logger.info("⚠️  Proceed with caution")
        exit(1)
    else:
        logger.error("💥 Fix required before proceeding")
        exit(2)

if __name__ == "__main__":
    main() 