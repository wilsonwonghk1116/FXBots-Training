#!/usr/bin/env python3
"""
🔍 RAY CLUSTER DEEP DIAGNOSTIC TOOL
Comprehensive analysis of Ray cluster resource issues
"""

import ray
import psutil
import subprocess
import json
import logging
import time
from typing import Dict, List, Any
import GPUtil

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RayClusterDiagnostic:
    def __init__(self):
        self.cluster_info = {}
        
    def check_ray_status(self):
        """Check if Ray is running and get basic info"""
        logger.info("🔍 CHECKING RAY CLUSTER STATUS")
        logger.info("=" * 50)
        
        try:
            # Check if Ray is initialized
            if not ray.is_initialized():
                logger.error("❌ Ray is not initialized!")
                return False
                
            # Get cluster resources
            cluster_resources = ray.cluster_resources()
            available_resources = ray.available_resources()
            
            logger.info("📊 CLUSTER RESOURCES:")
            for resource, amount in cluster_resources.items():
                available = available_resources.get(resource, 0)
                used = amount - available
                utilization = (used / amount * 100) if amount > 0 else 0
                logger.info(f"   {resource}: {used:.1f}/{amount:.1f} ({utilization:.1f}% used)")
                
            self.cluster_info['cluster_resources'] = cluster_resources
            self.cluster_info['available_resources'] = available_resources
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error checking Ray status: {e}")
            return False
    
    def check_actors_and_tasks(self):
        """Check active actors and tasks"""
        logger.info("\n🎭 CHECKING ACTORS AND TASKS")
        logger.info("=" * 50)
        
        try:
            # Get actor information
            actors = ray.util.list_named_actors()
            logger.info(f"📋 Named actors: {len(actors)}")
            for actor in actors:
                logger.info(f"   - {actor}")
            
            # Get node information
            nodes = ray.nodes()
            logger.info(f"\n🖥️  Cluster nodes: {len(nodes)}")
            
            alive_nodes = 0
            dead_nodes = 0
            
            for i, node in enumerate(nodes):
                status = "ALIVE" if node['Alive'] else "DEAD"
                if node['Alive']:
                    alive_nodes += 1
                else:
                    dead_nodes += 1
                    
                logger.info(f"   Node {i+1}: {status}")
                logger.info(f"      IP: {node.get('NodeManagerAddress', 'Unknown')}")
                logger.info(f"      Resources: {node.get('Resources', {})}")
                
                if not node['Alive']:
                    logger.warning(f"      ⚠️  DEAD NODE DETECTED!")
            
            logger.info(f"\n📊 Node Summary: {alive_nodes} alive, {dead_nodes} dead")
            
            self.cluster_info['actors'] = actors
            self.cluster_info['nodes'] = nodes
            self.cluster_info['alive_nodes'] = alive_nodes
            self.cluster_info['dead_nodes'] = dead_nodes
            
        except Exception as e:
            logger.error(f"❌ Error checking actors/tasks: {e}")
    
    def check_system_resources(self):
        """Check system-level resources"""
        logger.info("\n💻 CHECKING SYSTEM RESOURCES")
        logger.info("=" * 50)
        
        # CPU info
        cpu_count = psutil.cpu_count()
        cpu_usage = psutil.cpu_percent(interval=1)
        logger.info(f"🔧 CPU: {cpu_count} cores, {cpu_usage:.1f}% usage")
        
        # Memory info
        memory = psutil.virtual_memory()
        logger.info(f"🧠 RAM: {memory.used/1e9:.1f}GB/{memory.total/1e9:.1f}GB ({memory.percent:.1f}% used)")
        
        # GPU info
        try:
            gpus = GPUtil.getGPUs()
            logger.info(f"🎮 GPUs detected: {len(gpus)}")
            for i, gpu in enumerate(gpus):
                logger.info(f"   GPU {i} ({gpu.name}): {gpu.memoryUtil*100:.1f}% VRAM, {gpu.load*100:.1f}% usage")
                
            self.cluster_info['gpus'] = [(gpu.name, gpu.memoryUtil*100, gpu.load*100) for gpu in gpus]
        except Exception as e:
            logger.warning(f"⚠️  Could not get GPU info: {e}")
            
        self.cluster_info['cpu_count'] = cpu_count
        self.cluster_info['cpu_usage'] = cpu_usage
        self.cluster_info['memory_usage'] = memory.percent
    
    def check_ray_processes(self):
        """Check Ray-related processes"""
        logger.info("\n🔄 CHECKING RAY PROCESSES")
        logger.info("=" * 50)
        
        try:
            # Find Ray processes
            ray_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if 'ray' in proc.info['name'].lower() or any('ray' in cmd.lower() for cmd in proc.info['cmdline']):
                        ray_processes.append({
                            'pid': proc.info['pid'],
                            'name': proc.info['name'],
                            'cmdline': ' '.join(proc.info['cmdline'][:3])  # First 3 args
                        })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            logger.info(f"📋 Ray processes found: {len(ray_processes)}")
            for proc in ray_processes:
                logger.info(f"   PID {proc['pid']}: {proc['name']} - {proc['cmdline']}")
                
            self.cluster_info['ray_processes'] = ray_processes
            
        except Exception as e:
            logger.error(f"❌ Error checking Ray processes: {e}")
    
    def identify_resource_conflicts(self):
        """Identify potential resource allocation conflicts"""
        logger.info("\n⚠️  IDENTIFYING RESOURCE CONFLICTS")
        logger.info("=" * 50)
        
        issues = []
        
        # Check GPU over-allocation
        cluster_resources = self.cluster_info.get('cluster_resources', {})
        available_resources = self.cluster_info.get('available_resources', {})
        
        total_gpus = cluster_resources.get('GPU', 0)
        available_gpus = available_resources.get('GPU', 0)
        used_gpus = total_gpus - available_gpus
        
        if used_gpus > total_gpus:
            issues.append(f"🚨 GPU over-allocation: {used_gpus:.1f} used > {total_gpus:.1f} total")
        
        # Check CPU over-allocation
        total_cpus = cluster_resources.get('CPU', 0)
        available_cpus = available_resources.get('CPU', 0)
        used_cpus = total_cpus - available_cpus
        
        if used_cpus > total_cpus * 0.95:  # 95% threshold
            issues.append(f"🚨 CPU near/over capacity: {used_cpus:.1f}/{total_cpus:.1f} ({used_cpus/total_cpus*100:.1f}%)")
        
        # Check dead nodes
        if self.cluster_info.get('dead_nodes', 0) > 0:
            issues.append(f"🚨 Dead nodes detected: {self.cluster_info['dead_nodes']}")
        
        # Check for resource request pattern from error
        if available_gpus < 1 or available_cpus < 20:
            issues.append(f"🚨 Insufficient resources for typical request (GPU: 1.0, CPU: 20.0)")
            issues.append(f"   Available: GPU: {available_gpus:.1f}, CPU: {available_cpus:.1f}")
        
        if issues:
            logger.warning("🚨 RESOURCE CONFLICTS DETECTED:")
            for issue in issues:
                logger.warning(f"   {issue}")
        else:
            logger.info("✅ No obvious resource conflicts detected")
            
        self.cluster_info['issues'] = issues
        
        return issues
    
    def suggest_fixes(self):
        """Suggest fixes based on identified issues"""
        logger.info("\n🔧 SUGGESTED FIXES")
        logger.info("=" * 50)
        
        issues = self.cluster_info.get('issues', [])
        fixes = []
        
        if any('GPU over-allocation' in issue for issue in issues):
            fixes.append("1. 🔄 Restart Ray cluster: ray stop && ray start --head")
            fixes.append("2. 🧹 Kill zombie Ray processes")
            
        if any('CPU near/over capacity' in issue for issue in issues):
            fixes.append("3. 📉 Reduce CPU allocation per actor (e.g., 15 instead of 20)")
            
        if any('Dead nodes' in issue for issue in issues):
            fixes.append("4. 🖥️  Restart dead worker nodes")
            fixes.append("5. 🌐 Check network connectivity between nodes")
            
        if any('Insufficient resources' in issue for issue in issues):
            fixes.append("6. 📊 Adjust worker configuration to match available resources")
            fixes.append("7. 🎯 Use fewer workers or smaller resource requirements")
        
        # Always suggest cleanup
        fixes.extend([
            "8. 🧼 Clean up any stuck actors: ray.util.list_named_actors() and kill manually",
            "9. 🔍 Check for competing Ray applications",
            "10. 📋 Verify cluster configuration matches hardware"
        ])
        
        logger.info("🎯 RECOMMENDED ACTIONS:")
        for fix in fixes:
            logger.info(f"   {fix}")
            
        return fixes
    
    def generate_report(self):
        """Generate a comprehensive diagnostic report"""
        logger.info("\n📋 GENERATING DIAGNOSTIC REPORT")
        logger.info("=" * 50)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_file = f"ray_diagnostic_report_{timestamp}.json"
        
        try:
            with open(report_file, 'w') as f:
                json.dump(self.cluster_info, f, indent=2, default=str)
            
            logger.info(f"📄 Diagnostic report saved: {report_file}")
            
        except Exception as e:
            logger.error(f"❌ Could not save report: {e}")
    
    def run_full_diagnostic(self):
        """Run complete diagnostic suite"""
        logger.info("🚀 STARTING RAY CLUSTER DEEP DIAGNOSTIC")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Run all checks
        ray_ok = self.check_ray_status()
        if ray_ok:
            self.check_actors_and_tasks()
        
        self.check_system_resources()
        self.check_ray_processes()
        
        if ray_ok:
            self.identify_resource_conflicts()
            self.suggest_fixes()
        
        self.generate_report()
        
        duration = time.time() - start_time
        logger.info(f"\n🎉 DIAGNOSTIC COMPLETE ({duration:.1f}s)")
        logger.info("=" * 60)

def main():
    """Run the diagnostic"""
    
    # Initialize Ray if not already done
    try:
        if not ray.is_initialized():
            logger.info("🔌 Connecting to existing Ray cluster...")
            ray.init(address='auto')
    except Exception as e:
        logger.warning(f"⚠️  Could not connect to Ray: {e}")
        logger.info("ℹ️  Will run system-level diagnostics only")
    
    # Run diagnostic
    diagnostic = RayClusterDiagnostic()
    diagnostic.run_full_diagnostic()
    
    # Cleanup
    if ray.is_initialized():
        ray.shutdown()

if __name__ == "__main__":
    main() 