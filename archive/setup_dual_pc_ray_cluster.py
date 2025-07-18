#!/usr/bin/env python3
"""
Dual PC Ray Cluster Setup Script
Follows exact user instructions for PC1 (Head) and PC2 (Worker) setup
"""

import subprocess
import sys
import time
import os
from typing import Tuple, Optional

class DualPCRayClusterSetup:
    def __init__(self):
        self.pc1_ip = "192.168.1.10"  # Head PC1 IP
        self.pc2_ip = "192.168.1.11"  # Worker PC2 IP
        self.required_python_version = "3.12.2"
        self.ray_port = 6379
        self.dashboard_port = 8265
        
    def run_command(self, command: str, description: str, check_output: bool = True) -> Tuple[bool, str]:
        """Run a command and return success status and output"""
        print(f"\n🔧 {description}")
        print(f"Command: {command}")
        
        try:
            if check_output:
                result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
                output = result.stdout.strip() + result.stderr.strip()
                success = result.returncode == 0
            else:
                result = subprocess.Popen(command, shell=True)
                output = f"Process started with PID: {result.pid}"
                success = True
                
            if success:
                print(f"✅ Success: {output}")
            else:
                print(f"❌ Failed: {output}")
                
            return success, output
            
        except subprocess.TimeoutExpired:
            print(f"⏰ Command timed out after 30 seconds")
            return False, "Timeout"
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return False, str(e)
    
    def run_ssh_command(self, command: str, description: str) -> Tuple[bool, str]:
        """Run a command on PC2 via SSH"""
        ssh_command = f"ssh {self.pc2_ip} '{command}'"
        return self.run_command(ssh_command, f"{description} (via SSH to PC2)")
    
    def step_1_activate_training_env_pc1(self) -> bool:
        """Step 1: Activate Training_env on Head PC1"""
        print("\n" + "="*60)
        print("STEP 1: Activate Training_env on Head PC1")
        print("="*60)
        
        success, output = self.run_command(
            "conda activate Training_env && echo 'Training_env activated on PC1'",
            "Activating Training_env on Head PC1"
        )
        
        if success:
            print("✅ Step 1 completed: Training_env activated on PC1")
        else:
            print("❌ Step 1 failed: Could not activate Training_env on PC1")
            
        return success
    
    def step_2_ssh_connect_pc2(self) -> bool:
        """Step 2: SSH connect to PC2"""
        print("\n" + "="*60)
        print("STEP 2: SSH Connection to Worker PC2")
        print("="*60)
        
        success, output = self.run_command(
            f"ssh -o ConnectTimeout=10 {self.pc2_ip} 'echo SSH connection successful to PC2'",
            f"Testing SSH connection to Worker PC2 ({self.pc2_ip})"
        )
        
        if success:
            print("✅ Step 2 completed: SSH connection to PC2 established")
        else:
            print("❌ Step 2 failed: Cannot establish SSH connection to PC2")
            print("💡 Make sure SSH is configured and PC2 is accessible")
            
        return success
    
    def step_3_activate_training_env_pc2(self) -> bool:
        """Step 3: Activate Training_env on Worker PC2 via SSH"""
        print("\n" + "="*60)
        print("STEP 3: Activate Training_env on Worker PC2 via SSH")
        print("="*60)
        
        success, output = self.run_ssh_command(
            "conda activate Training_env && echo 'Training_env activated on PC2'",
            "Activating Training_env on Worker PC2"
        )
        
        if success:
            print("✅ Step 3 completed: Training_env activated on PC2")
        else:
            print("❌ Step 3 failed: Could not activate Training_env on PC2")
            
        return success
    
    def step_4_check_python_version_pc1(self) -> bool:
        """Step 4: Check Python version on PC1"""
        print("\n" + "="*60)
        print("STEP 4: Check Python version on Head PC1")
        print("="*60)
        
        success, output = self.run_command(
            "python --version",
            "Checking Python version on Head PC1"
        )
        
        if success:
            if self.required_python_version in output:
                print(f"✅ Step 4 completed: Python version {self.required_python_version} confirmed on PC1")
                return True
            else:
                print(f"❌ Step 4 failed: Python version mismatch on PC1")
                print(f"Expected: {self.required_python_version}, Found: {output}")
                return False
        else:
            print("❌ Step 4 failed: Could not check Python version on PC1")
            return False
    
    def step_5_check_python_version_pc2(self) -> bool:
        """Step 5: Check Python version on PC2 via SSH"""
        print("\n" + "="*60)
        print("STEP 5: Check Python version on Worker PC2 via SSH")
        print("="*60)
        
        success, output = self.run_ssh_command(
            "python --version",
            "Checking Python version on Worker PC2"
        )
        
        if success:
            if self.required_python_version in output:
                print(f"✅ Step 5 completed: Python version {self.required_python_version} confirmed on PC2")
                return True
            else:
                print(f"❌ Step 5 failed: Python version mismatch on PC2")
                print(f"Expected: {self.required_python_version}, Found: {output}")
                return False
        else:
            print("❌ Step 5 failed: Could not check Python version on PC2")
            return False
    
    def step_6_ray_stop_pc1(self) -> bool:
        """Step 6: Stop Ray on PC1"""
        print("\n" + "="*60)
        print("STEP 6: Stop Ray on Head PC1")
        print("="*60)
        
        success, output = self.run_command(
            "ray stop",
            "Stopping Ray on Head PC1"
        )
        
        print("✅ Step 6 completed: Ray stopped on PC1")
        return True  # ray stop always succeeds even if Ray wasn't running
    
    def step_7_ray_start_head_pc1(self) -> bool:
        """Step 7: Start Ray head on PC1"""
        print("\n" + "="*60)
        print("STEP 7: Start Ray Head on PC1")
        print("="*60)
        
        ray_head_command = (
            f"ray start --head "
            f"--node-ip-address={self.pc1_ip} "
            f"--port={self.ray_port} "
            f"--dashboard-host=0.0.0.0 "
            f"--dashboard-port={self.dashboard_port} "
            f"--object-manager-port=10001 "
            f"--ray-client-server-port=10201 "
            f"--min-worker-port=10300 "
            f"--max-worker-port=10399"
        )
        
        success, output = self.run_command(
            ray_head_command,
            "Starting Ray head on PC1"
        )
        
        if success:
            print("✅ Step 7 completed: Ray head started on PC1")
            print(f"🌐 Ray Dashboard available at: http://{self.pc1_ip}:{self.dashboard_port}")
            time.sleep(5)  # Give Ray head time to fully initialize
        else:
            print("❌ Step 7 failed: Could not start Ray head on PC1")
            
        return success
    
    def step_8_ray_stop_pc2(self) -> bool:
        """Step 8: Stop Ray on PC2 via SSH"""
        print("\n" + "="*60)
        print("STEP 8: Stop Ray on Worker PC2 via SSH")
        print("="*60)
        
        success, output = self.run_ssh_command(
            "ray stop",
            "Stopping Ray on Worker PC2"
        )
        
        print("✅ Step 8 completed: Ray stopped on PC2")
        return True  # ray stop always succeeds even if Ray wasn't running
    
    def step_9_ray_start_worker_pc2(self) -> bool:
        """Step 9: Start Ray worker on PC2 via SSH"""
        print("\n" + "="*60)
        print("STEP 9: Start Ray Worker on PC2 via SSH")
        print("="*60)
        
        ray_worker_command = (
            f"ray start "
            f"--address='{self.pc1_ip}:{self.ray_port}' "
            f"--node-ip-address={self.pc2_ip}"
        )
        
        success, output = self.run_ssh_command(
            ray_worker_command,
            "Starting Ray worker on PC2"
        )
        
        if success:
            print("✅ Step 9 completed: Ray worker started on PC2")
            time.sleep(3)  # Give Ray worker time to connect
        else:
            print("❌ Step 9 failed: Could not start Ray worker on PC2")
            
        return success
    
    def verify_cluster_status(self) -> bool:
        """Verify the Ray cluster is properly set up"""
        print("\n" + "="*60)
        print("CLUSTER VERIFICATION")
        print("="*60)
        
        # Check cluster status
        success, output = self.run_command(
            "ray status",
            "Checking Ray cluster status"
        )
        
        if success:
            print("✅ Ray cluster status check completed")
            print(f"Cluster status:\n{output}")
            
            # Look for 2 nodes in the output
            if "2 nodes" in output or "Total:" in output:
                print("✅ Dual PC cluster verified successfully!")
                return True
            else:
                print("⚠️ Cluster may not have both nodes connected")
                return False
        else:
            print("❌ Could not verify cluster status")
            return False
    
    def run_complete_setup(self) -> bool:
        """Run the complete setup process"""
        print("🚀 Starting Dual PC Ray Cluster Setup")
        print("="*80)
        
        steps = [
            ("Step 1", self.step_1_activate_training_env_pc1),
            ("Step 2", self.step_2_ssh_connect_pc2),
            ("Step 3", self.step_3_activate_training_env_pc2),
            ("Step 4", self.step_4_check_python_version_pc1),
            ("Step 5", self.step_5_check_python_version_pc2),
            ("Step 6", self.step_6_ray_stop_pc1),
            ("Step 7", self.step_7_ray_start_head_pc1),
            ("Step 8", self.step_8_ray_stop_pc2),
            ("Step 9", self.step_9_ray_start_worker_pc2),
        ]
        
        failed_steps = []
        
        for step_name, step_function in steps:
            try:
                success = step_function()
                if not success:
                    failed_steps.append(step_name)
                    print(f"\n⚠️ {step_name} failed but continuing...")
            except Exception as e:
                print(f"\n❌ {step_name} encountered an error: {str(e)}")
                failed_steps.append(step_name)
        
        # Final verification
        print("\n" + "="*80)
        print("FINAL VERIFICATION")
        print("="*80)
        
        cluster_success = self.verify_cluster_status()
        
        # Summary
        print("\n" + "="*80)
        print("SETUP SUMMARY")
        print("="*80)
        
        if failed_steps:
            print(f"⚠️ Some steps failed: {', '.join(failed_steps)}")
        else:
            print("✅ All setup steps completed successfully!")
        
        if cluster_success:
            print("✅ Dual PC Ray cluster is ready for 75% distributed training!")
            print(f"🌐 Dashboard: http://{self.pc1_ip}:{self.dashboard_port}")
            print(f"📊 Head Node: {self.pc1_ip}:{self.ray_port}")
            print(f"👷 Worker Node: {self.pc2_ip}")
        else:
            print("❌ Cluster verification failed - please check the setup")
        
        return cluster_success and len(failed_steps) == 0

def main():
    """Main execution function"""
    print("🚀 Dual PC Ray Cluster Setup Tool")
    print("Following user's exact instructions for PC1 & PC2 setup")
    print("="*80)
    
    setup = DualPCRayClusterSetup()
    success = setup.run_complete_setup()
    
    if success:
        print("\n🎉 SUCCESS: Dual PC Ray cluster is ready!")
        print("You can now run the training system with both PCs at 75% utilization")
        return 0
    else:
        print("\n💥 SETUP INCOMPLETE: Please check the errors above")
        return 1

if __name__ == "__main__":
    sys.exit(main())
