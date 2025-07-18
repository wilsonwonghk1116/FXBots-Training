#!/usr/bin/env python3
"""
Ray Cluster Connectivity Test
Tests Ray cluster connectivity between PC1 (head) and PC2 (worker)
"""

import socket
import time
import subprocess
import sys
from typing import List, Tuple

def get_local_ip() -> str:
    """Get local IP address"""
    try:
        # Connect to a remote address to determine local IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except:
        return "127.0.0.1"

def test_port_connectivity(host: str, port: int, timeout: float = 5.0) -> bool:
    """Test if a port is reachable"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            return result == 0
    except Exception:
        return False

def test_ray_ports(host: str) -> List[Tuple[int, str, bool]]:
    """Test all Ray cluster ports"""
    ray_ports = [
        (8265, "Ray head port"),
        (8076, "Node manager"), 
        (8077, "Object manager"),
        (10001, "Ray client server"),
        (8266, "Dashboard")
    ]
    
    results = []
    for port, description in ray_ports:
        print(f"Testing {host}:{port} ({description})...", end=" ")
        is_open = test_port_connectivity(host, port)
        status = "✅ OPEN" if is_open else "❌ CLOSED"
        print(status)
        results.append((port, description, is_open))
    
    return results

def check_ray_cluster_status():
    """Check Ray cluster status"""
    try:
        import ray
        
        print("\n🔍 Checking Ray cluster status...")
        
        # Try to connect to existing cluster
        try:
            ray.init(address='auto', ignore_reinit_error=True)
            
            # Get cluster info
            nodes = ray.nodes()
            resources = ray.cluster_resources()
            
            print(f"✅ Ray cluster connected")
            print(f"📊 Nodes: {len(nodes)}")
            print(f"📊 Resources: {resources}")
            
            # Check for worker nodes
            worker_nodes = [n for n in nodes if not n.get('is_head_node', False)]
            if worker_nodes:
                print(f"🖥️  Worker nodes: {len(worker_nodes)} (PC2 connected)")
            else:
                print("⚠️  No worker nodes found (PC2 not connected)")
            
            ray.shutdown()
            return True
            
        except Exception as e:
            print(f"❌ Ray cluster connection failed: {e}")
            return False
            
    except ImportError:
        print("❌ Ray not installed. Install with: pip install ray[default]")
        return False

def main():
    print("🧪 RAY CLUSTER CONNECTIVITY TEST")
    print("=================================")
    print()
    
    local_ip = get_local_ip()
    print(f"📍 Local IP: {local_ip}")
    print()
    
    # Test local ports (head node)
    print("🖥️  Testing PC1 (Head Node) ports:")
    local_results = test_ray_ports("localhost")
    
    # Check how many ports are open
    open_ports = sum(1 for _, _, is_open in local_results if is_open)
    total_ports = len(local_results)
    
    print(f"\n📊 PC1 Status: {open_ports}/{total_ports} ports open")
    
    if open_ports == 0:
        print("\n❌ No Ray ports are open on PC1!")
        print("   Please start Ray head node first:")
        print("   ./launch_fixed_training_75_percent.sh")
        return
    
    # Check Ray cluster status
    check_ray_cluster_status()
    
    print(f"\n🔗 PC2 CONNECTION INSTRUCTIONS:")
    print(f"   On PC2, run: ray start --address='{local_ip}:8265'")
    print()
    
    # Ask if user wants to test PC2 connection
    print("💡 TROUBLESHOOTING TIPS:")
    print("   1. Ensure firewall allows Ray ports:")
    print("      sudo ./setup_ray_firewall.sh")
    print("   2. Test basic connectivity from PC2:")
    print(f"      ping {local_ip}")
    print(f"      telnet {local_ip} 8265")
    print("   3. If still failing, check Ray logs:")
    print("      ray start --address='{local_ip}:8265' --verbose")
    print()

if __name__ == "__main__":
    main()
