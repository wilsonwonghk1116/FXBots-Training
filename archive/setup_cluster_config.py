#!/usr/bin/env python3
"""
Quick setup script to configure cluster_config.py with your SSH password
"""

import os

def setup_cluster_config():
    """Setup cluster configuration with SSH password"""
    
    print("🔧 CLUSTER CONFIGURATION SETUP")
    print("=" * 40)
    print("This will update your cluster_config.py file with the SSH password for PC2")
    print()
    
    # Get current password from config
    try:
        from cluster_config import PC2_SSH_PASSWORD
        if PC2_SSH_PASSWORD != "your_actual_password_here":
            print(f"✅ SSH password already configured")
            return True
    except:
        pass
    
    # Ask for password
    password = input("Enter SSH password for PC2 (w1@192.168.1.11): ").strip()
    
    if not password:
        print("❌ No password provided")
        return False
    
    # Escape quotes in password
    password = password.replace('"', '\\"')
    
    # Read current config
    config_file = "/home/w1/cursor-to-copilot-backup/TaskmasterForexBots/cluster_config.py"
    
    with open(config_file, 'r') as f:
        content = f.read()
    
    # Replace password line
    updated_content = content.replace(
        'PC2_SSH_PASSWORD = "your_actual_password_here"',
        f'PC2_SSH_PASSWORD = "{password}"'
    )
    
    # Write back
    with open(config_file, 'w') as f:
        f.write(updated_content)
    
    print("✅ SSH password configured successfully!")
    print("🚀 You can now run: python automated_cluster_training.py")
    
    return True

if __name__ == "__main__":
    setup_cluster_config()
