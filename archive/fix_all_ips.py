#!/usr/bin/env python3
"""
IP Address Mass Update Script
=============================
Replaces all old IP addresses with new ones throughout the project
Old: 192.168.1.10 (Head PC1) → New: 192.168.1.10
Old: 192.168.1.11 (Worker PC2) → New: 192.168.1.11
"""

import os
import re
import glob
from pathlib import Path

def fix_ips_in_project():
    """Replace all old IP addresses with new ones"""
    
    # IP mappings
    ip_replacements = {
        "192.168.1.10": "192.168.1.10",  # Head PC1
        "192.168.1.11": "192.168.1.11"   # Worker PC2
    }
    
    # File patterns to update
    file_patterns = [
        "*.py",
        "*.sh", 
        "*.md",
        "*.txt",
        "*.exp",
        "*.yaml",
        "*.yml"
    ]
    
    updated_files = []
    total_replacements = 0
    
    print("🔄 Starting IP address mass update...")
    print(f"📍 Old Head PC1: 192.168.1.10 → New: 192.168.1.10")
    print(f"📍 Old Worker PC2: 192.168.1.11 → New: 192.168.1.11")
    print("-" * 60)
    
    # Get current directory
    project_dir = Path(".")
    
    # Find all files matching patterns
    all_files = []
    for pattern in file_patterns:
        all_files.extend(project_dir.glob(f"**/{pattern}"))
    
    # Process each file
    for file_path in all_files:
        if file_path.is_file():
            try:
                # Read file content
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                original_content = content
                file_replacements = 0
                
                # Replace each IP
                for old_ip, new_ip in ip_replacements.items():
                    # Count occurrences before replacement
                    count = content.count(old_ip)
                    if count > 0:
                        content = content.replace(old_ip, new_ip)
                        file_replacements += count
                        total_replacements += count
                
                # Write back if changes were made
                if content != original_content:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    updated_files.append((str(file_path), file_replacements))
                    print(f"✅ {file_path}: {file_replacements} replacements")
                    
            except Exception as e:
                print(f"❌ Error processing {file_path}: {e}")
    
    print("-" * 60)
    print(f"🎉 IP Update Complete!")
    print(f"📁 Files updated: {len(updated_files)}")
    print(f"🔢 Total replacements: {total_replacements}")
    
    if updated_files:
        print("\n📋 Updated files summary:")
        for file_path, count in updated_files:
            print(f"   • {file_path}: {count} changes")
    
    print("\n✅ All IP addresses have been updated to new network configuration!")
    return len(updated_files), total_replacements

if __name__ == "__main__":
    fix_ips_in_project() 