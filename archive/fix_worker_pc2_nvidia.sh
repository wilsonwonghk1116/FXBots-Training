#!/bin/bash
# Fix NVIDIA Driver on Worker PC 2
# =================================
# 
# This script fixes the NVIDIA driver communication issue on Worker PC 2
# that is preventing GPU access in Ray distributed training.

echo "🔧 FIXING NVIDIA DRIVER ON WORKER PC 2"
echo "======================================="

# Check if sshpass is available
if ! command -v sshpass &> /dev/null; then
    echo "❌ sshpass not found. Please install: sudo apt-get install sshpass"
    exit 1
fi

echo "🔍 Diagnosing NVIDIA driver status on Worker PC 2..."

# Check current driver status
sshpass -p 'w' ssh -o StrictHostKeyChecking=no w2@192.168.1.11 << 'EOF'
echo "🔍 Current NVIDIA driver status:"
echo "--------------------------------"

# Check if NVIDIA driver is loaded
echo "📋 Checking loaded NVIDIA modules:"
lsmod | grep nvidia || echo "❌ No NVIDIA modules loaded"

# Check NVIDIA driver version
echo ""
echo "📋 Checking NVIDIA driver package:"
dpkg -l | grep nvidia-driver || echo "❌ No NVIDIA driver package found"

# Check for NVIDIA hardware
echo ""
echo "📋 Checking NVIDIA hardware:"
lspci | grep -i nvidia || echo "❌ No NVIDIA hardware detected"

# Check kernel modules blacklist
echo ""
echo "📋 Checking kernel module blacklist:"
if [ -f /etc/modprobe.d/blacklist-nvidia.conf ]; then
    echo "⚠️ NVIDIA blacklist found:"
    cat /etc/modprobe.d/blacklist-nvidia.conf
else
    echo "✅ No NVIDIA blacklist found"
fi

# Check dkms status
echo ""
echo "📋 Checking DKMS status:"
dkms status | grep nvidia || echo "❌ No NVIDIA DKMS modules"

echo ""
echo "🎯 Diagnosis complete. Starting driver fix..."
EOF

echo ""
echo "🔧 ATTEMPTING NVIDIA DRIVER FIX..."
echo "=================================="

# Fix NVIDIA driver on Worker PC 2
sshpass -p 'w' ssh -o StrictHostKeyChecking=no w2@192.168.1.11 << 'EOF'

# Update package database
echo "📦 Updating package database..."
sudo apt-get update

# Check for existing NVIDIA installations
echo ""
echo "🔍 Checking existing NVIDIA installations..."
dpkg -l | grep -i nvidia

# Purge existing NVIDIA installations that might be corrupted
echo ""
echo "🧹 Cleaning up existing NVIDIA installations..."
sudo apt-get purge -y 'nvidia-*' || true
sudo apt-get autoremove -y || true
sudo apt-get autoclean || true

# Remove any NVIDIA configuration files
echo ""
echo "🧹 Removing NVIDIA configuration files..."
sudo rm -rf /etc/X11/xorg.conf
sudo rm -rf /etc/nvidia*

# Install recommended NVIDIA driver
echo ""
echo "🚀 Installing recommended NVIDIA driver..."
sudo ubuntu-drivers devices
echo ""
echo "📥 Installing NVIDIA driver..."
sudo ubuntu-drivers autoinstall

# Alternative: Install specific driver version if autoinstall fails
echo ""
echo "📥 Installing NVIDIA driver 535 (backup method)..."
sudo apt-get install -y nvidia-driver-535 nvidia-dkms-535 || echo "⚠️ Backup installation failed"

# Load NVIDIA modules
echo ""
echo "🔌 Loading NVIDIA kernel modules..."
sudo modprobe nvidia
sudo modprobe nvidia_drm
sudo modprobe nvidia_uvm

# Create device files if they don't exist
echo ""
echo "📁 Creating NVIDIA device files..."
if [ ! -e /dev/nvidia0 ]; then
    sudo nvidia-modprobe
fi

# Set proper permissions
echo ""
echo "🔐 Setting NVIDIA device permissions..."
sudo chmod 666 /dev/nvidia* 2>/dev/null || true
sudo chmod 666 /dev/nvidiactl 2>/dev/null || true

# Add user to video group
echo ""
echo "👥 Adding user w2 to video group..."
sudo usermod -a -G video w2

echo ""
echo "✅ NVIDIA driver installation complete!"
echo ""
echo "⚠️  REBOOT REQUIRED!"
echo "     Worker PC 2 needs to be rebooted for driver changes to take effect."
echo "     After reboot, run 'nvidia-smi' to verify GPU detection."

EOF

echo ""
echo "🎯 NVIDIA DRIVER FIX COMPLETED"
echo "=============================="
echo ""
echo "📋 NEXT STEPS:"
echo "1. Reboot Worker PC 2 to load new NVIDIA drivers"
echo "2. After reboot, test with: sshpass -p 'w' ssh w2@192.168.1.11 'nvidia-smi'"
echo "3. If successful, restart Ray worker on Worker PC 2"
echo "4. Test dual-GPU distributed training"
echo ""
echo "💡 REBOOT COMMAND FOR WORKER PC 2:"
echo "   sshpass -p 'w' ssh w2@192.168.1.11 'sudo reboot'"
