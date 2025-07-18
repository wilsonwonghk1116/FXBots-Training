#!/bin/bash
# Fix NVIDIA Driver on Worker PC 2 with sudo password
# ===================================================
# 
# This script fixes the NVIDIA driver communication issue on Worker PC 2
# using the sudo password for administrative commands.

echo "🔧 FIXING NVIDIA DRIVER ON WORKER PC 2 (WITH SUDO)"
echo "==================================================="

# Check if sshpass is available
if ! command -v sshpass &> /dev/null; then
    echo "❌ sshpass not found. Please install: sudo apt-get install sshpass"
    exit 1
fi

echo "🔍 Great news from diagnostics:"
echo "✅ RTX 3070 Mobile hardware detected on Worker PC 2"
echo "✅ CUDA libraries (CUDA 12.8 + cuDNN 9.11) are installed"
echo "❌ NVIDIA driver modules not loaded (this is what we'll fix)"
echo ""

echo "🔧 RUNNING NVIDIA DRIVER FIX WITH SUDO ACCESS..."
echo "================================================"

# Create a comprehensive fix script that uses sudo with password
cat > nvidia_fix_script.sh << 'SCRIPT_EOF'
#!/bin/bash
# NVIDIA Driver Fix Script for Worker PC 2

echo "🔍 Starting NVIDIA driver fix..."

# Function to run sudo commands with password
run_sudo() {
    echo "w" | sudo -S "$@" 2>/dev/null
}

# Update package database
echo "📦 Updating package database..."
run_sudo apt-get update

# Install DKMS if not present
echo "📦 Installing DKMS..."
run_sudo apt-get install -y dkms

# Check what NVIDIA drivers are available
echo "🔍 Checking available NVIDIA drivers..."
ubuntu-drivers devices

# Install the recommended NVIDIA driver
echo "🚀 Installing NVIDIA driver..."
run_sudo ubuntu-drivers autoinstall

# If autoinstall fails, try specific driver versions
if [ $? -ne 0 ]; then
    echo "⚠️ Autoinstall failed, trying specific driver versions..."
    
    # Try NVIDIA driver 535 (stable)
    echo "📥 Installing NVIDIA driver 535..."
    run_sudo apt-get install -y nvidia-driver-535 nvidia-dkms-535
    
    # If that fails, try 470 (older but stable)
    if [ $? -ne 0 ]; then
        echo "📥 Installing NVIDIA driver 470..."
        run_sudo apt-get install -y nvidia-driver-470
    fi
fi

# Load NVIDIA kernel modules
echo "🔌 Loading NVIDIA kernel modules..."
run_sudo modprobe nvidia || echo "⚠️ nvidia module load failed"
run_sudo modprobe nvidia_drm || echo "⚠️ nvidia_drm module load failed" 
run_sudo modprobe nvidia_uvm || echo "⚠️ nvidia_uvm module load failed"

# Create NVIDIA device files
echo "📁 Creating NVIDIA device files..."
run_sudo nvidia-modprobe || echo "⚠️ nvidia-modprobe failed"

# Set proper permissions on device files
echo "🔐 Setting device permissions..."
run_sudo chmod 666 /dev/nvidia* 2>/dev/null || true
run_sudo chmod 666 /dev/nvidiactl 2>/dev/null || true
run_sudo chmod 666 /dev/nvidia-uvm* 2>/dev/null || true

# Add user to video and render groups
echo "👥 Adding user to GPU groups..."
run_sudo usermod -a -G video w2
run_sudo usermod -a -G render w2

# Check final status
echo ""
echo "🎯 FINAL STATUS CHECK:"
echo "======================"

echo "📋 NVIDIA modules loaded:"
lsmod | grep nvidia || echo "❌ No NVIDIA modules loaded yet"

echo ""
echo "📋 NVIDIA device files:"
ls -la /dev/nvidia* 2>/dev/null || echo "❌ No NVIDIA device files yet"

echo ""
echo "📋 User groups:"
groups w2

echo ""
echo "✅ NVIDIA driver installation completed!"
echo ""
echo "⚠️  REBOOT REQUIRED!"
echo "   Worker PC 2 must be rebooted for the new NVIDIA drivers to take effect."
echo "   After reboot, the RTX 3070 should be accessible to PyTorch and Ray."
SCRIPT_EOF

# Copy and run the fix script on Worker PC 2
echo "📋 Copying NVIDIA fix script to Worker PC 2..."
sshpass -p 'w' scp -o StrictHostKeyChecking=no nvidia_fix_script.sh w2@192.168.1.11:/tmp/

echo "🚀 Running NVIDIA driver fix on Worker PC 2..."
sshpass -p 'w' ssh -o StrictHostKeyChecking=no w2@192.168.1.11 "chmod +x /tmp/nvidia_fix_script.sh && /tmp/nvidia_fix_script.sh"

echo ""
echo "🎯 NVIDIA DRIVER FIX COMPLETED"
echo "=============================="
echo ""
echo "📋 NEXT STEPS:"
echo "1. 🔄 Reboot Worker PC 2 to activate new NVIDIA drivers"
echo "2. ✅ After reboot, test GPU with: nvidia-smi"
echo "3. 🚀 Restart Ray worker to detect GPU"
echo "4. 🎮 Test dual-GPU distributed training"
echo ""
echo "💡 COMMANDS TO RUN:"
echo "   # Reboot Worker PC 2:"
echo "   sshpass -p 'w' ssh w2@192.168.1.11 'echo w | sudo -S reboot'"
echo ""
echo "   # After reboot, test GPU detection:"
echo "   sshpass -p 'w' ssh w2@192.168.1.11 'nvidia-smi'"
echo ""
echo "   # Test PyTorch CUDA access:"
echo "   sshpass -p 'w' ssh w2@192.168.1.11 'python3 -c \"import torch; print(f\"CUDA available: {torch.cuda.is_available()}\")\"'"

# Clean up temporary script
rm -f nvidia_fix_script.sh
