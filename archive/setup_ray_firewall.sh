#!/bin/bash

# Ray Cluster Firewall Configuration
# Opens required ports for 2-PC Ray cluster connectivity

echo "🔥 RAY CLUSTER FIREWALL CONFIGURATION"
echo "====================================="
echo ""

# Required ports for Ray cluster
RAY_PORTS=(8265 8076 8077 10001 8266)
PORT_DESCRIPTIONS=("Ray head port" "Node manager" "Object manager" "Ray client server" "Dashboard")

echo "📋 Required Ray cluster ports:"
for i in "${!RAY_PORTS[@]}"; do
    echo "   ${RAY_PORTS[$i]} - ${PORT_DESCRIPTIONS[$i]}"
done
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "⚠️  This script needs to be run as root to configure firewall"
    echo "   Please run: sudo $0"
    exit 1
fi

# Detect firewall system
if command -v ufw &> /dev/null; then
    echo "🔧 Detected UFW firewall - configuring..."
    
    # Enable UFW if not already enabled
    ufw --force enable
    
    # Open Ray ports
    for i in "${!RAY_PORTS[@]}"; do
        port=${RAY_PORTS[$i]}
        description=${PORT_DESCRIPTIONS[$i]}
        echo "   Opening port $port ($description)..."
        ufw allow $port/tcp
    done
    
    echo "✅ UFW firewall configured for Ray cluster"
    ufw status
    
elif command -v firewall-cmd &> /dev/null; then
    echo "🔧 Detected firewalld - configuring..."
    
    # Open Ray ports
    for i in "${!RAY_PORTS[@]}"; do
        port=${RAY_PORTS[$i]}
        description=${PORT_DESCRIPTIONS[$i]}"
        echo "   Opening port $port ($description)..."
        firewall-cmd --permanent --add-port=$port/tcp
    done
    
    # Reload firewall
    firewall-cmd --reload
    
    echo "✅ firewalld configured for Ray cluster"
    firewall-cmd --list-ports
    
elif command -v iptables &> /dev/null; then
    echo "🔧 Detected iptables - configuring..."
    
    # Open Ray ports
    for port in "${RAY_PORTS[@]}"; do
        echo "   Opening port $port..."
        iptables -A INPUT -p tcp --dport $port -j ACCEPT
    done
    
    # Save iptables rules (varies by distribution)
    if command -v iptables-save &> /dev/null; then
        iptables-save > /etc/iptables/rules.v4 2>/dev/null || \
        iptables-save > /etc/sysconfig/iptables 2>/dev/null || \
        echo "⚠️  Please save iptables rules manually"
    fi
    
    echo "✅ iptables configured for Ray cluster"
    
else
    echo "❌ No supported firewall detected (ufw, firewalld, iptables)"
    echo "   Please manually open ports: ${RAY_PORTS[*]}"
fi

echo ""
echo "🧪 TESTING CONNECTIVITY:"
echo ""

# Get local IP
LOCAL_IP=$(hostname -I | awk '{print $1}')
echo "📍 PC1 (Head) IP: $LOCAL_IP"
echo ""

echo "🔍 Checking if ports are listening..."
for i in "${!RAY_PORTS[@]}"; do
    port=${RAY_PORTS[$i]}
    description=${PORT_DESCRIPTIONS[$i]}"
    
    if netstat -ln 2>/dev/null | grep -q ":$port "; then
        echo "   ✅ Port $port ($description) - LISTENING"
    else
        echo "   ❌ Port $port ($description) - NOT LISTENING"
    fi
done

echo ""
echo "💡 TO TEST FROM PC2:"
echo "   1. Test connectivity: telnet $LOCAL_IP 8265"
echo "   2. If telnet works, try: ray start --address='$LOCAL_IP:8265'"
echo ""
echo "🚀 NEXT STEPS:"
echo "   1. Run this script on PC1 (head node)"
echo "   2. Start Ray head: ./launch_fixed_training_75_percent.sh"
echo "   3. On PC2, run: ray start --address='$LOCAL_IP:8265'"
echo ""
