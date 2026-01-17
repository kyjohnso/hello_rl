#!/bin/bash
# Fix NVIDIA driver on Ubuntu 24.04 after kernel update

set -e

echo "======================================"
echo "Fixing NVIDIA Driver on thelio"
echo "======================================"
echo ""

# Check if dkms is installed
if ! command -v dkms &> /dev/null; then
    echo "Installing dkms..."
    sudo apt update
    sudo apt install -y dkms
fi

echo "Current kernel: $(uname -r)"
echo "Installed NVIDIA driver: $(dpkg -l | grep nvidia-driver | awk '{print $2, $3}')"
echo ""

# Reinstall NVIDIA driver to rebuild for current kernel
echo "Reinstalling NVIDIA driver modules..."
sudo apt install --reinstall nvidia-dkms-550

echo ""
echo "Loading NVIDIA kernel module..."
sudo modprobe nvidia

echo ""
echo "Testing NVIDIA driver..."
nvidia-smi

echo ""
echo "======================================"
echo "NVIDIA driver fix complete!"
echo "======================================"
