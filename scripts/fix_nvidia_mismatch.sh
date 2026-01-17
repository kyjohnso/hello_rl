#!/bin/bash
# Fix NVIDIA driver version mismatch

set -e

echo "======================================"
echo "Fixing NVIDIA Driver Version Mismatch"
echo "======================================"
echo ""

echo "Current situation:"
echo "  DKMS module: $(dkms status | grep nvidia)"
echo "  Driver package: $(dpkg -l | grep nvidia-driver | head -1 | awk '{print $2, $3}')"
echo ""

# Option 1: Remove conflicting DKMS module and reinstall matching driver
echo "Removing conflicting NVIDIA 580 DKMS module..."
sudo dkms remove nvidia/580.95.05 --all

echo "Purging old NVIDIA packages..."
sudo apt purge -y nvidia-*

echo "Installing NVIDIA driver 550..."
sudo apt update
sudo apt install -y nvidia-driver-550

echo ""
echo "Rebooting is required to load the driver..."
echo "After reboot, run: nvidia-smi"
echo ""
echo "Would you like to reboot now? (y/n)"
read -r response
if [[ "$response" =~ ^[Yy]$ ]]; then
    echo "Rebooting in 5 seconds..."
    sleep 5
    sudo reboot
else
    echo "Please reboot manually when ready: sudo reboot"
fi
