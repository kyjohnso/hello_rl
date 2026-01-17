#!/bin/bash
# Setup script for thelio (Linux + RTX 3090)
# Run this on thelio to set up the Python environment

set -e  # Exit on error

echo "======================================"
echo "Setting up hello_rl on thelio"
echo "======================================"

# Create project directory
mkdir -p ~/hello_rl
cd ~/hello_rl

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv

# Activate venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA 11.8 (adjust version based on your CUDA)
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install RL packages
echo "Installing RL packages..."
pip install stable-baselines3 gymnasium tensorboard

# Install Box2D for LunarLander
echo "Installing Box2D for LunarLander..."
pip install box2d-py

# Verify CUDA is working
echo ""
echo "======================================"
echo "Verifying GPU setup..."
echo "======================================"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "======================================"
echo "Setup complete!"
echo "======================================"
echo ""
echo "To copy the training script from your Mac:"
echo "  scp ~/projects/hello_rl/scripts/train_remote_gpu.py thelio:~/hello_rl/"
echo ""
echo "To run training:"
echo "  cd ~/hello_rl"
echo "  source venv/bin/activate"
echo "  python train_remote_gpu.py"
