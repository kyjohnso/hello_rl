#!/bin/bash
# Install swig, Python dev headers, and Box2D on thelio

set -e

echo "Installing build dependencies..."
sudo apt update
sudo apt install -y swig python3-dev python3.12-dev build-essential

echo "Installing Box2D in venv..."
cd ~/projects/hello_rl
source venv/bin/activate
pip install box2d-py

echo ""
echo "Box2D installed successfully!"
python -c "import Box2D; print('Box2D version:', Box2D.__version__)"
