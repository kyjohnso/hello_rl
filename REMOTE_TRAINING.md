# Remote GPU Training Guide

Train RL models on thelio (Linux + RTX 3090) and run them locally on Mac.

## Setup thelio (One-time)

### Option 1: SSH and run setup script
```bash
# From your Mac, copy setup script to thelio
scp scripts/setup_thelio.sh thelio:~/

# SSH into thelio
ssh thelio

# Run setup script
chmod +x ~/setup_thelio.sh
~/setup_thelio.sh
```

### Option 2: Manual setup
```bash
# SSH into thelio
ssh thelio

# Create project directory
mkdir -p ~/hello_rl
cd ~/hello_rl

# Create venv
python3 -m venv venv
source venv/bin/activate

# Install packages
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install stable-baselines3 gymnasium tensorboard box2d-py

# Verify GPU
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## Train on thelio

```bash
# From Mac: Copy training script to thelio
scp scripts/train_remote_gpu.py thelio:~/hello_rl/

# SSH into thelio
ssh thelio

# Run training
cd ~/hello_rl
source venv/bin/activate
python train_remote_gpu.py
```

Training will:
- Take ~10-20 minutes for 1M timesteps on RTX 3090
- Save model to `~/hello_rl_models/lunar_lander_ppo_1M.zip`
- Print evaluation results

## Download trained model to Mac

```bash
# Create models directory on Mac
mkdir -p ~/projects/hello_rl/models

# Download the trained model
scp thelio:~/hello_rl_models/lunar_lander_ppo_1M.zip ~/projects/hello_rl/models/
```

## Run model locally on Mac

```bash
# From your Mac project directory
venv/bin/python scripts/run_saved_model.py

# Or specify a custom model path
venv/bin/python scripts/run_saved_model.py ~/path/to/model.zip
```

This will:
- Load the GPU-trained model (automatically converts to CPU)
- Test for 10 episodes and print statistics
- Show visual demonstration

## Monitoring training remotely (optional)

If you want to monitor training in real-time:

```bash
# On thelio, run training with tensorboard
cd ~/hello_rl
source venv/bin/activate
tensorboard --logdir=./tb_logs --port=6006 &
python train_remote_gpu.py

# On Mac, create SSH tunnel
ssh -L 6006:localhost:6006 thelio

# Open browser to http://localhost:6006
```

## Tips

- **Longer training**: Edit `total_timesteps=1_000_000` in train_remote_gpu.py
- **Different algorithms**: Replace `PPO` with `A2C`, `DQN`, `SAC`, etc.
- **Different environments**: Replace `"LunarLander-v3"` with any Gymnasium environment
- **Save intermediate checkpoints**: Add `model.save()` calls during training
