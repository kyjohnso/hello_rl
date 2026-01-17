"""
Train LunarLander on GPU (for remote Linux machine with CUDA)
Saves the trained model to disk for local evaluation

Run this on thelio (Linux + RTX 3090)
"""

import gymnasium as gym
from stable_baselines3 import PPO
import torch
import os

def main():
    # Check GPU availability
    print("="*60)
    print("GPU Setup Check")
    print("="*60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("="*60 + "\n")

    # Create environment
    print("Creating LunarLander environment...")
    env = gym.make("LunarLander-v3")

    print("\nEnvironment Info:")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Train with PPO on GPU
    print("\nTraining PPO agent on GPU...")
    print("Training for 1M timesteps (will take ~10-20 minutes on RTX 3090)")
    print("="*60 + "\n")

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,  # Larger batch size for GPU
        batch_size=64,
        n_epochs=4,
        gamma=0.999,
        device="cuda",  # Force CUDA usage
    )

    # Train for 1 million timesteps
    model.learn(total_timesteps=1_000_000)

    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)

    # Save the model
    model_dir = os.path.expanduser("~/hello_rl_models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "lunar_lander_ppo_1M.zip")

    model.save(model_path)
    print(f"\nModel saved to: {model_path}")

    # Quick evaluation
    print("\nQuick evaluation (10 episodes)...")
    test_env = gym.make("LunarLander-v3")
    episode_rewards = []

    for episode in range(10):
        obs, _ = test_env.reset()
        episode_reward = 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            episode_reward += reward

            if terminated or truncated:
                episode_rewards.append(episode_reward)
                status = "LANDED" if episode_reward >= 200 else "CRASHED"
                print(f"Episode {episode+1}: {status} | Reward: {episode_reward:.2f}")
                break

    print(f"\n{'='*60}")
    print(f"Average reward: {sum(episode_rewards)/len(episode_rewards):.2f}")
    print(f"Best reward: {max(episode_rewards):.2f}")
    print(f"Successful landings (>200): {sum(1 for r in episode_rewards if r >= 200)}/10")
    print(f"{'='*60}")

    print(f"\nTo download the model to your local machine:")
    print(f"scp thelio:{model_path} ~/projects/hello_rl/models/")

    test_env.close()
    env.close()

if __name__ == "__main__":
    main()
