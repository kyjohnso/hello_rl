"""
Load and run a trained model saved from remote GPU training
Run this locally on Mac
"""

import gymnasium as gym
from stable_baselines3 import PPO
import os
import sys

def main():
    # Check for model path argument
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        # Default path
        model_path = os.path.expanduser("~/projects/hello_rl/models/lunar_lander_ppo_1M.zip")

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print(f"\nTo download from thelio:")
        print(f"  mkdir -p ~/projects/hello_rl/models")
        print(f"  scp thelio:~/hello_rl_models/lunar_lander_ppo_1M.zip ~/projects/hello_rl/models/")
        return

    print("="*60)
    print(f"Loading model from: {model_path}")
    print("="*60)

    # Load the model (automatically uses CPU on Mac)
    model = PPO.load(model_path)
    print("Model loaded successfully!\n")

    # Test without rendering
    print("Testing for 10 episodes...")
    test_env = gym.make("LunarLander-v3")
    episode_rewards = []

    for episode in range(10):
        obs, _ = test_env.reset()
        episode_reward = 0
        episode_steps = 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            episode_reward += reward
            episode_steps += 1

            if terminated or truncated:
                episode_rewards.append(episode_reward)
                status = "✓ LANDED" if episode_reward >= 200 else "✗ CRASHED"
                print(f"Episode {episode+1}: {status} | Reward: {episode_reward:.2f} | Steps: {episode_steps}")
                break

    print(f"\n{'='*60}")
    print(f"Average reward: {sum(episode_rewards)/len(episode_rewards):.2f}")
    print(f"Best reward: {max(episode_rewards):.2f}")
    print(f"Successful landings (>200): {sum(1 for r in episode_rewards if r >= 200)}/10")
    print(f"{'='*60}")

    test_env.close()

    # Visual demonstration
    print("\n" + "="*60)
    print("Running visual demonstration...")
    print("Close the window to exit")
    print("="*60 + "\n")

    render_env = gym.make("LunarLander-v3", render_mode="human")

    for episode in range(3):
        obs, _ = render_env.reset()
        episode_reward = 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = render_env.step(action)
            episode_reward += reward

            if terminated or truncated:
                status = "SUCCESS" if episode_reward >= 200 else "FAILED"
                print(f"Visual episode {episode+1}: {status} | Reward: {episode_reward:.2f}")
                break

    render_env.close()

if __name__ == "__main__":
    main()
