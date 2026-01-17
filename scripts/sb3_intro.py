"""
Stable-Baselines3 Introduction
Training an A2C agent on CartPole
"""

import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import A2C

def main():
    # Create environment without rendering for training
    print("Creating environment and training model...")
    env = gym.make("CartPole-v1")

    # Train the model
    model = A2C("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10_000)

    print("\nTraining complete!")

    # Test the trained agent
    print("\nTesting trained agent...")
    test_env = gym.make("CartPole-v1", render_mode="rgb_array")
    obs, _ = test_env.reset()

    episode_count = 0
    episode_steps = 0
    total_reward = 0
    episode_rewards = []

    for i in range(50000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        episode_steps += 1
        total_reward += reward

        if terminated or truncated:
            print(f"Episode {episode_count+1} finished after {episode_steps} steps, reward: {total_reward}")
            episode_rewards.append(total_reward)
            obs, _ = test_env.reset()
            episode_count += 1
            episode_steps = 0
            total_reward = 0

    # Print summary
    print(f"\nCompleted {episode_count} episodes")
    if episode_rewards:
        print(f"Average episode reward: {sum(episode_rewards)/len(episode_rewards):.2f}")
        print(f"Best episode reward: {max(episode_rewards):.2f}")

    test_env.close()
    env.close()

    # Final visual demonstration
    print("\n" + "="*50)
    print("Running visual demonstration...")
    print("Close the window to exit")
    print("="*50 + "\n")

    render_env = gym.make("CartPole-v1", render_mode="human")
    obs, _ = render_env.reset()

    for _ in range(5):  # Run 5 episodes with rendering
        episode_steps = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = render_env.step(action)
            episode_steps += 1

            if terminated or truncated:
                print(f"Visual episode finished after {episode_steps} steps")
                obs, _ = render_env.reset()
                break

    render_env.close()

if __name__ == "__main__":
    main()
