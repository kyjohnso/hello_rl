"""
LunarLander-v3 Training with PPO
Land a spacecraft on the moon!

Actions:
  0: do nothing
  1: fire left orientation engine
  2: fire main engine
  3: fire right orientation engine

Goal: Land safely between the flags with minimal fuel use
"""

import gymnasium as gym
from stable_baselines3 import PPO

def main():
    # Create environment
    print("Creating LunarLander environment...")
    env = gym.make("LunarLander-v3")

    print("\nEnvironment Info:")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print("\nObservation details:")
    print("  - x, y coordinates")
    print("  - x, y velocities")
    print("  - angle, angular velocity")
    print("  - left leg contact, right leg contact")

    # Train the model with PPO (better for this environment)
    print("\nTraining PPO agent...")
    print("This may take a few minutes...\n")

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=1024,
        batch_size=64,
        n_epochs=4,
        gamma=0.999,
    )

    model.learn(total_timesteps=200_000)

    print("\nTraining complete!")

    # Test the trained agent
    print("\nTesting trained agent for 10 episodes...")
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
                status = "LANDED" if episode_reward >= 200 else "CRASHED"
                print(f"Episode {episode+1}: {status} | Reward: {episode_reward:.2f} | Steps: {episode_steps}")
                break

    # Print summary
    print(f"\n{'='*50}")
    print(f"Average reward: {sum(episode_rewards)/len(episode_rewards):.2f}")
    print(f"Best reward: {max(episode_rewards):.2f}")
    print(f"Successful landings (>200): {sum(1 for r in episode_rewards if r >= 200)}/10")
    print(f"{'='*50}")

    test_env.close()
    env.close()

    # Visual demonstration
    print("\n" + "="*50)
    print("Running visual demonstration...")
    print("Watch the lander try to land safely!")
    print("Close the window to exit")
    print("="*50 + "\n")

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
