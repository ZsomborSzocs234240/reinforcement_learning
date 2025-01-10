import numpy as np
from stable_baselines3 import PPO
from robotics_wrapper_v1 import OT2Env

def evaluate_model(model, env, n_episodes=10, render=False):

    success_count = 0
    total_rewards = []
    episode_lengths = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        cumulative_reward = 0
        steps = 0

        while not done:
            # Predict the next action using the trained model
            action, _ = model.predict(obs, deterministic=True)

            # Step in the environment
            obs, reward, terminated, truncated, _ = env.step(action)

            # Accumulate rewards and steps
            cumulative_reward += reward
            steps += 1

            # Render the environment if required
            if render:
                env.render()

            # Check if the episode has ended
            done = terminated or truncated

        # Track metrics
        total_rewards.append(cumulative_reward)
        episode_lengths.append(steps)

        # Success if the episode ended due to termination (goal reached)
        if terminated:
            success_count += 1

        print(f"Episode {episode + 1}/{n_episodes} | Reward: {cumulative_reward:.2f} | Steps: {steps}")

    # Calculate evaluation metrics
    success_rate = (success_count / n_episodes) * 100
    avg_cumulative_reward = np.mean(total_rewards)
    avg_episode_length = np.mean(episode_lengths)

    return success_rate, avg_cumulative_reward, avg_episode_length


if __name__ == "__main__":
    
    # Load the trained model
    model = PPO.load("ot2_trained_model")

    # Initialize the environment
    env = OT2Env(render=False, max_steps=1000)  # Change render=True for visualization

    # Evaluate the model
    print("Evaluating the model...")
    success_rate, avg_cumulative_reward, avg_episode_length = evaluate_model(model, env, n_episodes=10, render=False)

    # Print evaluation results
    print("\nEvaluation Results:")
    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Average Cumulative Reward: {avg_cumulative_reward:.2f}")
    print(f"Average Episode Length: {avg_episode_length:.2f} steps")

    # Close the environment
    env.close()
