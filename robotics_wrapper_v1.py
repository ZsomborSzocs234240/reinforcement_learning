import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
from sim_class import Simulation

class OT2Env(gym.Env):
    def __init__(self, render=False, max_steps=1000):
        super(OT2Env, self).__init__()
        self.render = render
        self.max_steps = max_steps

        # Create the simulation environment
        self.sim = Simulation(num_agents=1, render=render)

        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        # keep track of the number of steps
        self.steps = 0

    def reset(self, seed: int = None):
        # being able to set a seed is required for reproducibility
        if seed is not None:
            np.random.seed(seed)

        # Reset the state of the environment to an initial state
        # set a random goal position for the agent, consisting of x, y, and z coordinates within the working area (you determined these values in the previous datalab task)
        workspace_bounds = {
            "x": (-0.2600, 0.1800),
            "y": (-0.2600, 0.1300),
            "z": (0.0800, 0.2000)
        }

        self.goal_position = np.array([
            np.random.uniform(*workspace_bounds["x"]),
            np.random.uniform(*workspace_bounds["y"]),
            np.random.uniform(*workspace_bounds["z"])
        ], dtype=np.float32)

        # Call the environment reset function
        observation = self.sim.reset(num_agents=1)

        # now we need to process the observation and extract the relevant information, the pipette position, convert it to a numpy array, and append the goal position and make sure the array is of type np.float32
        robot_id = next(iter(observation.keys()))
        pipette_position = np.array(observation[robot_id]['pipette_position'], dtype=np.float32)
        observation = np.concatenate([pipette_position, self.goal_position]).astype(np.float32)

        # Calculate the initial distance to the goal
        self.initial_distance = np.linalg.norm(pipette_position - self.goal_position)
        self.reached_halfway = False  # Track whether the halfway point is reached

        # Reset the number of steps
        self.steps = 0

        return observation, {}

    def step(self, action):
        # Execute one time step within the environment
        # since we are only controlling the pipette position, we accept 3 values for the action and need to append 0 for the drop action
        action = np.clip(action, self.action_space.low, self.action_space.high)
        action = np.append(action, 0)

        # Call the environment step function
        observation = self.sim.run([action]) # Why do we need to pass the action as a list? Think about the simulation class.

        # now we need to process the observation and extract the relevant information, the pipette position, convert it to a numpy array, and append the goal position and make sure the array is of type np.float32
        robot_id = next(iter(observation.keys()))
        pipette_position = np.array(observation[robot_id]['pipette_position'], dtype=np.float32)
        observation = np.concatenate([pipette_position, self.goal_position]).astype(np.float32)

        # Calculate the reward, this is something that you will need to experiment with to get the best results
        distance_to_goal = np.linalg.norm(pipette_position - self.goal_position)
        print(f"Distance to goal: {distance_to_goal}") # Debug

        # next we need to check if the if the task has been completed and if the episode should be terminated
        # To do this we need to calculate the distance between the pipette position and the goal position and if it is below a certain threshold, we will consider the task complete. 
        # What is a reasonable threshold? Think about the size of the pipette tip and the size of the plants.

        # Initialize the reward
        reward = 0.0

        # Distance based reward
        reward -= 10 * distance_to_goal

        # Bonus reward for reaching 50% of the initial distance
        if not self.reached_halfway and distance_to_goal < 0.5 * self.initial_distance:
            reward += 20.0  # Small bonus reward
            self.reached_halfway = True  # Ensure the reward is given only once

        # Fixed threshold for success
        accuracy_threshold = 0.01  # Set to 0.01 for 8.8 C, or 0.001 for 8.8 D
        terminated = distance_to_goal < accuracy_threshold

        # Add truncation condition based on step limit
        truncated = self.steps >= self.max_steps

        # Reward for successful completion
        if terminated:
            reward += 100.0  # Large bonus reward for success
            print("Task successfully completed!")

        # Print debug information
        print(f"Reward: {reward}")
        print(f"Terminated: {terminated}")
        print(f"Truncated: {truncated}")

        # Increment the step counter
        self.steps += 1

        # Return updated observation, reward, termination flag, truncation flag, and additional info
        return observation, reward, terminated, truncated, {}