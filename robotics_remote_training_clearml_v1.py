import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import argparse
from stable_baselines3 import PPO
from clearml import Task
import pybullet as p
from robotics_wrapper_v1 import OT2Env

# Ensure Direct mode for PyBullet (headless execution)
physicsClient = p.connect(p.DIRECT)

# Initialize ClearML task
task = Task.init(
    project_name='Mentor Group K/Group 2',
    task_name='OT2 Experiment Version 1 234240'
)

# Use Docker container for remote execution
task.set_base_docker('deanis/2023y2b-rl:latest')
task.execute_remotely(queue_name="default")

# Define hyperparameters via argparse
parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate for PPO")
parser.add_argument("--n_steps", type=int, default=2048, help="Number of steps per rollout")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs per training iteration")
args = parser.parse_args()

# Log hyperparameters to ClearML
task.connect(vars(args))

# Initialize the custom environment
env = OT2Env(render=False, max_steps=1000)

# Configure the PPO model with hyperparameters
model = PPO(
    'MlpPolicy',
    env,
    verbose=1,
    learning_rate=args.learning_rate,
    n_steps=args.n_steps,
    batch_size=args.batch_size,
    n_epochs=args.n_epochs,
)

# Train the model
try:
    print("Starting training...")
    model.learn(total_timesteps=1000000)
    print("Training completed successfully.")
except Exception as e:
    print(f"Training failed: {e}")
    raise

# Save the trained model
model_save_path = "ot2_trained_model"
model.save(model_save_path)
print(f"Model saved at {model_save_path}")

# Close the environment
env.close()

# Disconnect PyBullet
p.disconnect()