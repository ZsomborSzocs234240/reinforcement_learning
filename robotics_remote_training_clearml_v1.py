import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'
import subprocess
import sys
import argparse
import wandb
from stable_baselines3 import PPO
from clearml import Task
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import pybullet as p
from robotics_wrapper_v1 import OT2Env

# Add shimmy as a requirement
#Task.add_requirements("shimmy>=2.0")
#os.system("pip install shimmy>=2.0")

# Add pydantic and typing-extensions as requirements
#os.system("pip install pydantic>=2.10.5 typing-extensions>=4.12.2")

# Debugging Environment
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

# Install dependencies using os.system
print("Installing dependencies...")
try:
    os.system("pip install shimmy>=2.0")
    os.system("pip install pydantic>=2.10.5 typing-extensions>=4.12.2")
    print("Dependencies installed successfully.")
except Exception as e:
    print(f"Error installing dependencies: {e}")
    sys.exit(1)

# Verify installed versions
try:
    import pydantic
    import typing_extensions
    print(f"Pydantic version: {pydantic.VERSION}")
    print("Typing-extensions installed successfully.")
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    sys.exit(1)

# Ensure Direct mode for PyBullet (headless execution)
try:
    physicsClient = p.connect(p.DIRECT)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    print("PyBullet initialized in DIRECT mode.")
except Exception as e:
    print(f"Error initializing PyBullet: {e}")
    sys.exit(1)

# Initialize ClearML task
try:
    task = Task.init(
        project_name='Mentor Group K/Group 2',
        task_name='OT2 Experiment 2 234240'
    )
    task.set_base_docker('deanis/2023y2b-rl:latest')
    task.execute_remotely(queue_name="default")
    print("ClearML task initialized successfully.")
except Exception as e:
    print(f"Error initializing ClearML task: {e}")
    sys.exit(1)

# Ensure Direct mode for PyBullet (headless execution)
physicsClient = p.connect(p.DIRECT)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

# Initialize ClearML task
task = Task.init(
    project_name='Mentor Group K/Group 2',
    task_name='OT2 Experiment 2 234240'
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

# Initialize Weights & Biases (W&B)
wandb.init(
    project="RL_OT2",
    name="OT2 Experiment 2 234240",
    config={
        "learning_rate": args.learning_rate,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "total_timesteps": 1000000,
        "accuracy_threshold": 0.01  # (8.8 C or 8.8 D)
    }
)

# Initialize the custom environment
env = OT2Env(render=False, max_steps=1000)
env = Monitor(env)
env = DummyVecEnv([lambda: env])

# Define a custom callback to log metrics
class ClearMLCallback(BaseCallback):
    def __init__(self, task, verbose=0):
        super(ClearMLCallback, self).__init__(verbose)
        self.task = task

    def _on_step(self) -> bool:
        # Log the average reward
        rewards = self.locals.get('rewards', None)
        if rewards is not None:
            avg_reward = float(sum(rewards) / len(rewards)) if len(rewards) > 0 else 0
            self.task.get_logger().report_scalar(
                title='Reward',
                series='reward',
                value=avg_reward,
                iteration=self.num_timesteps
            )

        # Log loss
        if 'loss' in self.locals:
            loss = self.locals['loss']
            self.task.get_logger().report_scalar(
                title='Loss',
                series='loss',
                value=loss,
                iteration=self.num_timesteps
            )

        # Log entropy (if accessible)
        if 'entropy' in self.locals:
            entropy = self.locals['entropy']
            self.task.get_logger().report_scalar(
                title='Entropy',
                series='entropy',
                value=entropy,
                iteration=self.num_timesteps
            )

        # Log value loss (if accessible)
        if 'value_loss' in self.locals:
            value_loss = self.locals['value_loss']
            self.task.get_logger().report_scalar(
                title='Value Loss',
                series='value_loss',
                value=value_loss,
                iteration=self.num_timesteps
            )

        return True

# Initialize the callback
clearml_callback = ClearMLCallback(task)

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
model.learn(total_timesteps=1000000, callback=[clearml_callback])

# Save the trained model
model.save("ppo_ot2_model.zip")