import os

from stable_baselines3_copy.a2c import A2C
from stable_baselines3_copy.ddpg import DDPG
from stable_baselines3_copy.dqn import DQN
from stable_baselines3_copy.ppo import PPO
from stable_baselines3_copy.sac import SAC
from stable_baselines3_copy.td3 import TD3

# Read version from file
version_file = os.path.join(os.path.dirname(__file__), "version.txt")
with open(version_file, "r") as file_handler:
    __version__ = file_handler.read().strip()
