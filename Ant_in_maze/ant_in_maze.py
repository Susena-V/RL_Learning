import gymnasium as gym
import numpy as np
import torch

env = gym.make("Ant-v5")  # Gymnasium MuJoCo Ant
obs, _ = env.reset()

