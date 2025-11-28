import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

env = gym.make("InvertedPendulum-v5")

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    batch_size=64,
    gamma=0.99,
)

model.learn(total_timesteps=200_000)

model.save("ppo_inverted_pendulum")
env.close()
