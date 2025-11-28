import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("InvertedPendulum-v5", render_mode="human")

model = PPO.load("ppo_inverted_pendulum")

obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=False)
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()
