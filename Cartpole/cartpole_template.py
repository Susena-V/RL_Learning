import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# PARAMETERS
# -------------------------------
EPISODES = 50
MAX_STEPS = 500

# -------------------------------
# STORAGE
# -------------------------------
episode_rewards = []
step_rewards = []

# -------------------------------
# ENVIRONMENT
# -------------------------------
env = gym.make(
    "CartPole-v1",       # could be CustomCartPole-v0
    render_mode="human"
)

# -------------------------------
# MAIN LOOP
# -------------------------------
for ep in range(EPISODES):
    obs, _ = env.reset()
    total_reward = 0

    for step in range(MAX_STEPS):
        # -----------------------
        # ACTION SELECTION
        # -----------------------
        # placeholder: continuous action in [-1,1] or discrete 0/1
        # e.g., simple proportional controller on pole angle
        angle = obs[2]             # pole angle
        action = 0 if angle < 0 else 1  # move left if negative, right if positive
        # OR: action = np.random.choice([0, 1]) for random

        # -----------------------
        # STEP
        # -----------------------
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # -----------------------
        # REWARD TRACKING
        # -----------------------
        total_reward += reward
        step_rewards.append(reward)
        print(info)

        if done:
            break

    episode_rewards.append(total_reward)
    print(f"Episode {ep} | Reward: {total_reward}")

env.close()

# -------------------------------
# PLOTTING
# -------------------------------
plt.figure(figsize=(10,5))
plt.plot(episode_rewards, label="Episode Reward")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("CartPole / Continuous Skeleton")
plt.grid(True)
plt.show()
