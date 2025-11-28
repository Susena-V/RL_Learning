import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# PARAMETERS
# -------------------------------
EPISODES = 100
MAX_STEPS = 500

# -------------------------------
# STORAGE
# -------------------------------
episode_rewards = []
step_rewards = []

# -------------------------------
# ENVIRONMENT
# -------------------------------
env = gym.make("InvertedPendulum-v5", render_mode="human")

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
        # Placeholder controller
        # Simple: push in direction of pole angle
        pole_angle = obs[2]  # InvertedPendulum observation: [cart pos, cart vel, pole angle, pole vel]
        action = np.array([1.0]) if pole_angle > 0 else np.array([-1.0])
        # OR random: action = np.random.uniform(-3,3,size=(1,))

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

        if done:
            break

    episode_rewards.append(total_reward)
    print(f"Episode {ep} | Reward: {total_reward}")

env.close()

# -------------------------------
# PLOTTING
# -------------------------------
plt.figure(figsize=(10,5))
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Inverted Pendulum - Skeleton Run")
plt.grid(True)
plt.show()
