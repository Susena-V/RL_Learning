import gymnasium as gym
import numpy as np
import custom_cartpole

import matplotlib.pyplot as plt

NUM_BINS = 10

# Discretization ranges
cart_pos_space   = np.linspace(-4.8, 4.8, NUM_BINS)
cart_vel_space   = np.linspace(-3.0, 3.0, NUM_BINS)
pole_angle_space = np.linspace(-0.418, 0.418, NUM_BINS)
pole_vel_space   = np.linspace(-3.5, 3.5, NUM_BINS)

STATE_BINS = [cart_pos_space, cart_vel_space, pole_angle_space, pole_vel_space]

def discretize(obs):
    state = []
    for value, bins in zip(obs, STATE_BINS):
        idx = np.digitize(value, bins) - 1
        idx = np.clip(idx, 0, NUM_BINS - 1)
        state.append(idx)
    return tuple(state)

def state_to_index(state):
    i, j, k, l = state
    return ((i * NUM_BINS + j) * NUM_BINS + k) * NUM_BINS + l

# Epsilon-greedy policy
def epsilon_greedy_action(state_idx, epsilon, Q):
    if np.random.rand() < epsilon:
        return np.random.randint(2)
    return np.argmax(Q[state_idx])

# SARSA update
def sarsa_update(Q, s, a, r, s_next, a_next, alpha, gamma):
    predict = Q[s, a]
    target  = r + gamma * Q[s_next, a_next]
    Q[s, a] += alpha * (target - predict)
    


def plot_all_rewards(episode_rewards, step_rewards):
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))

    axs[0].plot(episode_rewards)
    axs[0].set_title("Episode Rewards")
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Total Reward")
    axs[0].grid(True)

    axs[1].plot(step_rewards, alpha=0.4)
    axs[1].set_title("Per-Step Rewards")
    axs[1].set_xlabel("Step")
    axs[1].set_ylabel("Reward")
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()
    
    
# ---------------------------------------------------
# MAIN SARSA LOOP
# ---------------------------------------------------
def run_sarsa():
    env = gym.make(
    "CustomCartPole-v0",
    gravity=30.0,       # double gravity → harder
    cart_mass=2.0,      # heavier cart
    pole_mass=0.2,      # heavier pole
    pole_length=1.0,    # longer pole
    force_mag=5.0,      # weaker force
    render_mode="human"
)

    num_states = NUM_BINS**4
    Q = np.zeros((num_states, 2))

    episodes = 200
    alpha = 0.1
    gamma = 0.99
    epsilon = 0.3
    
    step_rewards = []   # reward at each timestep across the whole run
    episode_rewards = []  # reward per episode


    for ep in range(episodes):
        obs, _ = env.reset()
        s_tuple = discretize(obs)
        s_idx = state_to_index(s_tuple)

        # choose initial action
        a = epsilon_greedy_action(s_idx, epsilon, Q)

        done = False
        total_reward = 0

        while not done:
            next_obs, reward, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            total_reward += reward

            s_next_tuple = discretize(next_obs)
            s_next_idx = state_to_index(s_next_tuple)

            # choose next action (SARSA!)
            a_next = epsilon_greedy_action(s_next_idx, epsilon, Q)

            # update rule
            sarsa_update(Q, s_idx, a, reward, s_next_idx, a_next, alpha, gamma)

            # shift forward
            s_idx = s_next_idx
            a = a_next
            
            step_rewards.append(reward)

        print(f"Episode {ep}, Reward: {total_reward}")

        episode_rewards.append(total_reward)
    
    plot_all_rewards(episode_rewards, step_rewards)
    env.close()


if __name__ == "__main__":
    run_sarsa()
