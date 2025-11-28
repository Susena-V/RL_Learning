import gymnasium as gym
import numpy as np
import custom_cartpole
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# 1. DISCRETIZATION (continuous → discrete)
# ---------------------------------------------------------

NUM_BINS = 10

cart_pos_space   = np.linspace(-4.8, 4.8, NUM_BINS)
cart_vel_space   = np.linspace(-3.0, 3.0, NUM_BINS)
pole_angle_space = np.linspace(-0.418, 0.418, NUM_BINS)
pole_vel_space   = np.linspace(-3.5, 3.5, NUM_BINS)

STATE_BINS = [
    cart_pos_space,
    cart_vel_space,
    pole_angle_space,
    pole_vel_space
]

def discretize(obs):
    """
    Convert continuous observation -> tuple of 4 discrete bins.
    """
    state = []
    for val, bins in zip(obs, STATE_BINS):
        idx = np.digitize(val, bins) - 1
        idx = max(0, min(NUM_BINS - 1, idx))
        state.append(idx)
    return tuple(state)


# Convert (i,j,k,l) to single index for Q-table
def state_to_index(state_tuple):
    i, j, k, l = state_tuple
    return ((i * NUM_BINS + j) * NUM_BINS + k) * NUM_BINS + l


# ---------------------------------------------------------
# 2. Q TABLE INITIALIZATION
# ---------------------------------------------------------

num_states = NUM_BINS ** 4
num_actions = 2       # CartPole: 0 = push left, 1 = push right

Q = np.zeros((num_states, num_actions))

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



# ---------------------------------------------------------
# 3. EPSILON GREEDY POLICY
# ---------------------------------------------------------

def epsilon_greedy_action(state_idx, epsilon):
    """
    With probability ε choose random action.
    With probability 1-ε choose argmax of Q.
    """
    if np.random.random() < epsilon:
        return np.random.randint(num_actions)
    return np.argmax(Q[state_idx])


# ---------------------------------------------------------
# 4. Q-LEARNING UPDATE
# ---------------------------------------------------------

def update_q(state_idx, action, reward, next_state_idx, alpha, gamma):
    """
    Standard Q-learning update rule:
    Q(s,a) ← Q(s,a) + α [ r + γ max_a'(Q(s',a')) − Q(s,a) ]
    """
    old_q = Q[state_idx, action]
    next_max = np.max(Q[next_state_idx])

    Q[state_idx, action] = old_q + alpha * (
        reward + gamma * next_max - old_q
    )


# ---------------------------------------------------------
# 5. TRAINING LOOP
# ---------------------------------------------------------

def run_cartpole():
    env = gym.make(
    "CustomCartPole-v0",
    gravity=30.0,       # double gravity → harder
    cart_mass=2.0,      # heavier cart
    pole_mass=0.2,      # heavier pole
    pole_length=1.0,    # longer pole
    force_mag=5.0,      # weaker force
    render_mode="human"
)

    obs, _ = env.reset()

    episodes = 200
    alpha = 0.1      # learning rate
    gamma = 0.9    # discount factor
    epsilon = 0.5
    epsilon_min = 0.05
    epsilon_decay = 0.9
    
    step_rewards = []   # reward at each timestep across the whole run
    episode_rewards = []  # reward per episode


    for ep in range(episodes):
        obs, _ = env.reset()
        state = discretize(obs)
        state_idx = state_to_index(state)

        total_reward = 0
        done = False

        while not done:
            # choose action
            action = epsilon_greedy_action(state_idx, epsilon)

            # step in environment
            new_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # discretize next state
            new_state = discretize(new_obs)
            new_state_idx = state_to_index(new_state)

            # update Q-table
            update_q(state_idx, action, reward, new_state_idx, alpha, gamma)

            state_idx = new_state_idx
            total_reward += reward
            step_rewards.append(reward)

        # decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        print(f"Episode {ep}, Reward: {total_reward}, Epsilon: {epsilon:.3f}")
        
        episode_rewards.append(total_reward)

    plot_all_rewards(episode_rewards, step_rewards)
    env.close()


if __name__ == "__main__":
    run_cartpole()
    
