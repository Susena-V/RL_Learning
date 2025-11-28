import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1. PPO Actor-Critic Networks
# -------------------------------

class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, 1)
        self.tanh = nn.Tanh()
        self.log_std = nn.Parameter(torch.zeros(1))  # learnable sigma

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mu = self.tanh(self.mu(x))  # action mean ∈ [-1,1]
        std = torch.exp(self.log_std)
        return mu, std


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.value(x)


# -------------------------------
# 2. Helper functions
# -------------------------------

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    values = values + [0]

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)

    returns = [adv + v for adv, v in zip(advantages, values[:-1])]
    return advantages, returns


# -------------------------------
# 3. PPO Training Loop
# -------------------------------

env = gym.make("InvertedPendulum-v5", render_mode="human")
state_dim = env.observation_space.shape[0]

actor = Actor(state_dim)
critic = Critic(state_dim)

optimizer = optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=3e-4)

eps_clip = 0.2
epochs = 10
batch_size = 2048

episode_rewards = []

states_buffer = []
actions_buffer = []
logprobs_buffer = []
rewards_buffer = []
values_buffer = []
dones_buffer = []


def rollout(batch_size):
    states_buffer.clear()
    actions_buffer.clear()
    logprobs_buffer.clear()
    rewards_buffer.clear()
    values_buffer.clear()
    dones_buffer.clear()

    steps = 0
    obs, _ = env.reset()
    while steps < batch_size:
        state = torch.FloatTensor(obs)
        mu, std = actor(state)
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        next_obs, reward, terminated, truncated, _ = env.step([action.item()])
        done = terminated or truncated

        value = critic(state).item()

        states_buffer.append(state)
        actions_buffer.append(action)
        logprobs_buffer.append(log_prob)
        rewards_buffer.append(reward)
        values_buffer.append(value)
        dones_buffer.append(done)

        obs = next_obs
        steps += 1

        if done:
            obs, _ = env.reset()

    return np.sum(rewards_buffer)


# -------------------------------
# 4. PPO Main Loop
# -------------------------------

num_train_iters = 200

for it in range(num_train_iters):
    total_reward = rollout(batch_size)
    episode_rewards.append(total_reward)

    # Compute GAE advantages + returns
    advantages, returns = compute_gae(
        rewards_buffer, values_buffer, dones_buffer
    )

    advantages = torch.FloatTensor(advantages)
    returns = torch.FloatTensor(returns)
    old_logprobs = torch.stack(logprobs_buffer)

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Update PPO for a few epochs
    for _ in range(epochs):
        mu, std = actor(torch.stack(states_buffer))
        dist = torch.distributions.Normal(mu, std)
        new_logprobs = dist.log_prob(torch.stack(actions_buffer))
        entropy = dist.entropy().mean()

        # Ratio r(theta)
        ratio = torch.exp(new_logprobs - old_logprobs)

        # Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
        actor_loss = - torch.mean(torch.min(surr1, surr2))

        # Critic loss
        values = critic(torch.stack(states_buffer)).squeeze()
        critic_loss = torch.mean((returns - values)**2)

        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Iteration {it}  |  Reward: {total_reward}")

# -------------------------------
# 5. Plot Training Curve
# -------------------------------

plt.figure(figsize=(10,5))
plt.plot(episode_rewards)
plt.xlabel("Iteration")
plt.ylabel("Episode Reward")
plt.title("PPO on Inverted Pendulum")
plt.grid(True)
plt.show()
