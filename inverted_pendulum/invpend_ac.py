import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

# Environment
env = gym.make("InvertedPendulum-v5", render_mode="human")
state_dim = env.observation_space.shape[0]
action_dim = 1  # continuous

# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)  # mean action
        self.tanh = nn.Tanh()
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        mu = self.tanh(self.fc2(x))
        return mu

# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)  # state value
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        value = self.fc2(x)
        return value

# Networks and optimizers
actor = Actor(state_dim)
critic = Critic(state_dim)
actor_opt = optim.Adam(actor.parameters(), lr=1e-3)
critic_opt = optim.Adam(critic.parameters(), lr=1e-3)

sigma = 0.1
gamma = 0.99
num_episodes = 200

for ep in range(num_episodes):
    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        state = torch.FloatTensor(obs)
        mu = actor(state)
        dist = torch.distributions.Normal(mu, sigma)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        next_obs, reward, terminated, truncated, _ = env.step([action.item()])
        done = terminated or truncated
        total_reward += reward

        # Critic estimates
        value = critic(state)
        next_value = critic(torch.FloatTensor(next_obs))
        td_target = reward + gamma * next_value * (1 - int(done))
        advantage = td_target - value

        # Actor update
        actor_loss = -log_prob * advantage.detach()
        actor_opt.zero_grad()
        actor_loss.backward()
        actor_opt.step()

        # Critic update
        critic_loss = advantage.pow(2)
        critic_opt.zero_grad()
        critic_loss.backward()
        critic_opt.step()

        obs = next_obs

    print(f"Episode {ep} | Reward: {total_reward}")
