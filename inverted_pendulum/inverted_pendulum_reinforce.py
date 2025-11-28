import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Create environment
env = gym.make("InvertedPendulum-v5", render_mode="human")

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 16)
        self.fc3 = nn.Linear(16, 1)  # output mean action
        self.log_sigma = nn.Parameter(torch.tensor(0.0))  # trainable log standard deviation
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mu = self.tanh(self.fc3(x))
        sigma = torch.exp(self.log_sigma)  # ensure positive
        return mu, sigma


def sample_action(policy, state):
    state_tensor = torch.FloatTensor(state)
    mu, sigma = policy(state_tensor)
    dist = torch.distributions.Normal(mu, sigma)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return action.item(), log_prob


def compute_returns(rewards, gamma=0.99):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns

policy = PolicyNetwork(state_dim=env.observation_space.shape[0])
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
num_episodes = 300
sigma = 0.1
save_threshold = 300
episode_rewards = []

for ep in range(num_episodes):
    obs, _ = env.reset()
    done = False
    
    states, actions, log_probs, rewards = [], [], [], []
    total_reward = 0

    while not done:
        action, log_prob = sample_action(policy, obs)
        next_obs, reward, terminated, truncated, _ = env.step([action])  # env expects array
        done = terminated or truncated

        states.append(obs)
        actions.append(action)
        log_probs.append(log_prob)
        rewards.append(reward)
        total_reward += reward
        obs = next_obs

    # Compute discounted returns
    returns = compute_returns(rewards, gamma=0.99)
    returns = torch.FloatTensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    # REINFORCE update
    loss = 0
    for G, log_prob in zip(returns, log_probs):
        loss += -log_prob * G

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # ---- SAVE MODEL WHEN REWARD > 400 ----


    if total_reward > save_threshold:
        torch.save(policy.state_dict(), "reinforce_pendulum_finding_best.pth")
        print(f"✓ Saved model at episode {ep} with reward {total_reward:.2f}")
        save_threshold = float('inf')  # prevents saving again
        save_threshold = total_reward


    episode_rewards.append(total_reward)
    print(f"Episode {ep} | Total Reward: {total_reward} | sigma: {torch.exp(policy.log_sigma).item():.4f} | loss: {loss.item():.4f} ")


plt.figure(figsize=(10,5))
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Inverted Pendulum - REINFORCE (Gaussian Policy)")
plt.grid(True)
plt.show()

