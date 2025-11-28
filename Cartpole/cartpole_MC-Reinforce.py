import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import custom_cartpole



# -------------------------------
# ENVIRONMENT
# -------------------------------
env = gym.make(
    "CustomCartPole-v0",
    gravity=30.0,       # double gravity → harder
    cart_mass=2.0,      # heavier cart
    pole_mass=0.2,      # heavier pole
    pole_length=1.0,    # longer pole
    force_mag=5.0,      # weaker force
    render_mode="human"
)

# -------------------------------
# REINFORCE
# -------------------------------

# Policy Network

class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4,5)
        self.fc2 = nn.Linear(5,2)
        self.fc3 = nn.Linear(2,1)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))
        return x
    
    def sample_action(self, state):
        state = torch.FloatTensor(state)
        action = self.forward(state)
        return action.item()


# MonteCarlo Reinforce Returns

def compute_returns(rewards, gamma = 0.9):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0,G)
    return returns

# -------------------------------
# MAIN LOOP
# -------------------------------

policy = PolicyNetwork()
optimizer = optim.Adam(policy.parameters(), lr = 0.1)
gamma = 0.9
episodes = 200
episode_rewards = []
step_rewards = []



for ep in range(episodes):
    obs, _ = env.reset()
    states, actions, rewards = [], [], []
    done = False

    while not done:
        
        # Action Selection
        action = policy.sample_action(obs)  
        
        action = 0 if action<0 else 1   

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        states.append(obs)
        actions.append(action)
        rewards.append(reward)

        obs = next_obs

    total_reward = sum(rewards)
    
    episode_rewards.append(total_reward)
    returns = compute_returns(rewards, gamma)
    
    states = torch.FloatTensor(states)
    actions = torch.FloatTensor(actions)
    returns = torch.FloatTensor(returns)
    
    optimizer.zero_grad()
    
    for s, a, G in zip(states, actions, returns):
        probs = policy.forward(s)
        dist = torch.distributions.Bernoulli((probs+1)/2)
        log_prob = dist.log_prob(torch.tensor(a))
        loss = -log_prob * G
        loss.backward()
        
    optimizer.step()
    

    print(f"Episode {ep} | Reward: {total_reward}")

env.close()

# -------------------------------
# PLOTTING
# -------------------------------
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("REINFORCE 4-5-2-1 Network on CartPole")
plt.grid(True)
plt.show()
