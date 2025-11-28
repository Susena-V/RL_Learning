import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# ---- Your same Policy Network ----
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 16)
        self.fc3 = nn.Linear(16, 1)
        self.tanh = nn.Tanh()

        # learned log sigma
        self.log_sigma = nn.Parameter(torch.tensor(-0.5))

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mu = self.tanh(self.fc3(x))
        return mu

# ---- Load environment ----
env = gym.make("InvertedPendulum-v5", render_mode="human")

# ---- Load best model ----
state_dim = env.observation_space.shape[0]
policy = PolicyNetwork(state_dim=state_dim)
policy.load_state_dict(torch.load("reinforce_pendulum_the_best.pth"))
#rreinforce_pendulum_the_best.pth - is the best 
policy.eval()

print("Loaded model: reinforce_pendulum_best.pth")

# ---- Simulation ----
obs, _ = env.reset()
done = False
total_reward = 0

mu_values = []
timesteps = []

t = 0

while not done:
    state_tensor = torch.FloatTensor(obs)

    # deterministic or stochastic?
    deterministic = True

    mu = policy(state_tensor)
    sigma = torch.exp(policy.log_sigma)

    if deterministic:
        action = mu.detach().numpy()
    else:
        dist = torch.distributions.Normal(mu, sigma)
        action = dist.sample().detach().numpy()

    next_obs, reward, terminated, truncated, _ = env.step([float(action)])
    done = terminated or truncated

    total_reward += reward
    obs = next_obs
    
    mu_values.append(mu.item())
    timesteps.append(t)
    
    t+=1

print(f"\nSimulation finished. Total reward = {total_reward:.2f}")
env.close()

plt.figure(figsize=(10,4))
plt.plot(timesteps, mu_values)
plt.xlabel("Timestep")
plt.ylabel("Mean action (mu)")
plt.title("Deterministic Mean Action over Time - Inverted Pendulum")
plt.grid(True)
plt.show()