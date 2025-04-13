import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import pyspiel

# === Config ===
episodes = 500
board_size = 5
save_path = f"./weights/rainbow_dqn/rainbow_dqn_hex_{board_size}_{episodes}"
os.makedirs(save_path, exist_ok=True)

# === Game ===
game = pyspiel.load_game(f"hex(board_size={board_size})")
state_size = game.observation_tensor_size()
action_size = game.num_distinct_actions()

# === Replay Buffer ===
Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# === Noisy Linear ===
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        self.register_buffer("weight_epsilon", torch.zeros(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.full((out_features,), sigma_init))
        self.register_buffer("bias_epsilon", torch.zeros(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return nn.functional.linear(x, weight, bias)


# === Rainbow Q-Network ===
class RainbowQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            NoisyLinear(input_dim, 128),
            nn.ReLU(),
            NoisyLinear(128, 128),
            nn.ReLU(),
            NoisyLinear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# === Rainbow Agent ===
class RainbowDQNAgent:
    def __init__(self, state_size, action_size, gamma=0.99, lr=1e-3, batch_size=32):
        self.q_net = RainbowQNetwork(state_size, action_size)
        self.target_q_net = RainbowQNetwork(state_size, action_size)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer()
        self.gamma = gamma
        self.batch_size = batch_size
        self.steps = 0
        self.update_target_steps = 100

    def act(self, state, legal_actions):
        with torch.no_grad():
            q_values = self.q_net(torch.FloatTensor(state).unsqueeze(0)).squeeze(0)
            legal_qs = q_values[legal_actions]
            best_action = legal_actions[torch.argmax(legal_qs).item()]
            return best_action

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return

        transitions = self.buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state = torch.FloatTensor(batch.state)
        action = torch.LongTensor(batch.action).unsqueeze(1)
        reward = torch.FloatTensor(batch.reward).unsqueeze(1)
        next_state = torch.FloatTensor(batch.next_state)
        done = torch.FloatTensor(batch.done).unsqueeze(1)

        q_values = self.q_net(state).gather(1, action)

        with torch.no_grad():
            next_actions = self.q_net(next_state).argmax(1, keepdim=True)
            target_q = self.target_q_net(next_state).gather(1, next_actions)
            expected_q = reward + self.gamma * target_q * (1 - done)

        loss = nn.functional.mse_loss(q_values, expected_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.update_target_steps == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

# === Initialize 2 Agents ===
agents = [RainbowDQNAgent(state_size, action_size) for _ in range(2)]

flipped = False
# === Training Loop ===
for ep in range(1, episodes + 1):
    state = game.new_initial_state()
    total_reward = [0, 0]

    # swithing which player starts
    if ep % 2 == 0:
        agents = agents[::-1]
        flipped = True
    else:
        flipped = False


    while not state.is_terminal():
        current_player = state.current_player()
        obs = state.observation_tensor(current_player)
        legal_actions = state.legal_actions()

        action = agents[current_player].act(obs, legal_actions)
        prev_state = obs.copy()

        state.apply_action(action)

        reward = state.rewards()
        done = state.is_terminal()
        next_obs = state.observation_tensor(current_player)

        agents[current_player].buffer.push(
            prev_state, action, reward[current_player], next_obs, done
        )

        agents[current_player].q_net.net[0].reset_noise()
        agents[current_player].q_net.net[2].reset_noise()
        agents[current_player].q_net.net[4].reset_noise()

        agents[current_player].learn()

        total_reward[current_player] += reward[current_player]

    print(f"Episode {ep}: P0 reward = {total_reward[0]}, P1 reward = {total_reward[1]}")

if flipped:
    agents = agents[::-1]
# === Save models ===
for pid, agent in enumerate(agents):
    torch.save(agent.q_net.state_dict(), os.path.join(save_path, f"rainbow_dqn_p{pid}.pt"))
print(f"Rainbow DQN models saved in: {save_path}")
