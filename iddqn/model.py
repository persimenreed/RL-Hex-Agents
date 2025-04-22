# iddqn/model.py
import random
import torch
import torch.nn as nn
import numpy as np
from collections import deque

GAMMA = 0.99
BATCH_SIZE = 64
TARGET_UPDATE_FREQ = 1000
EPS_START = 1.0
EPS_END   = 0.05

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def add(self, *t):   self.buffer.append(t)
    def sample(self, n):
        batch = random.sample(self.buffer, n)
        return map(np.array, zip(*batch))
    def __len__(self): return len(self.buffer)

class QNet(nn.Module):
    def __init__(self, input_dim, n_actions, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )
    def forward(self, x): return self.net(x)

class IDDQNAgent:
    def __init__(self, obs_dim, n_actions, hidden_size, lr, buf_capacity):
        self.epsilon    = EPS_START
        self.online     = QNet(obs_dim, n_actions, hidden_size)
        self.target     = QNet(obs_dim, n_actions, hidden_size)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.opt        = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.buf        = ReplayBuffer(buf_capacity)
        self.steps      = 0
        self.loss_sum   = 0.0
        self.loss_count = 0

    def act(self, obs, legal_actions):
        if random.random() < self.epsilon:
            return random.choice(legal_actions)
        obs_v = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        q     = self.online(obs_v)[0].detach().numpy()
        mask  = np.full_like(q, -np.inf, dtype=np.float32)
        mask[legal_actions] = 0.0
        return int(np.argmax(q + mask))

    def train_step(self):
        if len(self.buf) < BATCH_SIZE:
            return
        s_v  = torch.tensor(s, dtype=torch.float32, device=device)
        ns_v = torch.tensor(ns, dtype=torch.float32, device=device)
        a_v  = torch.tensor(a, dtype=torch.int64,   device=device)
        r_v  = torch.tensor(r, dtype=torch.float32, device=device)
        d_v  = torch.tensor(d, dtype=torch.float32, device=device)

        with torch.no_grad():
            na = self.online(ns_v).argmax(dim=1, keepdim=True)
            nq = self.target(ns_v).gather(1, na).squeeze(1)
            td = r_v + GAMMA * (1 - d_v) * nq

        q_vals = self.online(s_v).gather(1, a_v.unsqueeze(1)).squeeze(1)
        loss   = nn.functional.mse_loss(q_vals, td)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self.steps += 1
        l = loss.item()
        self.loss_sum   += l
        self.loss_count += 1
        if self.steps % TARGET_UPDATE_FREQ == 0:
            self.target.load_state_dict(self.online.state_dict())
        return l
