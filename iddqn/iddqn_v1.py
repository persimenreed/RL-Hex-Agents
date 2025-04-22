# iddqn_v8.py

import os
import time
import random
import csv
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pyspiel
from open_spiel.python import rl_environment

# ─────────────────── Config ────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_HOURS           = 4
MAX_TRAIN_SEC         = TRAIN_HOURS * 3600.0
BOARD_SIZES           = [5, 8, 11]

# Tunable hyper‑params per board size
LEARNING_RATES = {5: 1e-3, 8: 3e-4, 11: 1e-4}
BUFFER_SIZES   = {5: 100_000, 8: 500_000, 11: 500_000}
HIDDEN_SIZES   = {5: 128,     8: 256,     11: 256}

# Shared DQN hyper‑params
GAMMA              = 0.99
EPS_START          = 1.0
EPS_END            = 0.05
BATCH_SIZE         = 64
TARGET_UPDATE_FREQ = 1000

# Logging & snapshots
LOG_INTERVAL_SEC      = 60.0
SNAPSHOT_INTERVAL_SEC = 600   # 10 min

def compute_epsilon(elapsed):
    """Linear decay from EPS_START→EPS_END over first half of MAX_TRAIN_SEC."""
    half = MAX_TRAIN_SEC / 2.0
    if elapsed < half:
        frac = elapsed / half
        return EPS_START + frac * (EPS_END - EPS_START)
    return EPS_END

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def add(self, *transition):
        self.buffer.append(transition)
    def sample(self, n):
        batch = random.sample(self.buffer, n)
        return map(np.array, zip(*batch))
    def __len__(self):
        return len(self.buffer)

class QNet(nn.Module):
    def __init__(self, input_dim, n_actions, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )
    def forward(self, x):
        return self.net(x)

class IDDQNAgent:
    def __init__(self, obs_dim, n_actions, hidden_size, lr, buf_capacity):
        self.epsilon = EPS_START
        self.online  = QNet(obs_dim, n_actions, hidden_size).to(device)
        self.target  = QNet(obs_dim, n_actions, hidden_size).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.opt     = optim.Adam(self.online.parameters(), lr=lr)
        self.buf     = ReplayBuffer(buf_capacity)
        self.steps   = 0
        self.losses  = []

    def act(self, obs, legal_actions):
        if random.random() < self.epsilon:
            return random.choice(legal_actions)
        obs_v = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        q     = self.online(obs_v)[0].detach().cpu().numpy()
        mask  = np.full_like(q, -np.inf, dtype=np.float32)
        mask[legal_actions] = 0.0
        return int(np.argmax(q + mask))

    def train_step(self):
        if len(self.buf) < BATCH_SIZE:
            return None
        s, a, r, ns, d = self.buf.sample(BATCH_SIZE)
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
        self.losses.append(loss.item())
        if self.steps % TARGET_UPDATE_FREQ == 0:
            self.target.load_state_dict(self.online.state_dict())
        return loss.item()

# ─────────────────── Main Loop ───────────────────
for b in BOARD_SIZES:
    print(f"Starting IDDQN training with board size {b}×{b}")
    lr      = LEARNING_RATES[b]
    buf_cap = BUFFER_SIZES[b]
    hid     = HIDDEN_SIZES[b]

    root     = f"./weights/iddqn/iddqn_hex_{b}_{TRAIN_HOURS}h"
    os.makedirs(root, exist_ok=True)
    meta_csv = os.path.join(root, f"metadata{b}x{b}.csv")

    with open(meta_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["elapsed_s","episode","epsilon","steps","buffer_len","avg_td_loss"])

        game   = pyspiel.load_game(f"hex(board_size={b})")
        env    = rl_environment.Environment(game, include_full_state=True)
        obs_d  = game.observation_tensor_size()
        n_act  = game.num_distinct_actions()
        agents = [IDDQNAgent(obs_d, n_act, hid, lr, buf_cap) for _ in range(env.num_players)]

        start     = time.time()
        next_log  = start + LOG_INTERVAL_SEC
        next_snap = start + SNAPSHOT_INTERVAL_SEC
        episode   = 0

        while True:
            episode += 1
            now     = time.time()
            elapsed = now - start
            eps     = compute_epsilon(elapsed)
            for ag in agents:
                ag.epsilon = eps

            ts = env.reset()
            while not ts.last():
                p   = ts.observations["current_player"]
                act = agents[p].act(
                    ts.observations["info_state"][p],
                    ts.observations["legal_actions"][p]
                )
                nts = env.step([act])
                agents[p].buf.add(
                    ts.observations["info_state"][p],
                    act,
                    nts.rewards[p],
                    nts.observations["info_state"][p],
                    nts.last()
                )
                agents[p].train_step()
                ts = nts

            if now >= next_log:
                print(f"[Episode {episode}] board={b}x{b} elapsed={elapsed:.1f}s / {MAX_TRAIN_SEC:.1f}s  ε={eps:.2f}")
                next_log += LOG_INTERVAL_SEC

            if now >= next_snap:
                buf_len  = len(agents[0].buf)
                avg_loss = sum(agents[0].losses) / len(agents[0].losses) if agents[0].losses else None
                writer.writerow([int(elapsed), episode, eps, agents[0].steps, buf_len, avg_loss])
                f.flush()
                for ag in agents:
                    ag.losses.clear()
                for i, ag in enumerate(agents):
                    torch.save(
                        ag.online.state_dict(),
                        os.path.join(root, f"agent{i}_iddqn_{int(elapsed)}s.pt")
                    )
                next_snap += SNAPSHOT_INTERVAL_SEC

            if elapsed >= MAX_TRAIN_SEC:
                print(f"Completed IDDQN board {b}×{b}.")
                break
