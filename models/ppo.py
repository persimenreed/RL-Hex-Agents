#!/usr/bin/env python3
"""
PPO agent for Hex in OpenSpiel. Trains for 4 hours per board size, logs and snapshots like existing algos.
"""
import os
import time
import csv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pyspiel
from open_spiel.python import rl_environment

SEED = 42
random.seed(SEED)
numpy_seed = SEED
np.random.seed(numpy_seed)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_HOURS = 4
MAX_TRAIN_SEC = TRAIN_HOURS * 3600.0
BOARD_SIZES = [5, 8, 11]

LR = {5: 2e-4, 8: 1e-4, 11: 1e-4}
UPDATE_EPOCHS = 4
GAMMA = 0.99
GAE_LAMBDA = 0.95
EPS_CLIP = 0.2
BATCH_SIZE = 64
SNAPSHOT_INTERVAL_SEC = 600
LOG_INTERVAL_SEC = 60.0

class CNNPolicy(nn.Module):
    def __init__(self, in_channels, board_size, n_actions, hidden_dim=256):
        super().__init__()
        self.board_size = board_size
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        conv_out = 64 * board_size * board_size
        self.fc_shared = nn.Sequential(
            nn.Linear(conv_out, hidden_dim), nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, n_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        shared = self.conv(x)
        shared = self.fc_shared(shared)
        return self.policy_head(shared), self.value_head(shared)

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.values = []

    def clear(self):
        self.__init__()

class PPOAgent:
    def __init__(self, obs_dim, board_size, n_actions, lr):
        self.board_size = board_size
        self.n_actions = n_actions
        self.in_channels = obs_dim // (board_size * board_size)
        self.policy = CNNPolicy(self.in_channels, board_size, n_actions).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.buffer = RolloutBuffer()

    def select_action(self, obs, legal):
        x = torch.tensor(obs, dtype=torch.float32, device=device)
        x = x.view(1, self.in_channels, self.board_size, self.board_size)
        logits, value = self.policy(x)
        mask = torch.full_like(logits, float('-inf'))
        mask[0, legal] = 0.0
        probs = torch.softmax(logits + mask, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        self.buffer.states.append(x.detach())
        self.buffer.actions.append(action.detach())
        self.buffer.logprobs.append(dist.log_prob(action).detach())
        self.buffer.values.append(value.squeeze(1).detach())
        return int(action.item())

    def finish_episode(self):
        rewards = self.buffer.rewards
        values = [v.item() for v in self.buffer.values]
        dones = self.buffer.is_terminals
        returns = []
        advs = []
        gae = 0
        next_value = 0
        for r, val, done in zip(reversed(rewards), reversed(values), reversed(dones)):
            if done:
                next_value = 0
                gae = 0
            delta = r + GAMMA * next_value - val
            gae = delta + GAMMA * GAE_LAMBDA * gae
            advs.insert(0, gae)
            next_value = val
            returns.insert(0, gae + val)
        states = torch.cat(self.buffer.states)
        actions = torch.tensor([a.item() for a in self.buffer.actions], device=device)
        old_logprobs = torch.stack(self.buffer.logprobs)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        advantages = torch.tensor(advs, dtype=torch.float32, device=device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_size = len(rewards)
        for _ in range(UPDATE_EPOCHS):
            perm = np.random.permutation(dataset_size)
            for start in range(0, dataset_size, BATCH_SIZE):
                idx = perm[start:start+BATCH_SIZE]
                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_logprobs = old_logprobs[idx]
                batch_returns = returns[idx]
                batch_advs = advantages[idx]
                logits, vals = self.policy(batch_states)
                dist = torch.distributions.Categorical(logits=logits)
                logprobs = dist.log_prob(batch_actions)
                ratios = torch.exp(logprobs - batch_old_logprobs)
                surr1 = ratios * batch_advs
                surr2 = torch.clamp(ratios, 1 - EPS_CLIP, 1 + EPS_CLIP) * batch_advs
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.functional.mse_loss(vals.squeeze(), batch_returns)
                entropy = dist.entropy().mean()
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        self.buffer.clear()

    def save(self, root, elapsed):
        fname = os.path.join(root, f"ppo_policy_{int(elapsed)}s.pt")
        torch.save(self.policy.state_dict(), fname)

if __name__ == '__main__':
    for b in BOARD_SIZES:
        print(f"Starting PPO training with board size {b}×{b}")
        root = f"./weights/ppo/ppo_hex_{b}_{TRAIN_HOURS}h"
        os.makedirs(root, exist_ok=True)
        meta_csv = os.path.join(root, f"metadata_{b}x{b}.csv")
        with open(meta_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["elapsed_s", "episode", "avg_episode_reward"])

            game = pyspiel.load_game(f"hex(board_size={b})")
            env = rl_environment.Environment(game, include_full_state=True)
            obs_dim = game.observation_tensor_size()
            n_act = game.num_distinct_actions()
            agent = PPOAgent(obs_dim, b, n_act, LR[b])

            start = time.time()
            next_log = start + LOG_INTERVAL_SEC
            next_snap = start + SNAPSHOT_INTERVAL_SEC
            episode = 0

            while True:
                episode += 1
                now = time.time()
                elapsed = now - start
                ts = env.reset()
                ep_reward = 0
                while not ts.last():
                    p = ts.observations['current_player']
                    obs = ts.observations['info_state'][p]
                    legal = ts.observations['legal_actions'][p]
                    action = agent.select_action(obs, legal)
                    nts = env.step([action])
                    agent.buffer.rewards.append(nts.rewards[p])
                    agent.buffer.is_terminals.append(nts.last())
                    ep_reward += nts.rewards[p]
                    ts = nts
                agent.finish_episode()
                if now >= next_log:
                    print(f"[Episode {episode}] board={b}×{b} elapsed={elapsed:.1f}s / {MAX_TRAIN_SEC:.1f}s")
                    next_log += LOG_INTERVAL_SEC
                if now >= next_snap:
                    writer.writerow([int(elapsed), episode, ep_reward])
                    f.flush()
                    agent.save(root, elapsed)
                    next_snap += SNAPSHOT_INTERVAL_SEC
                if elapsed >= MAX_TRAIN_SEC:
                    print(f"Completed PPO board {b}×{b}.")
                    break
