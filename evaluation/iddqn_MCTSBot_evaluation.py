#!/usr/bin/env python3
import os
import glob
import csv
import torch
import sys
import random
import numpy as np
import pyspiel

# allow import of iddqn_v1.py
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from iddqn.iddqn_v1 import QNet
from open_spiel.python.algorithms import mcts

# ───────────── Evaluation hyperparams ─────────────
BOARD_SIZE  = 5
HIDDEN_SIZE = 128   # must match iddqn_v1’s hidden size for BOARD_SIZE
MCTS_GAMES  = 100
MCTS_SIMS   = {5: 2, 8: 500, 11: 1000}[BOARD_SIZE]
#SEED        = 42

#random.seed(SEED)
#np.random.seed(SEED)
#torch.manual_seed(SEED)

SNAPSHOT_DIR = os.path.join(ROOT, "weights", "iddqn", f"iddqn_hex_{BOARD_SIZE}_4h")
EVAL_CSV     = os.path.join(ROOT, "evaluation", f"iddqn_eval_{BOARD_SIZE}.csv")

class EvalIDDQNAgent:
    def __init__(self, ckpt_path, obs_dim, n_actions):
        self.net = QNet(obs_dim, n_actions, HIDDEN_SIZE)
        self.net.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        self.net.eval()

    def act(self, obs, legal_actions):
        obs_v   = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        q_vals  = self.net(obs_v)[0].detach().numpy()
        mask    = np.full_like(q_vals, -np.inf, dtype=np.float32)
        mask[legal_actions] = 0.0
        return int(np.argmax(q_vals + mask))


class EvalMCTSBot:
    def __init__(self, game, sims):
        self.bot = mcts.MCTSBot(
            game, 1.0, sims,
            mcts.RandomRolloutEvaluator(),
            solve=True,
            child_selection_fn=mcts.SearchNode.puct_value,
            verbose=False,
            dont_return_chance_node=True
        )
    def act(self, state):
        root = self.bot.mcts_search(state)
        return root.best_child().action


def evaluate_against_mcts(game, obs_dim, n_actions, ckpt_path):
    agent    = EvalIDDQNAgent(ckpt_path, obs_dim, n_actions)
    mcts_bot = EvalMCTSBot(game, MCTS_SIMS)
    wins = 0
    for i in range(MCTS_GAMES):
        state = game.new_initial_state()
        swap  = (i >= MCTS_GAMES // 2)
        while not state.is_terminal():
            pid   = state.current_player()
            legal = list(state.legal_actions())
            if (pid == 0 and not swap) or (pid == 1 and swap):
                move = agent.act(state.observation_tensor(), legal)
            else:
                move = mcts_bot.act(state)
            state.apply_action(move)
        winner = state.returns()[0] if not swap else state.returns()[1]
        if winner > 0:
            wins += 1
    return wins / MCTS_GAMES


if __name__ == "__main__":
    assert os.path.isdir(SNAPSHOT_DIR), f"No snapshots in {SNAPSHOT_DIR}"
    game      = pyspiel.load_game(f"hex(board_size={BOARD_SIZE})")
    obs_dim   = game.observation_tensor_size()
    n_actions = game.num_distinct_actions()

    paths0 = sorted(glob.glob(os.path.join(SNAPSHOT_DIR, "agent0_*s.pt")))
    paths1 = sorted(glob.glob(os.path.join(SNAPSHOT_DIR, "agent1_*s.pt")))

    pairs = []
    for p0 in paths0:
        t  = int(os.path.basename(p0).split("_")[-1].replace("s.pt",""))
        p1 = p0.replace("agent0_", "agent1_")
        if os.path.exists(p1):
            pairs.append((t, p0, p1))
    pairs.sort(key=lambda x: x[0])

    os.makedirs(os.path.dirname(EVAL_CSV), exist_ok=True)
    with open(EVAL_CSV, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["timestamp_s", "agent0_mcts_win", "agent1_mcts_win"])
        for t, ckpt0, ckpt1 in pairs:
            r0 = evaluate_against_mcts(game, obs_dim, n_actions, ckpt0)
            r1 = evaluate_against_mcts(game, obs_dim, n_actions, ckpt1)
            print(f"@{t}s → agent0: {r0:.2f}, agent1: {r1:.2f}")
            writer.writerow([t, f"{r0:.3f}", f"{r1:.3f}"])
    print(f"Wrote evaluation to {EVAL_CSV}")
