#!/usr/bin/env python3
"""
Evaluate PPO policies against MCTS for Hex (5×5, 8×8, 11×11) and write separate CSVs.
"""
import os
import sys
import glob
import csv
import torch
import numpy as np
import pyspiel

# Ensure project root is on PYTHONPATH so we can import ppo_v0
SCRIPT_DIR   = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
sys.path.insert(0, PROJECT_ROOT)

from open_spiel.python.algorithms import mcts
from ppo.ppo_v0 import CNNPolicy

def evaluate_vs_mcts(game, agent, mcts_bot, pid=0):
    wins = 0
    games = 100
    for i in range(games):
        state = game.new_initial_state()
        swap  = (i >= games // 2)
        while not state.is_terminal():
            pid_cur = state.current_player()
            is_agent = (pid_cur == pid and not swap) or (pid_cur != pid and swap)
            move = agent.act(state) if is_agent else mcts_bot.act(state)
            state.apply_action(move)
        winner = state.returns()[pid] if not swap else state.returns()[1-pid]
        if winner > 0:
            wins += 1
    return wins / games

class EvalMCTSBot:
    def __init__(self, game, sims=100):
        self.bot = mcts.MCTSBot(
            game, 1.0, sims,
            mcts.RandomRolloutEvaluator(),
            solve=True,
            child_selection_fn=mcts.SearchNode.puct_value,
            verbose=False,
            dont_return_chance_node=True
        )
    def act(self, state):
        return self.bot.mcts_search(state).best_child().action

class EvalPPOAgent:
    """Wrap a trained PPO CNNPolicy for evaluation"""
    def __init__(self, ckpt_path, obs_dim, board_size, n_actions):
        self.board_size = board_size
        self.in_ch      = obs_dim // (board_size * board_size)
        self.net        = CNNPolicy(self.in_ch, board_size, n_actions)
        self.net.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        self.net.eval()

    def act(self, state):
        obs   = state.observation_tensor()
        legal = list(state.legal_actions())
        x = torch.tensor(obs, dtype=torch.float32)
        x = x.view(1, self.in_ch, self.board_size, self.board_size)
        with torch.no_grad():
            logits, _ = self.net(x)
            logits = logits.numpy()[0]
        mask = np.full_like(logits, -np.inf, dtype=np.float32)
        mask[legal] = 0.0
        return int(np.argmax(logits + mask))

if __name__ == "__main__":
    for B in [5, 8, 11]:
        game      = pyspiel.load_game(f"hex(board_size={B})")
        sims      = {5:100, 8:100, 11:100}[B]
        mcts_bot  = EvalMCTSBot(game, sims)
        obs_dim   = game.observation_tensor_size()
        n_actions = game.num_distinct_actions()

        os.makedirs("evaluation", exist_ok=True)
        out_csv = f"evaluation/eval_{B}.csv"

        with open(out_csv, "w", newline="") as fout:
            writer = csv.writer(fout)
            writer.writerow(["timestamp_s", "ppo_mcts_win"])

            snap_dir = os.path.join(
                PROJECT_ROOT, "weights", "ppo", f"ppo_hex_{B}_4h"
            )
            paths = glob.glob(os.path.join(snap_dir, "ppo_policy_*s.pt"))
            paths.sort(key=lambda p: int(os.path.basename(p).split('_')[-1].replace('s.pt','')))

            for pth in paths:
                t     = int(os.path.basename(pth).split("_")[-1].replace("s.pt",""))
                agent = EvalPPOAgent(pth, obs_dim, B, n_actions)
                r     = evaluate_vs_mcts(game, agent, mcts_bot)
                print(f"[PPO {B}×{B}] @{t}s → {r:.2f}")
                writer.writerow([t, f"{r:.3f}"])

        print(f"Wrote PPO results to {out_csv}")
