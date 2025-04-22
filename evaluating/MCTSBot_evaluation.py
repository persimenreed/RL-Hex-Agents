#!/usr/bin/env python3
import os
import glob
import csv
import torch
import random
import numpy as np
import pyspiel

from collections import deque
from open_spiel.python.algorithms import mcts

# ───────────── Evaluation hyperparams ─────────────
BOARD_SIZE    = 8               # override per‐board
HIDDEN_SIZE   = 128             # must match your IDDQN hidden size
SNAPSHOT_DIR  = f"./weights/iddqn/iddqn_hex_{BOARD_SIZE}_4h"
EVAL_CSV      = os.path.join(SNAPSHOT_DIR, "evaluation.csv")
SELF_GAMES    = 10              # head‑to‑head games
MCTS_GAMES    = 50              # games vs MCTSBot
MCTS_SIMS     = 500             # sims/move for MCTSBot
SEED          = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ───────────── Agent wrappers ─────────────
class EvalIDDQNAgent:
    def __init__(self, ckpt_path, obs_dim, n_actions):
        from iddqn_v8 import QNet  # adjust import to your QNet definition
        self.net = QNet(obs_dim, n_actions, HIDDEN_SIZE)
        self.net.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        self.net.eval()

    def act(self, obs, legal_actions):
        obs_v = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        q_vals = self.net(obs_v)[0].detach().numpy()
        mask   = np.full_like(q_vals, -np.inf, dtype=np.float32)
        mask[legal_actions] = 0.0
        return int(np.argmax(q_vals + mask))


class EvalMCTSBot:
    def __init__(self, game, sims):
        # use RandomRolloutEvaluator for stronger baseline, or change to other evaluator
        self.bot = mcts.MCTSBot(
            game, 1.0, sims,
            mcts.RandomRolloutEvaluator(),
            solve=True,
            child_selection_fn=mcts.SearchNode.puct_value,
            verbose=False,
            dont_return_chance_node=True
        )
    def act(self, state):
        # state is a pyspiel.State
        return self.bot.mcts_search(state).action


# ───────────── Game‐playing logic ─────────────
def play_head_to_head(game, agent0, agent1, first_player):
    """Play one game. first_player=0 means agent0 starts; =1 means agent1 starts."""
    state = game.new_initial_state()
    # if first_player=1, we let agent1 move first (nothing to swap)
    while not state.is_terminal():
        pid = state.current_player()
        if pid == 0:
            a = agent0.act(state.observation_tensor(), 
                           state.legal_actions_mask().tolist().index(1))
        else:
            a = agent1.act(state)
        state.apply_action(a)
    # returns is list [r0, r1]
    returns = state.returns()
    return returns[0] > 0  # True if player0 wins


def evaluate_snapshot(game, obs_dim, n_actions, ckpt0, ckpt1):
    # load two agents
    a0 = EvalIDDQNAgent(ckpt0, obs_dim, n_actions)
    a1 = EvalIDDQNAgent(ckpt1, obs_dim, n_actions)

    # head‑to‑head
    wins0 = 0
    for i in range(SELF_GAMES):
        first = i % 2
        if play_head_to_head(game, a0, a1, first):
            wins0 += (first == 0)  # player0 win counts if a0 was player0
        else:
            wins0 += (first == 1)  # if a1 win and a1 was player0
    # champion index: 0 if a0 won >=50%, else 1
    champ = 0 if wins0 >= (SELF_GAMES / 2) else 1
    champ_agent = a0 if champ == 0 else a1

    # vs MCTSBot
    mcts_bot = EvalMCTSBot(game, MCTS_SIMS)
    wins = 0
    for i in range(MCTS_GAMES):
        state = game.new_initial_state()
        while not state.is_terminal():
            pid = state.current_player()
            if pid == 0:
                a = champ_agent.act(
                    state.observation_tensor(), 
                    state.legal_actions_mask().tolist().index(1)
                )
            else:
                a = mcts_bot.act(state)
            state.apply_action(a)
        if state.returns()[0] > 0:
            wins += 1
    win_rate = wins / MCTS_GAMES

    return champ, wins0 / SELF_GAMES, win_rate


# ───────────── Main evaluation loop ─────────────
if __name__ == "__main__":
    game       = pyspiel.load_game(f"hex(board_size={BOARD_SIZE})")
    obs_dim    = game.observation_tensor_size()
    n_actions  = game.num_distinct_actions()

    # find all snapshot pairs sorted by timestamp
    # expects files: agent0_iddqn_{t}s.pt  and agent1_iddqn_{t}s.pt
    c0 = sorted(glob.glob(os.path.join(SNAPSHOT_DIR, "agent0_*s.pt")))
    c1 = sorted(glob.glob(os.path.join(SNAPSHOT_DIR, "agent1_*s.pt")))

    assert len(c0) == len(c1), "mismatched snapshots"

    with open(EVAL_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp_s","champion","h2h_winrate","mcts_winrate"])
        for ckpt0, ckpt1 in zip(c0, c1):
            t = int(os.path.basename(ckpt0).split("_")[-1].rstrip("s.pt"))
            champ, h2h, mcts_wr = evaluate_snapshot(
                game, obs_dim, n_actions, ckpt0, ckpt1
            )
            print(f"@{t}s → champ={champ}, self={h2h:.2f}, mcts={mcts_wr:.2f}")
            w.writerow([t, champ, round(h2h,3), round(mcts_wr,3)])
