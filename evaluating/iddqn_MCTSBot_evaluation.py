#!/usr/bin/env python3
import os
import glob
import csv
import torch
import random
import numpy as np
import pyspiel
from RL-Hex-Agents.iddqn.iddqn_v1 import *

from open_spiel.python.algorithms import mcts

# ───────────── Evaluation hyperparams ─────────────
BOARD_SIZE    = 5               # change to 5 or 11 for other boards
HIDDEN_SIZE   = 128             # must match your IDDQN hidden size
SELF_GAMES    = 10              # head‑to‑head games
MCTS_GAMES    = 50              # games vs MCTSBot
MCTS_SIMS     = 500             # sims/move for MCTSBot
SEED          = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ──────────── Paths ────────────
project_root   = os.path.dirname(os.path.dirname(__file__))
SNAPSHOT_DIR   = os.path.join(
    project_root, "weights", "iddqn", f"iddqn_hex_{BOARD_SIZE}_4h"
)
EVAL_DIR       = os.path.join(project_root, "evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)
EVAL_CSV       = os.path.join(EVAL_DIR, f"iddqn_evaluation_{BOARD_SIZE}x{BOARD_SIZE}.csv")

# ───────────── Agent wrappers ─────────────
class EvalIDDQNAgent:
    def __init__(self, ckpt_path, obs_dim, n_actions):
        # import your QNet definition
        self.net = QNet(obs_dim, n_actions, HIDDEN_SIZE)
        self.net.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        self.net.eval()

    def act(self, obs, legal_actions):
        obs_v   = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        q_vals  = self.net(obs_v)[0].detach().numpy()
        # mask out illegal moves
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
        return self.bot.mcts_search(state).action

# ───────────── Game logic ─────────────
def play_match(game, agent0, agent1, first_player):
    state = game.new_initial_state()
    while not state.is_terminal():
        pid = state.current_player()
        if pid == 0:
            # agent0 plays player 0
            legal = list(state.legal_actions())
            a     = agent0.act(state.observation_tensor(), legal)
        else:
            # agent1 plays player 1
            legal = list(state.legal_actions())
            a     = agent1.act(state.observation_tensor(), legal)
        state.apply_action(a)
    # returns [r0,r1]
    return state.returns()[0] > 0

def evaluate_snapshot(game, obs_dim, n_actions, ckpt0, ckpt1):
    # load two agents
    a0 = EvalIDDQNAgent(ckpt0, obs_dim, n_actions)
    a1 = EvalIDDQNAgent(ckpt1, obs_dim, n_actions)

    # head‑to‑head
    wins0 = 0
    for i in range(SELF_GAMES):
        first = i % 2  # alternate who goes first
        if play_match(game, a0, a1, first):
            # if player0 wins and first==0 OR player1 wins and first==1
            wins0 += 1
    champ_idx   = 0 if wins0 >= (SELF_GAMES/2) else 1
    champion    = a0 if champ_idx == 0 else a1
    h2h_winrate = wins0 / SELF_GAMES

    # champion vs MCTS
    mcts_bot = EvalMCTSBot(game, MCTS_SIMS)
    wins     = 0
    for i in range(MCTS_GAMES):
        state = game.new_initial_state()
        while not state.is_terminal():
            pid   = state.current_player()
            if pid == 0:
                legal = list(state.legal_actions())
                a     = champion.act(state.observation_tensor(), legal)
            else:
                a     = mcts_bot.act(state)
            state.apply_action(a)
        if state.returns()[0] > 0:
            wins += 1
    mcts_winrate = wins / MCTS_GAMES

    return champ_idx, h2h_winrate, mcts_winrate

# ───────────── Main ─────────────
if __name__ == "__main__":
    assert os.path.isdir(SNAPSHOT_DIR), f"No snapshots found in {SNAPSHOT_DIR}"
    game      = pyspiel.load_game(f"hex(board_size={BOARD_SIZE})")
    obs_dim   = game.observation_tensor_size()
    n_actions = game.num_distinct_actions()

    # find all matching snapshot pairs
    c0 = sorted(glob.glob(os.path.join(SNAPSHOT_DIR, "agent0_*s.pt")))
    c1 = sorted(glob.glob(os.path.join(SNAPSHOT_DIR, "agent1_*s.pt")))
    assert len(c0) == len(c1) and len(c0)>0, "No matching snapshots"

    with open(EVAL_CSV, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["timestamp_s","champion","h2h_win","mcts_win"])
        for ckpt0, ckpt1 in zip(c0, c1):
            # extract timestamp in seconds from filename
            fname = os.path.basename(ckpt0)
            t = int(fname.split("_")[-1].replace("s.pt",""))
            champ, h2h, mcts_wr = evaluate_snapshot(
                game, obs_dim, n_actions, ckpt0, ckpt1
            )
            print(f"@{t}s → champ={champ}, self={h2h:.2f}, mcts={mcts_wr:.2f}")
            writer.writerow([t, champ, f"{h2h:.3f}", f"{mcts_wr:.3f}"])
    print(f"Wrote evaluation to {EVAL_CSV}")
