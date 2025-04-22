#!/usr/bin/env python3
import os
import glob
import csv
import torch
import sys
import random
import numpy as np
import pyspiel
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from iddqn.model import QNet      
from open_spiel.python.algorithms import mcts

# ───────────── Evaluation hyperparams ─────────────
BOARD_SIZE    = 5               # change to 5 or 11 for other boards
HIDDEN_SIZE   = 128             # must match your IDDQN hidden size
SELF_GAMES    = 2              # head‑to‑head games
MCTS_GAMES    = 100              # games vs MCTSBot
MCTS_SIMS    = {5: 200, 8: 500, 11: 1000}[BOARD_SIZE] # sims/move for MCTSBot
SEED          = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ──────────── Paths ────────────
SNAPSHOT_DIR = os.path.join(ROOT, "weights", "iddqn", f"iddqn_hex_{BOARD_SIZE}_4h")
EVAL_CSV     = os.path.join(ROOT, "evaluation", f"iddqn_eval_{BOARD_SIZE}.csv")

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
        root = self.bot.mcts_search(state)
        # pick the child with highest visit/explore count
        return root.best_child().action

# ───────────── Game logic ─────────────
def play_match(game, agent_black, agent_white):
    state = game.new_initial_state()
    while not state.is_terminal():
        pid   = state.current_player()
        legal = list(state.legal_actions())
        if pid == 0:
            move = agent_black.act(state.observation_tensor(), legal)
        else:
            move = agent_white.act(state.observation_tensor(), legal)
        state.apply_action(move)
    # returns True if Black (player0) wins
    return state.returns()[0] > 0

def evaluate_snapshot(game, obs_dim, n_actions, ckpt0, ckpt1):
    # load two eval agents
    a0 = EvalIDDQNAgent(ckpt0, obs_dim, n_actions)
    a1 = EvalIDDQNAgent(ckpt1, obs_dim, n_actions)

    # ─── head‐to‐head, alternating who is Black ───
    wins_as_black = 0
    for i in range(SELF_GAMES):
        if i % 2 == 0:
            # even: a0 is Black, a1 is White
            if play_match(game, a0, a1):
                wins_as_black += 1
        else:
            # odd: a1 is Black, a0 is White
            if not play_match(game, a0, a1):
                # a1 won as Black ⇒ count it
                wins_as_black += 1

    # compute agent0’s overall win‐rate vs agent1:
    # agent0 wins when it was Black and won_as_black,
    # plus when it was White and Black lost:
    h2h_wins = (wins_as_black if SELF_GAMES % 2 == 0 else wins_as_black)
    # but easier: just count wins for a0:
    #    a0_black_wins + a0_white_wins:
    a0_black_wins = sum(play_match(game, a0, a1) for i in range(0, SELF_GAMES, 2))
    a0_white_wins = sum(not play_match(game, a0, a1) for i in range(1, SELF_GAMES, 2))
    h2h_rate      = (a0_black_wins + a0_white_wins) / SELF_GAMES

    # champion = 0 if a0 stronger, else 1
    champ_idx = 0 if h2h_rate >= 0.5 else 1
    champ     = a0 if champ_idx == 0 else a1

    # ─── champion vs MCTSBot, both colors ───
    mcts_bot = EvalMCTSBot(game, MCTS_SIMS)
    mcts_wins = 0
    for i in range(MCTS_GAMES):
        state = game.new_initial_state()
        swap  = (i >= (MCTS_GAMES // 2))
        while not state.is_terminal():
            pid   = state.current_player()
            legal = list(state.legal_actions())
            if (pid == 0 and not swap) or (pid == 1 and swap):
                # champion’s turn
                move = champ.act(state.observation_tensor(), legal)
            else:
                # MCTSBot’s turn
                move = mcts_bot.act(state)
            state.apply_action(move)
        # winner: if swap then champion was White → check returns()[1]
        winner = state.returns()[0] if not swap else state.returns()[1]
        if winner > 0:
            mcts_wins += 1

    mcts_rate = mcts_wins / MCTS_GAMES

    return champ_idx, h2h_rate, mcts_rate

    # ───────────── Main ─────────────
if __name__ == "__main__":
    assert os.path.isdir(SNAPSHOT_DIR), f"No snapshots found in {SNAPSHOT_DIR}"
    game      = pyspiel.load_game(f"hex(board_size={BOARD_SIZE})")
    obs_dim   = game.observation_tensor_size()
    n_actions = game.num_distinct_actions()

    # find all matching snapshot pairs
    paths0 = glob.glob(os.path.join(SNAPSHOT_DIR, "agent0_*s.pt"))
    paths1 = glob.glob(os.path.join(SNAPSHOT_DIR, "agent1_*s.pt"))

    # pair and sort by numeric timestamp
    pairs = []
    for p0 in paths0:
        t = int(os.path.basename(p0).split("_")[-1].replace("s.pt",""))
        p1 = p0.replace("agent0_", "agent1_")
        pairs.append((t, p0, p1))

    pairs.sort(key=lambda x: x[0])  # sort by t

    with open(EVAL_CSV, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["timestamp_s","champion","h2h_win","mcts_win"])
        for t, ckpt0, ckpt1 in pairs:
            champ, h2h, mcts_wr = evaluate_snapshot(
                game, obs_dim, n_actions, ckpt0, ckpt1
            )
            print(f"@{t}s → champ={champ}, self={h2h:.2f}, mcts={mcts_wr:.2f}")
            writer.writerow([t, champ, f"{h2h:.3f}", f"{mcts_wr:.3f}"])
    print(f"Wrote evaluation to {EVAL_CSV}")
