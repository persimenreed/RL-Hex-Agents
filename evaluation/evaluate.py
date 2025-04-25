#!/usr/bin/env python3
import os
import glob
import csv
import argparse

import numpy as np
import torch
import tensorflow.compat.v1 as tf
import pyspiel

# ─── project root setup ──────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
import sys
sys.path.insert(0, ROOT)

from open_spiel.python.algorithms import mcts
from open_spiel.python import rl_environment

# your local NFSP implementation (from nfsp.py)
from nfsp import NFSP

# your PPO training file
import ppo_v0                                                              
# :contentReference[oaicite:0]{index=0}&#8203;:contentReference[oaicite:1]{index=1}

tf.disable_v2_behavior()

# ─── evaluation hyper‐parameters ────────────────────────────────────────────
BOARD_SIZE   = 5
TRAIN_HOURS  = 4
MCTS_GAMES   = 100
MCTS_SIMS    = {5: 100, 8: 100, 11: 100}[BOARD_SIZE]

# must match your NFSP training params
RESERVOIRS          = {5: 100_000, 8: 300_000, 11: 500_000}
HIDDEN_SIZES        = {5: [64,64], 8: [128,128], 11: [128,128]}
ANTICIPATORY_PARAMS = {5: 0.1,     8: 0.1,       11: 0.1}

device = torch.device("cpu")


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


class EvalPPOAgent:
    """Wrap your trained PPO policy_net (CNNPolicy) for evaluation."""
    def __init__(self, ckpt_path, obs_dim, board_size, n_actions):
        in_ch = obs_dim // (board_size * board_size)
        self.net = ppo_v0.CNNPolicy(in_ch, board_size, n_actions).to(device)
        self.net.load_state_dict(torch.load(ckpt_path, map_location=device))
        self.net.eval()
        self.pid = 0
        self.in_ch = in_ch
        self.board_size = board_size

    def act(self, state):
        obs   = state.observation_tensor()
        legal = list(state.legal_actions())
        x = torch.tensor(obs, dtype=torch.float32, device=device)
        x = x.view(1, self.in_ch, self.board_size, self.board_size)
        with torch.no_grad():
            logits, _ = self.net(x)
            logits = logits.cpu().numpy()[0]
        mask = np.full_like(logits, -np.inf, dtype=np.float32)
        mask[legal] = 0.0
        return int(np.argmax(logits + mask))


class EvalNFSPAgent:
    """Wrap an NFSP player (pid=0 or 1) and allow restoring from any .ckpt."""
    def __init__(self, sess, pid, obs_dim, n_actions):
        self.pid = pid
        self.sess = sess
        # build exactly as in training
        self.agent = NFSP(
            sess, pid,
            obs_dim, n_actions,
            hidden_layers_sizes=HIDDEN_SIZES[BOARD_SIZE],
            reservoir_buffer_capacity=RESERVOIRS[BOARD_SIZE],
            anticipatory_param=ANTICIPATORY_PARAMS[BOARD_SIZE],
            batch_size=32,
            rl_learning_rate=1e-3,
            sl_learning_rate=1e-4,
            min_buffer_size_to_learn=1000,
            learn_every=32,
            optimizer_str="adam"
        )
        # initialize then we'll overwrite with each checkpoint
        self.sess.run(tf.global_variables_initializer())

    def restore(self, ckpt_prefix):
        saver = tf.train.Saver()
        saver.restore(self.sess, ckpt_prefix)

    def act(self, state):
        # build a minimal TimeStep for OpenSpiel’s NFSP.step()
        p = state.current_player()
        obs   = state.observation_tensor()[p]
        legal = list(state.legal_actions())
        ts = rl_environment.TimeStep(
            step_type=rl_environment.StepType.MID,
            reward=0.0,
            discount=1.0,
            observations={
                "info_state":   [obs, obs],
                "legal_actions":[legal, legal]
            }
        )
        out = self.agent.step(ts, is_evaluation=True)
        return out.action


def evaluate_vs_mcts(game, agent, mcts_bot):
    wins = 0
    for i in range(MCTS_GAMES):
        state = game.new_initial_state()
        swap  = (i >= MCTS_GAMES // 2)
        while not state.is_terminal():
            pid = state.current_player()
            is_agent_turn = (
                (pid == agent.pid and not swap) or
                (pid != agent.pid and swap)
            )
            move = agent.act(state) if is_agent_turn else mcts_bot.act(state)
            state.apply_action(move)

        # check if *that* agent won
        winner = (
            state.returns()[agent.pid]
            if not swap else state.returns()[1 - agent.pid]
        )
        if winner > 0:
            wins += 1
    return wins / MCTS_GAMES


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["nfsp", "ppo"], required=True)
    args = p.parse_args()

    game      = pyspiel.load_game(f"hex(board_size={BOARD_SIZE})")
    obs_dim   = game.observation_tensor_size()
    n_actions = game.num_distinct_actions()
    mcts_bot  = EvalMCTSBot(game, MCTS_SIMS)

    os.makedirs("evaluation", exist_ok=True)
    out_csv = os.path.join("evaluation", f"eval_{BOARD_SIZE}.csv")

    with open(out_csv, "w", newline="") as fout:
        writer = csv.writer(fout)

        if args.algo == "ppo":
            writer.writerow(["timestamp_s", "ppo_mcts_win"])
            snap_dir = os.path.join(
                ROOT, "weights", "ppo",
                f"ppo_hex_{BOARD_SIZE}_{TRAIN_HOURS}h"
            )
            paths = sorted(glob.glob(os.path.join(
                snap_dir, "ppo_policy_*s.pt"
            )))
            for pth in paths:
                t = int(os.path.basename(pth).split("_")[-1].replace("s.pt", ""))
                agent = EvalPPOAgent(pth, obs_dim, BOARD_SIZE, n_actions)
                r = evaluate_vs_mcts(game, agent, mcts_bot)
                print(f"@{t}s → PPO vs MCTS: {r:.2f}")
                writer.writerow([t, f"{r:.3f}"])

        else:  # nfsp
            writer.writerow(
                ["timestamp_s", "nfsp0_mcts_win", "nfsp1_mcts_win"]
            )
            snap_dir = os.path.join(
                ROOT, "weights", "nfsp",
                f"nfsp_hex_{BOARD_SIZE}_{TRAIN_HOURS}h"
            )
            # collect the generic full‐graph checkpoints
            idxs = sorted(glob.glob(os.path.join(
                snap_dir, "nfsp_hex_p0_*s.ckpt.index"
            )))
            prefixes = [i[:-len(".index")] for i in idxs]

            with tf.Session() as sess:
                a0 = EvalNFSPAgent(sess, pid=0, obs_dim=obs_dim,
                                   n_actions=n_actions)
                a1 = EvalNFSPAgent(sess, pid=1, obs_dim=obs_dim,
                                   n_actions=n_actions)

                for prefix in prefixes:
                    t = int(
                        os.path.basename(prefix)
                          .split("_")[-1]
                          .replace("s.ckpt","")
                    )
                    # restore the same snapshot for both players
                    a0.restore(prefix)
                    a1.restore(prefix)

                    r0 = evaluate_vs_mcts(game, a0, mcts_bot)
                    r1 = evaluate_vs_mcts(game, a1, mcts_bot)
                    print(f"@{t}s → NFSP p0: {r0:.2f}, p1: {r1:.2f}")
                    writer.writerow([t, f"{r0:.3f}", f"{r1:.3f}"])

    print(f"Wrote evaluation to {out_csv}")
