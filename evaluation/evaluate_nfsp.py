#!/usr/bin/env python3
"""
evaluate_nfsp.py

Read your existing evaluation/eval_{B}.csv, append nfsp0_mcts_win and nfsp1_mcts_win
(mapped by sorted avg-network timestamps → row order), then copy the single best
(agent) checkpoint files (both avg_ and q_ nets) into best_weight/nfsp/hex_{B}/.
"""
import os
import sys
import glob
import csv
import shutil

import numpy as np
import tensorflow.compat.v1 as tf
import pyspiel

from open_spiel.python.algorithms import mcts
from open_spiel.python.algorithms.nfsp import NFSP

tf.disable_v2_behavior()

SCRIPT_DIR   = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
sys.path.insert(0, PROJECT_ROOT)


def evaluate_vs_mcts(game, agent, mcts_bot, pid, num_games=100):
    wins = 0
    for i in range(num_games):
        state = game.new_initial_state()
        swap  = (i >= num_games // 2)
        while not state.is_terminal():
            cur      = state.current_player()
            is_agent = (cur == pid and not swap) or (cur != pid and swap)
            move     = agent.act(state) if is_agent else mcts_bot.act(state)
            state.apply_action(move)
        winner = state.returns()[pid] if not swap else state.returns()[1-pid]
        if winner > 0:
            wins += 1
    return wins / num_games


class EvalMCTSBot:
    def __init__(self, game, sims=100):
        self.bot = mcts.MCTSBot(
            game, 1.0, sims,
            mcts.RandomRolloutEvaluator(),
            solve=False,
            child_selection_fn=mcts.SearchNode.puct_value,
            verbose=False,
            dont_return_chance_node=True
        )
    def act(self, state):
        return self.bot.mcts_search(state).best_child().action


class EvalNFSPAgent:
    def __init__(self, sess, pid, obs_dim, n_actions, board_size):
        self.pid  = pid
        self.sess = sess
        self.agent = NFSP(
            sess, pid,
            obs_dim, n_actions,
            hidden_layers_sizes={5:[64,64], 8:[128,128], 11:[128,128]}[board_size],
            reservoir_buffer_capacity={5:100000, 8:300000,11:500000}[board_size],
            anticipatory_param={5:0.1,8:0.1,11:0.1}[board_size],
            batch_size=32,
            rl_learning_rate=1e-3,
            sl_learning_rate=1e-4,
            min_buffer_size_to_learn=1000,
            learn_every=32,
            optimizer_str="adam"
        )
        sess.run(tf.global_variables_initializer())

    def restore_avg(self, avg_prefix):
        """Restore only the avg-policy network variables."""
        saver = tf.train.Saver(self.agent._avg_network.variables)
        saver.restore(self.sess, avg_prefix)

    def act(self, state):
        obs  = np.array(state.observation_tensor(), dtype=np.float32)
        info = obs.reshape(1, -1)
        probs = self.sess.run(
            self.agent._avg_policy_probs,
            feed_dict={self.agent._info_state_ph: info}
        )[0]
        legal = state.legal_actions()
        mask  = np.zeros_like(probs); mask[legal] = 1.0
        return int(np.argmax(probs * mask))


if __name__ == "__main__":
    for B in [5, 8, 11]:
        print(f"\n=== Board size {B} ===")
        game      = pyspiel.load_game(f"hex(board_size={B})")
        sims      = {5:100, 8:100, 11:100}[B]
        mcts_bot  = EvalMCTSBot(game, sims)
        obs_dim   = game.observation_tensor_size()
        n_actions = game.num_distinct_actions()

        # 1) load existing CSV
        csv_path = os.path.join(PROJECT_ROOT, "evaluation", f"eval_{B}.csv")
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows   = list(reader)

        # 2) append new columns if missing
        cols = ["nfsp0_mcts_win", "nfsp1_mcts_win"]
        if all(c in header for c in cols):
            print(f"  → {csv_path} already has NFSP columns, skipping.")
            continue
        header.extend(cols)

        # 3) gather sorted avg-network prefixes for each pid
        ckpt_dir = os.path.join(PROJECT_ROOT, "weights", "nfsp", f"nfsp_hex_{B}_4h")
        avg0 = sorted(
            glob.glob(os.path.join(ckpt_dir, "avg_network_pid0_*s.ckpt.index")),
            key=lambda p: int(os.path.basename(p).split('_')[-1].replace('s.ckpt.index',''))
        )
        avg1 = sorted(
            glob.glob(os.path.join(ckpt_dir, "avg_network_pid1_*s.ckpt.index")),
            key=lambda p: int(os.path.basename(p).split('_')[-1].replace('s.ckpt.index',''))
        )
        # strip the ".index" to get prefixes
        avg0 = [p[:-len(".index")] for p in avg0]
        avg1 = [p[:-len(".index")] for p in avg1]

        if len(avg0) < len(rows) or len(avg1) < len(rows):
            raise RuntimeError(
                f"Not enough avg_network ckpts (p0:{len(avg0)}, p1:{len(avg1)}) "
                f"for {len(rows)} rows in {csv_path}"
            )

        # 4) evaluate
        best_wr   = -1.0
        best_pid  = None
        best_ts   = None

        tf.reset_default_graph()
        with tf.Session() as sess:
            agent0 = EvalNFSPAgent(sess, pid=0, obs_dim=obs_dim, n_actions=n_actions, board_size=B)
            agent1 = EvalNFSPAgent(sess, pid=1, obs_dim=obs_dim, n_actions=n_actions, board_size=B)

            for i, row in enumerate(rows):
                pre0 = avg0[i]
                pre1 = avg1[i]
                ts   = os.path.basename(pre0).split('_')[-1].replace('s.ckpt','')

                agent0.restore_avg(pre0)
                r0 = evaluate_vs_mcts(game, agent0, mcts_bot, pid=0)
                agent1.restore_avg(pre1)
                r1 = evaluate_vs_mcts(game, agent1, mcts_bot, pid=1)

                row.extend([f"{r0:.3f}", f"{r1:.3f}"])
                print(f"   [{i+1}/{len(rows)}] @{ts}s → p0 {r0:.3f}, p1 {r1:.3f}")

                # track best single-agent
                if r0 > best_wr:
                    best_wr, best_pid, best_ts = r0, 0, ts
                if r1 > best_wr:
                    best_wr, best_pid, best_ts = r1, 1, ts

        # 5) write back CSV
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        print(f"  → updated {csv_path}")

        # 6) copy best avg+q for best_pid at best_ts
        dst = os.path.join(PROJECT_ROOT, "best_weight", "nfsp", f"hex_{B}")
        os.makedirs(dst, exist_ok=True)
        pattern = os.path.join(ckpt_dir, f"*pid{best_pid}_{best_ts}s.ckpt*")
        for fn in glob.glob(pattern):
            shutil.copy(fn, dst)
        print(f"  → best NFSP pid{best_pid}@{best_ts}s (win={best_wr:.3f}) copied to {dst}")
