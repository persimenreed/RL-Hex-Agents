#!/usr/bin/env python3
import os
import sys
import glob
import csv
import shutil

import tensorflow.compat.v1 as tf
import numpy as np
import pyspiel

tf.disable_v2_behavior()

SCRIPT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
sys.path.insert(0, PROJECT_ROOT)

from open_spiel.python.algorithms import mcts

def evaluate_vs_mcts(game, agent, mcts_bot, pid=0, num_games=100):
    wins = 0
    for i in range(num_games):
        state = game.new_initial_state()
        swap  = (i >= num_games // 2)
        while not state.is_terminal():
            cur = state.current_player()
            is_agent = (cur == pid and not swap) or (cur != pid and swap)
            move = agent.act(state) if is_agent else mcts_bot.act(state)
            state.apply_action(move)
        winner = state.returns()[pid] if not swap else state.returns()[1-pid]
        if winner > 0:
            wins += 1
    return wins / num_games


class EvalMCTSBot:
    def __init__(self, game, sims=50):
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


class EvalAlphaZeroAgent:
    def __init__(self, sess, ckpt_prefix):
        saver = tf.train.import_meta_graph(ckpt_prefix + ".meta", clear_devices=True)
        saver.restore(sess, ckpt_prefix)
        g = tf.get_default_graph()
        self.obs_ph = g.get_tensor_by_name("input:0")
        self.legals_ph = g.get_tensor_by_name("legals_mask:0")
        self.train_ph = g.get_tensor_by_name("training:0")
        self.probs_t = g.get_tensor_by_name("policy_softmax:0")
        self.sess = sess

    def act(self, state):
        obs  = state.observation_tensor()
        mask = np.array(state.legal_actions_mask(), dtype=np.float32)
        feed = {
            self.obs_ph: [obs],
            self.legals_ph: [mask],
            self.train_ph: False,
        }
        probs = self.sess.run(self.probs_t, feed_dict=feed)[0]
        probs = probs * mask
        return int(np.argmax(probs))


def main():
    for B in [11]:
        print(f"\nBoard size {B}")
        game = pyspiel.load_game(f"hex(board_size={B})")
        sims = {5:50, 8:50, 11:50}[B]
        mcts_bot = EvalMCTSBot(game, sims)

        csv_path = os.path.join(PROJECT_ROOT, "evaluation", f"eval_{B}.csv")
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)

        new_col = "az_mcts_win"
        if new_col in header:
            print(f"{csv_path} already has {new_col}, skipping.")
            continue
        header.append(new_col)

        ckpt_dir = os.path.join(
            PROJECT_ROOT, "weights", "alpha_zero", f"alpha_zero_hex_{B}_4h"
        )
        idxs = glob.glob(os.path.join(ckpt_dir, "checkpoint-*.index"))
        prefixes = sorted([p[:-6] for p in idxs],
                          key=lambda p: int(os.path.basename(p).split("-")[-1]))
        if len(prefixes) < len(rows):
            raise RuntimeError(
                f"Found {len(prefixes)} AZ checkpoints but {len(rows)} CSV rows for B={B}"
            )

        # track best
        best_wr = -1.0
        best_pref = None

        tf.reset_default_graph()
        with tf.Session() as sess:
            for i, row in enumerate(rows):
                ckpt = prefixes[i]
                agent = EvalAlphaZeroAgent(sess, ckpt)
                wr = evaluate_vs_mcts(game, agent, mcts_bot, pid=0)
                print(f"[{i+1}/{len(rows)}] {os.path.basename(ckpt)} → {wr:.3f}")
                row.append(f"{wr:.3f}")

                if wr > best_wr:
                    best_wr, best_pref = wr, ckpt

        # write CSV
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        print(f"Updated {csv_path}")

        # copy best
        if best_pref:
            dst_dir = os.path.join(SCRIPT_DIR, "best_weight", "alpha_zero", f"hex_{B}")
            os.makedirs(dst_dir, exist_ok=True)
            for fn in glob.glob(best_pref + "*"):
                shutil.copy(fn, dst_dir)
            print(f"Best checkpoint (wr={best_wr:.3f}) copied to {dst_dir}")

if __name__ == "__main__":
    main()
