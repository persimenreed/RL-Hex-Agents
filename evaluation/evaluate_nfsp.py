#!/usr/bin/env python3
"""
Evaluate NFSP average policies against MCTS for Hex (5×5, 8×8, 11×11) and write separate CSVs.
"""
import os
import sys
import glob
import csv
import numpy as np
import tensorflow.compat.v1 as tf
import pyspiel
from open_spiel.python.algorithms import mcts
from open_spiel.python.algorithms.nfsp import NFSP

# disable v2 behavior for TensorFlow
tf.disable_v2_behavior()

# ensure project root is on PYTHONPATH
SCRIPT_DIR   = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
sys.path.insert(0, PROJECT_ROOT)


def evaluate_vs_mcts(game, agent, mcts_bot, pid):
    wins = 0
    games = 100
    for i in range(games):
        state = game.new_initial_state()
        swap = (i >= games // 2)
        while not state.is_terminal():
            current = state.current_player()
            agent_turn = (current == pid and not swap) or (current != pid and swap)
            move = agent.act(state) if agent_turn else mcts_bot.act(state)
            state.apply_action(move)
        winner = state.returns()[pid] if not swap else state.returns()[1 - pid]
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

class EvalNFSPAgent:
    def __init__(self, sess, pid, obs_dim, n_actions, board_size):
        self.pid = pid
        self.sess = sess
        # instantiate with training hyperparameters
        self.agent = NFSP(
            sess, pid,
            obs_dim, n_actions,
            hidden_layers_sizes={5:[64,64], 8:[128,128], 11:[128,128]}[board_size],
            reservoir_buffer_capacity={5:100000, 8:300000, 11:500000}[board_size],
            anticipatory_param={5:0.1, 8:0.1, 11:0.1}[board_size],
            batch_size=32,
            rl_learning_rate=1e-3,
            sl_learning_rate=1e-4,
            min_buffer_size_to_learn=1000,
            learn_every=32,
            optimizer_str="adam"
        )
        # initialize variables
        sess.run(tf.global_variables_initializer())

    def restore(self, prefix):
        saver = tf.train.Saver()
        saver.restore(self.sess, prefix)

    def act(self, state):
        # directly compute average-policy probs
        obs = state.observation_tensor()
        info = np.reshape(obs, [1, -1])
        probs = self.sess.run(
            self.agent._avg_policy_probs,
            feed_dict={self.agent._info_state_ph: info}
        )[0]
        legal = list(state.legal_actions())
        mask = np.zeros_like(probs)
        mask[legal] = 1.0
        masked = probs * mask
        return int(np.argmax(masked))

if __name__ == "__main__":
    # evaluate for each board size
    for B in [5, 8, 11]:
        game = pyspiel.load_game(f"hex(board_size={B})")
        sims = {5:100, 8:100, 11:100}[B]
        mcts_bot = EvalMCTSBot(game, sims)
        obs_dim = game.observation_tensor_size()
        n_actions = game.num_distinct_actions()

        os.makedirs("evaluation", exist_ok=True)
        out_csv = f"evaluation/eval_{B}.csv"

        # collect checkpoint prefixes (strip .index)
        idx_paths = glob.glob(
            os.path.join(
                PROJECT_ROOT,
                f"weights/nfsp/nfsp_hex_{B}_4h/nfsp_hex_p0_*s.ckpt.index"
            )
        )
        prefixes = [p[:-len(".index")] for p in idx_paths]
        prefixes.sort(key=lambda p: int(os.path.basename(p).split('_')[-1].replace('s.ckpt','')))

        with tf.Session() as sess, open(out_csv, "w", newline="") as fout:
            writer = csv.writer(fout)
            writer.writerow(["timestamp_s", "nfsp0_mcts_win", "nfsp1_mcts_win"])

            # create agents once
            a0 = EvalNFSPAgent(sess, pid=0, obs_dim=obs_dim, n_actions=n_actions, board_size=B)
            a1 = EvalNFSPAgent(sess, pid=1, obs_dim=obs_dim, n_actions=n_actions, board_size=B)

            for prefix in prefixes:
                t = int(os.path.basename(prefix).split('_')[-1].replace('s.ckpt',''))
                a0.restore(prefix)
                a1.restore(prefix)
                r0 = evaluate_vs_mcts(game, a0, mcts_bot, pid=0)
                r1 = evaluate_vs_mcts(game, a1, mcts_bot, pid=1)
                print(f"[NFSP {B}×{B}] @{t}s → p0 {r0:.2f}, p1 {r1:.2f}")
                writer.writerow([t, f"{r0:.3f}", f"{r1:.3f}"])

        print(f"Wrote NFSP results to {out_csv}")
