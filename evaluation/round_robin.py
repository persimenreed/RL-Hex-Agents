#!/usr/bin/env python3
"""
round_robin.py

Run a round-robin tournament among the best PPO, NFSP, and AlphaZero agents for Hex
on 5×5, 8×8, and 11×11 boards. Logs match results to CSV and draws all plots.
"""

import os
import sys
import glob
import csv
import random
import numpy as np
import torch
import pyspiel
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Disable TF v2 behavior
tf.disable_v2_behavior()

# Paths
SCRIPT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)
BEST_WEIGHT = os.path.join(SCRIPT_DIR, "best_weight")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Config
BOARDS = [5, 8, 11]
MODELS = ["PPO", "NFSP", "AZ"]
CMAP = {"PPO": "C0", "NFSP": "C1", "AZ": "C2"}

# --- Agents ---

class EvalPPOAgent:
    def __init__(self, ckpt_path, obs_dim, board_size, n_actions):
        from models.ppo import CNNPolicy
        self.board_size = board_size
        self.in_ch = obs_dim // (board_size * board_size)
        self.net = CNNPolicy(self.in_ch, board_size, n_actions)
        self.net.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        self.net.eval()

    def act(self, state):
        obs = state.observation_tensor()
        legal = list(state.legal_actions())
        x = torch.tensor(obs, dtype=torch.float32)
        x = x.view(1, self.in_ch, self.board_size, self.board_size)
        with torch.no_grad():
            logits, _ = self.net(x)
            logits = logits.numpy()[0]
        mask = np.full_like(logits, -np.inf, dtype=np.float32)
        mask[legal] = 0.0
        return int(np.argmax(logits + mask))

class EvalNFSPAgent:
    def __init__(self, sess, pid, obs_dim, n_actions, board_size):
        from open_spiel.python.algorithms.nfsp import NFSP
        self.pid = pid
        self.sess = sess
        self.agent = NFSP(
            sess, pid,
            obs_dim, n_actions,
            hidden_layers_sizes=[64],
            reservoir_buffer_capacity=20_000,
            anticipatory_param=0.1,
            batch_size=128,
            rl_learning_rate=0.01,
            sl_learning_rate=0.005,
            min_buffer_size_to_learn=1000,
            learn_every=64,
            optimizer_str="adam"
        )
        sess.run(tf.global_variables_initializer())

    def restore_avg(self, prefix):
        reader = tf.train.NewCheckpointReader(prefix)
        ckpt_vars = set(reader.get_variable_to_shape_map().keys())
        graph_vars = self.agent._avg_network.variables
        ckpt_pref = next(iter(ckpt_vars)).split('/')[0]
        graph_pref = graph_vars[0].op.name.split('/')[0]
        var_map = {}
        for var in graph_vars:
            ckpt_name = var.op.name.replace(graph_pref, ckpt_pref)
            if ckpt_name in ckpt_vars:
                var_map[ckpt_name] = var
        saver = tf.train.Saver(var_list=var_map)
        saver.restore(self.sess, prefix)

    def act(self, state):
        obs = np.array(state.observation_tensor(), dtype=np.float32)
        info = obs.reshape(1, -1)
        probs = self.sess.run(self.agent._avg_policy_probs, feed_dict={self.agent._info_state_ph: info})[0]
        legal = state.legal_actions()
        mask = np.zeros_like(probs)
        mask[legal] = 1.0
        return int(np.argmax(probs * mask))

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
        obs = np.array(state.observation_tensor(), dtype=np.float32)
        mask = np.array(state.legal_actions_mask(), dtype=np.float32)
        feed = {self.obs_ph: [obs], self.legals_ph: [mask], self.train_ph: False}
        probs = self.sess.run(self.probs_t, feed_dict=feed)[0]
        probs *= mask
        return int(np.argmax(probs))

# --- Evaluation helpers ---

def evaluate_vs(game, agent, opp, pid, n_games=100):
    a_start_a_win = 0
    a_start_b_win = 0
    b_start_a_win = 0
    b_start_b_win = 0

    for i in range(n_games):
        state = game.new_initial_state()
        for _ in range(1):
            if state.is_terminal():
                break
            actions = state.legal_actions()
            state.apply_action(random.choice(actions))
        swap = (i >= n_games // 2)
        first = (state.current_player() == pid) ^ swap

        while not state.is_terminal():
            cur = state.current_player()
            move = agent.act(state) if ((cur == pid) != swap) else opp.act(state)
            state.apply_action(move)

        model_a_won = (state.returns()[pid] if not swap else state.returns()[1 - pid]) > 0

        if first:  # Model A starts
            if model_a_won:
                a_start_a_win += 1
            else:
                a_start_b_win += 1
        else:      # Model B starts
            if model_a_won:
                b_start_a_win += 1
            else:
                b_start_b_win += 1

    return a_start_a_win, a_start_b_win, b_start_a_win, b_start_b_win


def plot_confusion_matrix(mat, models, out_path):
    fig, ax = plt.subplots(figsize=(6,6))
    cax = ax.matshow(mat, cmap="Blues", vmin=0, vmax=1)

    for (i, j), val in np.ndenumerate(mat):
        color = "black" if val < 0.5 else "white"
        ax.text(j, i, f"{val*100:.1f}%", ha="center", va="center", color=color, fontsize=12, fontweight="bold")

    ax.set_xticks(range(2))
    ax.set_yticks(range(2))
    ax.set_xticklabels([f"{models[0]} Starts", f"{models[1]} Starts"])
    ax.set_yticklabels([f"{models[0]} Wins", f"{models[1]} Wins"])

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_round_robin_bar(rel_results, board_size):
    matchups = [("PPO", "NFSP"), ("AZ", "PPO"), ("NFSP", "AZ")]
    labels = []
    wins_bottom = []
    wins_top = []

    for bot, top in matchups:
        wr_bot = None
        for _, m1, m2, wr in rel_results:
            if m1 == bot and m2 == top:
                wr_bot = wr
                break
            if m1 == top and m2 == bot:
                wr_bot = 1.0 - wr
                break
        if wr_bot is None:
            raise ValueError(f"No result for {bot} vs {top}")
        labels.append(f"{bot} vs {top}")
        wins_bottom.append(wr_bot)
        wins_top.append(1.0 - wr_bot)

    x = np.arange(len(labels))
    width = 0.6

    fig, ax = plt.subplots()
    for i, (bot, _) in enumerate(matchups):
        ax.bar(x[i], wins_bottom[i], width, color=CMAP[bot])
    for i, (_, top) in enumerate(matchups):
        ax.bar(x[i], wins_top[i], width, bottom=wins_bottom[i], color=CMAP[top])

    for i in range(len(labels)):
        ax.text(x[i], wins_bottom[i]/2, f"{wins_bottom[i]*100:.0f}%", ha="center", va="center", color="white", fontweight="bold")
        ax.text(x[i], wins_bottom[i]+wins_top[i]/2, f"{wins_top[i]*100:.0f}%", ha="center", va="center", color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Win rate")
    #ax.set_title(f"{board_size}×{board_size} Round-robin results")
    legend_handles = [Patch(color=CMAP[m], label=m if m != "AZ" else "AlphaZero") for m in MODELS]
    ax.legend(handles=legend_handles, loc="upper right")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"round_robin_{board_size}x{board_size}.png"))
    plt.close()

# --- Main ---

def main():
    all_results = []
    confusion_stats = []

    for B in BOARDS:
        tf.reset_default_graph()
        print(f"--- Board {B}×{B} ---")
        game = pyspiel.load_game(f"hex(board_size={B})")

        ppo = EvalPPOAgent(sorted(glob.glob(os.path.join(BEST_WEIGHT, "ppo", f"hex_{B}", "*.pt")))[-1],
                           game.observation_tensor_size(), B, game.num_distinct_actions())

        sess_n = tf.Session()
        nfsp = EvalNFSPAgent(sess_n, 0, game.observation_tensor_size(), game.num_distinct_actions(), B)
        nfsp.restore_avg(sorted(glob.glob(os.path.join(BEST_WEIGHT, "nfsp", f"hex_{B}", "avg_network_pid*_*.ckpt.index")))[-1][:-6])

        sess_az = tf.Session()
        az = EvalAlphaZeroAgent(sess_az, sorted(glob.glob(os.path.join(BEST_WEIGHT, "alpha_zero", f"hex_{B}", "*.meta")))[-1][:-5])

        bots = [("PPO", ppo), ("NFSP", nfsp), ("AZ", az)]

        board_results = []

        for i in range(len(bots)):
            for j in range(i+1, len(bots)):
                m1, a1 = bots[i]
                m2, a2 = bots[j]
                a_start_a_win, a_start_b_win, b_start_a_win, b_start_b_win = evaluate_vs(game, a1, a2, pid=0)

                total_a_start = a_start_a_win + a_start_b_win
                total_b_start = b_start_a_win + b_start_b_win

                mat = np.array([
                    [
                        a_start_a_win / total_a_start if total_a_start > 0 else 0.0,
                        b_start_a_win / total_b_start if total_b_start > 0 else 0.0
                    ],
                    [
                        a_start_b_win / total_a_start if total_a_start > 0 else 0.0,
                        b_start_b_win / total_b_start if total_b_start > 0 else 0.0
                    ]
                ])

                save_p = os.path.join(OUTPUT_DIR, f"confusion_{B}x{B}_{m1}_vs_{m2}.png")
                plot_confusion_matrix(mat, [m1, m2], save_p)

                # Calculate total win rate of model A
                total_games = a_start_a_win + a_start_b_win + b_start_a_win + b_start_b_win
                wins_for_a = a_start_a_win + b_start_a_win
                wr = wins_for_a / total_games if total_games > 0 else 0.0

                # Save to results for round robin bar plot
                board_results.append((B, m1, m2, wr))
                all_results.append((B, m1, m2, wr))

        plot_round_robin_bar(board_results, B)


    # Aggregate confusion
    agg_mat = np.zeros((3, 3))
    counts = np.zeros((3, 3))
    idx = {m: i for i, m in enumerate(MODELS)}
    for _, m1, m2, wr in all_results:
        agg_mat[idx[m1], idx[m2]] += wr
        agg_mat[idx[m2], idx[m1]] += (1-wr)
        counts[idx[m1], idx[m2]] += 1
        counts[idx[m2], idx[m1]] += 1
    agg_mat /= np.maximum(counts, 1)

    agg_out = os.path.join(OUTPUT_DIR, "confusion_aggregated.png")
    plot_confusion_matrix(agg_mat, MODELS, agg_out)
    print(f"Aggregated confusion → {agg_out}")

    # --- Aggregated round robin bar plot across all board sizes ---

    matchups = [("PPO", "NFSP"), ("AZ", "PPO"), ("NFSP", "AZ")]
    labels = []
    wins_bottom = []
    wins_top = []

    for bot, top in matchups:
        wrs = []
        for _, m1, m2, wr in all_results:
            if m1 == bot and m2 == top:
                wrs.append(wr)
            elif m1 == top and m2 == bot:
                wrs.append(1.0 - wr)
        if not wrs:
            raise RuntimeError(f"No aggregated results found for {bot} vs {top}")
        wr_bot = np.mean(wrs)
        labels.append(f"{bot} vs {top}")
        wins_bottom.append(wr_bot)
        wins_top.append(1.0 - wr_bot)

    x = np.arange(len(labels))
    width = 0.6

    fig, ax = plt.subplots()
    for i, (bot, _) in enumerate(matchups):
        ax.bar(x[i], wins_bottom[i], width, color=CMAP[bot])
    for i, (_, top) in enumerate(matchups):
        ax.bar(x[i], wins_top[i], width, bottom=wins_bottom[i], color=CMAP[top])

    for i in range(len(labels)):
        ax.text(x[i], wins_bottom[i]/2, f"{wins_bottom[i]*100:.0f}%", ha="center", va="center", color="white", fontweight="bold")
        ax.text(x[i], wins_bottom[i]+wins_top[i]/2, f"{wins_top[i]*100:.0f}%", ha="center", va="center", color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Win rate")
    #ax.set_title("Aggregated Round-robin Results Across All Boards")

    legend_handles = [Patch(color=CMAP[m], label=m if m != "AZ" else "AlphaZero") for m in MODELS]
    ax.legend(handles=legend_handles, loc="upper right")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "round_robin_aggregated.png"))
    plt.close()

    print(f"Aggregated round robin plot → {os.path.join(OUTPUT_DIR, 'round_robin_aggregated.png')}")


if __name__ == "__main__":
    main()
