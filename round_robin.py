#!/usr/bin/env python3
"""
round_robin.py

Run a round-robin tournament among the best PPO, NFSP, and AlphaZero agents for Hex
on 5×5, 8×8, and 11×11 boards. Logs match results to CSV and draws a summary plot.
"""
import os
import sys
import glob
import numpy as np
import tensorflow.compat.v1 as tf
import torch
import pyspiel
import matplotlib.pyplot as plt
import csv

# Disable TF v2 behaviors
tf.disable_v2_behavior()

# Make sure project root is on PYTHONPATH
SCRIPT_DIR   = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

# --- Agent definitions ---

class EvalMCTSBot:
    def __init__(self, game, sims):
        from open_spiel.python.algorithms import mcts
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

class EvalPPOAgent:
    def __init__(self, ckpt_path, obs_dim, board_size, n_actions):
        from ppo.ppo_v0 import CNNPolicy
        self.board_size = board_size
        self.in_ch      = obs_dim // (board_size * board_size)
        self.net        = CNNPolicy(self.in_ch, board_size, n_actions)
        self.net.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        self.net.eval()

    def act(self, state):
        obs   = state.observation_tensor()
        legal = list(state.legal_actions())
        x     = torch.tensor(obs, dtype=torch.float32)
        x     = x.view(1, self.in_ch, self.board_size, self.board_size)
        with torch.no_grad():
            logits, _ = self.net(x)
            logits    = logits.numpy()[0]
        mask = np.full_like(logits, -np.inf, dtype=np.float32)
        mask[legal] = 0.0
        return int(np.argmax(logits + mask))

class EvalNFSPAgent:
    def __init__(self, sess, pid, obs_dim, n_actions, board_size):
        from open_spiel.python.algorithms.nfsp import NFSP
        self.pid  = pid
        self.sess = sess
        self.agent = NFSP(
            sess, pid,
            obs_dim, n_actions,
            hidden_layers_sizes={5:[64,64],8:[128,128],11:[128,128]}[board_size],
            reservoir_buffer_capacity={5:100000,8:300000,11:500000}[board_size],
            anticipatory_param={5:0.1,8:0.1,11:0.1}[board_size],
            batch_size=32,
            rl_learning_rate=1e-3,
            sl_learning_rate=1e-4,
            min_buffer_size_to_learn=1000,
            learn_every=32,
            optimizer_str="adam"
        )
        sess.run(tf.global_variables_initializer())

    def restore_avg(self, prefix):
        saver = tf.train.Saver(self.agent._avg_network.variables)
        saver.restore(self.sess, prefix)

    def act(self, state):
        obs   = np.array(state.observation_tensor(), dtype=np.float32)
        info  = obs.reshape(1, -1)
        probs = self.sess.run(
            self.agent._avg_policy_probs,
            feed_dict={self.agent._info_state_ph: info}
        )[0]
        legal = state.legal_actions()
        mask  = np.zeros_like(probs); mask[legal] = 1.0
        return int(np.argmax(probs * mask))

class EvalAlphaZeroAgent:
    def __init__(self, sess, ckpt_prefix):
        saver = tf.train.import_meta_graph(ckpt_prefix + ".meta", clear_devices=True)
        saver.restore(sess, ckpt_prefix)
        g = tf.get_default_graph()
        self.obs_ph    = g.get_tensor_by_name("input:0")
        self.legals_ph = g.get_tensor_by_name("legals_mask:0")
        self.train_ph  = g.get_tensor_by_name("training:0")
        self.probs_t   = g.get_tensor_by_name("policy_softmax:0")
        self.sess      = sess

    def act(self, state):
        obs  = np.array(state.observation_tensor(), dtype=np.float32)
        mask = np.array(state.legal_actions_mask(), dtype=np.float32)
        feed = {self.obs_ph: [obs], self.legals_ph: [mask], self.train_ph: False}
        probs = self.sess.run(self.probs_t, feed_dict=feed)[0]
        probs *= mask
        return int(np.argmax(probs))

# --- Match and tournament logic ---
import random

def evaluate_vs_opponent(game, agent, opp, pid, num_games=100):
    wins = 0
    for i in range(num_games):
        state = game.new_initial_state()
        for _ in range(1):
            if state.is_terminal(): break
            actions = state.legal_actions()
            state.apply_action(random.choice(actions))
        swap  = (i >= num_games//2)
        while not state.is_terminal():
            cur      = state.current_player()
            is_agent = (cur == pid and not swap) or (cur != pid and swap)
            move     = agent.act(state) if is_agent else opp.act(state)
            state.apply_action(move)
        result = state.returns()[pid] if not swap else state.returns()[1-pid]
        if result > 0:
            wins += 1
    return wins / num_games


def main():
    #boards = [5, 8, 11]
    boards = [5]
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results = []

    for B in boards:
        print(f"\n--- Board {B} ---")
        game = pyspiel.load_game(f"hex(board_size={B})")

        # PPO: look in best_weight/ppo first, then weights/ppo fallback
        ppo_dir = os.path.join(PROJECT_ROOT, "best_weight", "ppo", f"hex_{B}")
        ppths = glob.glob(os.path.join(ppo_dir, "*.pt"))
        if not ppths:
            ppths = glob.glob(os.path.join(PROJECT_ROOT, "weights", "ppo", f"ppo_hex_{B}_4h", "ppo_policy_*s.pt"))
        ppth = sorted(ppths)[-1]
        ppo  = EvalPPOAgent(ppth, game.observation_tensor_size(), B, game.num_distinct_actions())

        # NFSP: best_weight/nfsp then fallback
        nfsp_dir = os.path.join(PROJECT_ROOT, "best_weight", "nfsp", f"hex_{B}")
        avg_ckpts = glob.glob(os.path.join(nfsp_dir, "avg_network_pid0_*s.ckpt.index"))
        if not avg_ckpts:
            avg_ckpts = glob.glob(os.path.join(PROJECT_ROOT, "weights", "nfsp", f"nfsp_hex_{B}_4h", "avg_network_pid0_*s.ckpt.index"))
        avg0 = avg_ckpts[-1][:-len(".index")]
        sess_n = tf.Session()
        nfsp   = EvalNFSPAgent(sess_n, 0, game.observation_tensor_size(), game.num_distinct_actions(), B)
        nfsp.restore_avg(avg0)

        # AlphaZero: best_weight/alpha_zero then fallback
        az_dir = os.path.join(PROJECT_ROOT, "best_weight", "alpha_zero", f"hex_{B}")
        metas  = glob.glob(os.path.join(az_dir, "*.meta"))
        if not metas:
            metas = glob.glob(os.path.join(PROJECT_ROOT, "weights", "alpha_zero", f"alpha_zero_hex_{B}_4h", "checkpoint-*.meta"))
        az_pref = metas[-1][:-len(".meta")]
        sess_az = tf.Session()
        az      = EvalAlphaZeroAgent(sess_az, az_pref)

        bots = [("PPO", ppo), ("NFSP", nfsp), ("AZ", az)]
        for i in range(len(bots)):
            for j in range(i+1, len(bots)):
                name_i, agent_i = bots[i]
                name_j, agent_j = bots[j]
                wr_i = evaluate_vs_opponent(game, agent_i, agent_j, pid=0)
                print(f"{name_i} vs {name_j}: {wr_i:.3f}/{1-wr_i:.3f}")
                all_results.append((B, name_i, name_j, wr_i))

    # write CSV
    csv_out = os.path.join(OUTPUT_DIR, f"round_robin_results_{B}x{B}.csv")
    with open(csv_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["board_size","model1","model2","winrate_model1"])
        w.writerows(all_results)
    print(f"Results → {csv_out}")

        # new stacked‐bar plotting
        # --- plot (replace your old plotting section with this) ---
        # --- plot (replacement) ---
    from matplotlib.patches import Patch

    # we collected all_results as (board, model1, model2, wr_model1)
    # pick out only this board:
    b = boards[0]
    rel = [r for r in all_results if r[0] == b]

    # define exactly the order and pairs you want
    matchups = [
        ("PPO",  "NFSP"),
        ("AZ",   "PPO"),
        ("NFSP", "AZ")
    ]
    # color map
    cmap = {"PPO": "C0", "NFSP": "C1", "AZ": "C2"}

    labels      = []
    wins_bottom = []
    wins_top    = []
    for bot, top in matchups:
        # try direct lookup
        wr_bot = None
        for _, m1, m2, wr in rel:
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


    x     = np.arange(len(labels))
    width = 0.6

    fig, ax = plt.subplots()
    # bottom bars
    for i, (bot, _) in enumerate(matchups):
        ax.bar(
            x[i], wins_bottom[i], width,
            color=cmap[bot]
        )
    # top bars
    for i, (_, top) in enumerate(matchups):
        ax.bar(
            x[i], wins_top[i], width,
            bottom=wins_bottom[i],
            color=cmap[top]
        )

    # annotate
    for i in range(len(labels)):
        # bottom annotation
        ax.text(
            x[i], wins_bottom[i]/2,
            f"{wins_bottom[i]*100:.0f}%",
            ha="center", va="center", color="white", fontweight="bold"
        )
        # top annotation
        ax.text(
            x[i],
            wins_bottom[i] + wins_top[i]/2,
            f"{wins_top[i]*100:.0f}%",
            ha="center", va="center", color="white", fontweight="bold"
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Win rate")
    ax.set_title(f"{b}×{b} Round-robin results")

    # custom legend
    legend_handles = [
        Patch(color=cmap["PPO"],  label="PPO"),
        Patch(color=cmap["NFSP"], label="NFSP"),
        Patch(color=cmap["AZ"],   label="AlphaZero"),
    ]
    ax.legend(handles=legend_handles, loc="upper right")

    plt.tight_layout()
    png_out = os.path.join(OUTPUT_DIR, f"round_robin_{b}x{b}.png")
    plt.savefig(png_out)
    print(f"Plot → {png_out}")



if __name__ == "__main__":
    main()
