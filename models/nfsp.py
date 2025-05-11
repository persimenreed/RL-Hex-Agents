#!/usr/bin/env python3

import os
import random
import time
import csv
import numpy as np
from multiprocessing import Process

import pyspiel
import tensorflow.compat.v1 as tf
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import nfsp as nfsp_mod
from open_spiel.python.algorithms.nfsp import NFSP

# replacement for Open_Spiel NFSP _act function. I had trouble with NaN values.
def safe_act(self, info_state, legal_actions):
    info_state = np.reshape(info_state, [1, -1])
    _, avg_probs = self._session.run(
        [self._avg_policy, self._avg_policy_probs],
        feed_dict={self._info_state_ph: info_state})
    action_probs = avg_probs[0]

    probs = np.zeros(self._num_actions, dtype=np.float64)
    probs[legal_actions] = action_probs[legal_actions]

    eps_guard = 1e-8
    probs = np.nan_to_num(probs, nan=eps_guard,
                         posinf=eps_guard, neginf=eps_guard)
    total = probs.sum()
    if total <= 0:
        probs[:] = 0.0
        probs[legal_actions] = 1.0 / len(legal_actions)
    else:
        probs /= total

    action = np.random.choice(self._num_actions, p=probs)
    return action, probs

nfsp_mod.NFSP._act = safe_act



tf.disable_v2_behavior()

# ──────────────────── CONFIGURATION ────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

TRAIN_HOURS = 4
MAX_TRAIN_SEC = TRAIN_HOURS * 3600.0

BOARD_SIZES = [5, 8, 11]
SNAPSHOT_INTERVAL = 600.0
LOG_INTERVAL = 60.0

# NFSP hyperparameters
HIDDEN_LAYERS = [64, 64]
RESERVOIR_BUFFER_CAPACITY = 50_000
ANTICIPATORY_START = 0.1
ANTICIPATORY_END = 0.01
BATCH_SIZE = 128
RL_LR = 0.005
SL_LR = 0.001
MIN_BUFFER_TO_LEARN = 1_000
LEARN_EVERY = 64

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_DURATION = MAX_TRAIN_SEC / 2.0

# reducing epsilon over time. Goes from 1 to 0.05 for the first 50% of training, 0.05 afterwards.
def compute_epsilon(elapsed):
    if elapsed < EPS_DECAY_DURATION:
        return EPS_START + (elapsed/EPS_DECAY_DURATION)*(EPS_END - EPS_START)
    return EPS_END

# reducing anticipatory over time
def compute_eta(elapsed):
    frac = min(1.0, elapsed / MAX_TRAIN_SEC)
    return ANTICIPATORY_START + frac*(ANTICIPATORY_END - ANTICIPATORY_START)
# ────────────────────────────────────────────────────────────────────────────────

def train_board(b: int):
    print(f"\n Starting NFSP hex {b}×{b}")
    root = f"./weights/nfsp/nfsp_hex_{b}_{TRAIN_HOURS}h"
    os.makedirs(root, exist_ok=True)
    meta_csv = os.path.join(root, f"metadata_{b}x{b}.csv")
    with open(meta_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["elapsed_s","episode","eta","epsilon"])

    game = pyspiel.load_game(f"hex(board_size={b})")
    env  = rl_environment.Environment(game, include_full_state=True)
    sess = tf.Session()

    agents = []
    for pid in range(env.num_players):
        ag = NFSP(
            sess, pid,
            game.observation_tensor_size(),
            game.num_distinct_actions(),
            hidden_layers_sizes=HIDDEN_LAYERS,
            reservoir_buffer_capacity=RESERVOIR_BUFFER_CAPACITY,
            anticipatory_param=ANTICIPATORY_START,
            batch_size=BATCH_SIZE,
            rl_learning_rate=RL_LR,
            sl_learning_rate=SL_LR,
            min_buffer_size_to_learn=MIN_BUFFER_TO_LEARN,
            learn_every=LEARN_EVERY,
            optimizer_str="adam",
            epsilon_start=EPS_START,
            epsilon_decay_duration=EPS_DECAY_DURATION
        )
        agents.append(ag)

    sess.run(tf.global_variables_initializer())

    start = time.time()
    next_log = start + LOG_INTERVAL
    next_snap = start + SNAPSHOT_INTERVAL
    episode = 0

    while True:
        episode += 1
        now = time.time()
        elapsed = now - start

        # compute schedules
        eps = compute_epsilon(elapsed)
        eta = compute_eta(elapsed)

        # override NFSP internals
        for ag in agents:
            ag._rl_agent._epsilon = eps
            ag._anticipatory_param = eta
            ag._sample_episode_policy()

        # run one episode
        ts = env.reset()
        p  = ts.observations["current_player"]
        out = agents[p].step(ts, is_evaluation=True)
        ts  = env.step([out.action])
        while not ts.last():
            p   = ts.observations["current_player"]
            out = agents[p].step(ts)
            ts  = env.step([out.action])

        # logging
        if now >= next_log:
            print(f"[E{episode:5d}] b={b}×{b}  elapsed={elapsed:6.0f}s / {MAX_TRAIN_SEC:.0f}s  ε={eps:.3f}  η={eta:.3f}")
            next_log += LOG_INTERVAL

        # storing weights every 10 minutes
        if now >= next_snap:
            with open(meta_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([int(elapsed), episode, eta, eps])
            for pid, ag in enumerate(agents):
                avg_saver = tf.train.Saver(ag._avg_network.variables)
                avg_path = os.path.join(root, f"avg_network_pid{pid}_{int(elapsed)}s.ckpt")
                avg_saver.save(sess, avg_path)
                q_saver = tf.train.Saver(ag._rl_agent._q_network.variables)
                q_path = os.path.join(root, f"q_network_pid{pid}_{int(elapsed)}s.ckpt")
                q_saver.save(sess, q_path)
            next_snap += SNAPSHOT_INTERVAL

        if elapsed >= MAX_TRAIN_SEC:
            print(f"Completed NFSP hex {b}×{b}")
            break

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    for b in BOARD_SIZES:
        p = Process(target=train_board, args=(b,))
        p.start()
        p.join(MAX_TRAIN_SEC + 200)
        if p.is_alive():
            print(f"Time limit reached for {b}×{b}; terminating.")
            p.terminate()
            p.join()
        else:
            print(f"Finished early for {b}×{b}.")
    print("\nAll NFSP runs done.")
