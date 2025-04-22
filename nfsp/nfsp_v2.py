import os
import random
import time
import csv
import numpy as np
import pyspiel
import tensorflow.compat.v1 as tf
from open_spiel.python.algorithms.nfsp import NFSP
from open_spiel.python import rl_environment

tf.disable_v2_behavior()

# ───────────────── Configuration ──────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

TRAIN_HOURS           = 4
MAX_TRAIN_SEC         = TRAIN_HOURS * 3600.0
BOARD_SIZES           = [5, 8, 11]
ETA_START, ETA_END    = 0.9, 0.1
EPS_START, EPS_END    = 1.0, 0.05

SNAPSHOT_INTERVAL_SEC = 600
LOG_INTERVAL_SEC      = 60.0

def compute_epsilon(elapsed):
    half = MAX_TRAIN_SEC / 2.0
    if elapsed < half:
        return EPS_START + (elapsed/half)*(EPS_END - EPS_START)
    return EPS_END

for b in BOARD_SIZES:
    print(f"Starting NFSP training with board size {b}×{b}")
    root     = f"./weights/nfsp/nfsp_hex_{b}_{TRAIN_HOURS}h"
    os.makedirs(root, exist_ok=True)
    meta_csv = os.path.join(root, "metadata.csv")

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
                hidden_layers_sizes=[64,64],
                reservoir_buffer_capacity=100_000,
                anticipatory_param=ETA_START,
                batch_size=32,
                rl_learning_rate=0.01,
                sl_learning_rate=0.01,
                min_buffer_size_to_learn=1000,
                learn_every=64,
                optimizer_str="adam",
                epsilon_start=EPS_START,
                epsilon_decay_duration=1
            )
            agents.append(ag)
        sess.run(tf.global_variables_initializer())

        start       = time.time()
        next_snap   = start + SNAPSHOT_INTERVAL_SEC
        next_log    = start + LOG_INTERVAL_SEC
        episode     = 0

        while True:
            episode += 1
            now     = time.time()
            elapsed = now - start
            frac    = min(1.0, elapsed / MAX_TRAIN_SEC)
            eta     = ETA_START + frac * (ETA_END - ETA_START)
            eps     = compute_epsilon(elapsed)

            for ag in agents:
                ag._anticipatory_param = eta
                ag._rl_agent._epsilon  = eps
                ag._sample_episode_policy()

            ts  = env.reset()
            p   = ts.observations['current_player']
            out = agents[p].step(ts, is_evaluation=True)
            ts  = env.step([out.action])

            while not ts.last():
                p   = ts.observations['current_player']
                out = agents[p].step(ts)
                ts  = env.step([out.action])

            # per-minute log
            if now >= next_log:
                print(f"[Episode {episode}] board={b}x{b} elapsed={elapsed:.1f}s / {MAX_TRAIN_SEC:.1f}s  ε={eps:.2f}")
                next_log += LOG_INTERVAL_SEC

            # snapshot + metadata every 10 min
            if now >= next_snap:
                writer.writerow([int(elapsed), episode, eta, eps])
                f.flush()
                for pid, ag in enumerate(agents):
                    ag.save(root)
                    saver = tf.train.Saver()
                    ckpt  = os.path.join(root, f"nfsp_hex_p{pid}_{int(elapsed)}s.ckpt")
                    saver.save(sess, ckpt)
                    saver.export_meta_graph(ckpt + ".meta", as_text=True)
                next_snap += SNAPSHOT_INTERVAL_SEC

            if elapsed >= MAX_TRAIN_SEC:
                print(f"Completed NFSP board {b}×{b}.")
                break
