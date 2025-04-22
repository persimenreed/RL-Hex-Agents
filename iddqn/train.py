# iddqn/train.py
import os, time, random, csv
import numpy as np, torch, pyspiel
from open_spiel.python import rl_environment
from .model import IDDQNAgent, ReplayBuffer

def main():
    BOARD_SIZES = [5,8,11]
    TRAIN_HOURS = 4
    for b in BOARD_SIZES:
        print(f"Starting IDDQN training with board size {b}×{b}")
        lr      = LEARNING_RATES[b]
        buf_cap = BUFFER_SIZES[b]
        hid     = HIDDEN_SIZES[b]

        root     = f"./weights/iddqn/iddqn_hex_{b}_{TRAIN_HOURS}h"
        os.makedirs(root, exist_ok=True)
        meta_csv = os.path.join(root, f"metadata_{b}x{b}.csv")

        with open(meta_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["elapsed_s","episode","epsilon","steps","buffer_len","avg_td_loss"])

            game   = pyspiel.load_game(f"hex(board_size={b})")
            env    = rl_environment.Environment(game, include_full_state=True)
            obs_d  = game.observation_tensor_size()
            n_act  = game.num_distinct_actions()
            agents = [IDDQNAgent(obs_d, n_act, hid, lr, buf_cap)
                    for _ in range(env.num_players)]

            start     = time.time()
            next_log  = start + LOG_INTERVAL_SEC
            next_snap = start + SNAPSHOT_INTERVAL_SEC
            episode   = 0

            while True:
                episode += 1
                now     = time.time()
                elapsed = now - start
                eps     = compute_epsilon(elapsed)
                for ag in agents:
                    ag.epsilon = eps

                ts = env.reset()
                while not ts.last():
                    p   = ts.observations["current_player"]
                    act = agents[p].act(
                        ts.observations["info_state"][p],
                        ts.observations["legal_actions"][p]
                    )
                    nts = env.step([act])
                    agents[p].buf.add(
                        ts.observations["info_state"][p],
                        act,
                        nts.rewards[p],
                        nts.observations["info_state"][p],
                        nts.last()
                    )
                    agents[p].train_step()
                    ts = nts

                if now >= next_log:
                    print(f"[Episode {episode}] board={b}x{b} elapsed={elapsed:.1f}s / {MAX_TRAIN_SEC:.1f}s  ε={eps:.2f}")
                    next_log += LOG_INTERVAL_SEC

                if now >= next_snap:
                    buf_len  = len(agents[0].buf)
                    avg_loss = (agents[0].loss_sum / agents[0].loss_count
                                if agents[0].loss_count else None)
                    writer.writerow([int(elapsed), episode, eps,
                                    agents[0].steps, buf_len, avg_loss])
                    f.flush()
                    # reset loss stats
                    for ag in agents:
                        ag.loss_sum   = 0.0
                        ag.loss_count = 0

                    for i, ag in enumerate(agents):
                        torch.save(
                            ag.online.state_dict(),
                            os.path.join(root, f"agent{i}_iddqn_{int(elapsed)}s.pt")
                        )
                    next_snap += SNAPSHOT_INTERVAL_SEC

                if elapsed >= MAX_TRAIN_SEC:
                    print(f"Completed IDDQN board {b}×{b}.")
                    break

if __name__ == "__main__":
    main()
