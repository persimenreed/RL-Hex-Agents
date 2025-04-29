import os
import time
import tensorflow as tf
from multiprocessing import Process
from open_spiel.python.algorithms.alpha_zero.alpha_zero import alpha_zero, Config

def run_alpha_zero(config):
    alpha_zero(config)

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    if tf.config.list_physical_devices('GPU'):
        print("Running on GPU")
    else:
        print("Running on CPU")

    TRAIN_HOURS = 24
    TRAIN_SECONDS = (TRAIN_HOURS * 3600) + 200

    #BOARD_SIZES = [5, 8, 11]
    BOARD_SIZES = [8]

    for b in BOARD_SIZES:
        output_path = f"./weights/alpha_zero/alpha_zero_hex_{b}_{TRAIN_HOURS}h"
        os.makedirs(output_path, exist_ok=True)

        config = Config(
            game=f"hex(board_size={b})",
            path=output_path,
            learning_rate=0.001,
            weight_decay=0.0001,
            train_batch_size=128,
            replay_buffer_size=50000,
            replay_buffer_reuse=3,
            max_steps=1000000,
            checkpoint_freq=1,
            actors=32,
            evaluators=4,
            evaluation_window=20,
            eval_levels=1,
            uct_c=1.0,
            max_simulations=200,
            policy_alpha=0.3,
            policy_epsilon=0.25,
            temperature=1.0,
            temperature_drop=10,
            nn_model='mlp',
            nn_width=64,
            nn_depth=2,
            observation_shape=None,
            output_size=None,
            quiet=True,
        )

        print(f"\nStarting AlphaZero for {b}×{b} hex: {TRAIN_HOURS}h limit")
        proc = Process(target=run_alpha_zero, args=(config,))
        proc.start()

        proc.join(TRAIN_SECONDS)
        if proc.is_alive():
            print(f"Time limit reached for {b}×{b} — terminating process.")
            proc.terminate()
            proc.join()
        else:
            print(f"Finished early for {b}×{b}.")

    print("\nAll done.")

if __name__ == "__main__":
    main()
