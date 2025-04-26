import tensorflow as tf
import os
from open_spiel.python.algorithms.alpha_zero.alpha_zero import alpha_zero, Config


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    if tf.config.list_physical_devices('GPU'):
        print("Running on GPU")
    else:
        print("Running on CPU")

    # Training config
    TRAIN_HOURS = 4
    #BOARD_SIZES = [5]
    BOARD_SIZES = [8, 11]

    BOARD_PARAMS = {
        5: dict(nn_model="mlp", nn_width=64, nn_depth=4, max_simulations=100, replay_buffer_reuse=5, train_batch_size=128, learning_rate=0.01),
        8: dict(nn_model="mlp", nn_width=128, nn_depth=4, max_simulations=300, replay_buffer_reuse=5, train_batch_size=128, learning_rate=0.01),
        11: dict(nn_model="mlp", nn_width=256, nn_depth=4, max_simulations=800, replay_buffer_reuse=7, train_batch_size=512, learning_rate=0.01),
    }

    for b in BOARD_SIZES:
        p = BOARD_PARAMS[b]
        output_path = f"./weights/alpha_zero/alpha_zero_hex_{b}_{TRAIN_HOURS}h"
        os.makedirs(output_path, exist_ok=True)

        config = Config(
            game=f"hex(board_size={b})",
            path=output_path,
            learning_rate=p["learning_rate"],
            weight_decay=1e-4,
            train_batch_size=p["train_batch_size"],
            replay_buffer_size=50000,
            replay_buffer_reuse=p["replay_buffer_reuse"],
            max_steps=165,
            checkpoint_freq=1,
            actors=2,
            evaluators=1,
            evaluation_window=20,
            eval_levels=1,
            uct_c=1.0,
            max_simulations=p["max_simulations"],
            policy_alpha=0.3,
            policy_epsilon=0.25,
            temperature=1.0,
            temperature_drop=10,
            nn_model=p["nn_model"],
            nn_width=p["nn_width"],
            nn_depth=p["nn_depth"],
            observation_shape=None,
            output_size=None,
            quiet=True,
        )

        print(f"Starting AlphaZero training with board size {b}×{b}")
        alpha_zero(config)

if __name__ == "__main__":
    main()
