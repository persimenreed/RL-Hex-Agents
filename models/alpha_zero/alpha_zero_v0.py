from open_spiel.python.algorithms.alpha_zero import alpha_zero
from open_spiel.python.algorithms.alpha_zero.alpha_zero import Config
import os

output_path = "./weights/alpha_zero/alpha_zero_hex_5_1"
os.makedirs(output_path, exist_ok=True)

config = Config(
    #game="hex_swap(board_size=5)",
    game="hex(board_size=5)",
    path=output_path,
    learning_rate=0.01,
    weight_decay=1e-4,
    train_batch_size=32,
    replay_buffer_size=10000,
    replay_buffer_reuse=3,
    max_steps=1,
    checkpoint_freq=1,
    actors=2,
    evaluators=1,
    evaluation_window=10,
    eval_levels=1,

    uct_c=1.0,
    max_simulations=50,
    policy_alpha=0.3,
    policy_epsilon=0.25,
    temperature=1.0,
    temperature_drop=10,

    nn_model="mlp",
    nn_width=64,
    nn_depth=2,
    observation_shape=None,
    output_size=None,

    quiet=False,
)

if __name__ == "__main__":
    alpha_zero.alpha_zero(config)