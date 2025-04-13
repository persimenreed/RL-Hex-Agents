from open_spiel.python.algorithms.alpha_zero import alpha_zero
from open_spiel.python.algorithms.alpha_zero.alpha_zero import Config
import os

# only cpu
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

board_size = 11
train_time = 30

output_path = f"./weights/alpha_zero/alpha_zero_hex_{board_size}_{train_time}"
os.makedirs(output_path, exist_ok=True)

config = Config(
    game=f"hex(board_size={board_size})",
    path=output_path,
    learning_rate=0.003,
    weight_decay=1e-4,
    train_batch_size=128,
    replay_buffer_size=30000,
    replay_buffer_reuse=3,
    max_steps=10000,
    checkpoint_freq=1,
    actors=2,
    evaluators=1,
    evaluation_window=10,
    eval_levels=1,

    uct_c=1.0,
    max_simulations=100,
    policy_alpha=0.3,
    policy_epsilon=0.25,
    temperature=1.0,
    temperature_drop=10,

    nn_model="mlp",
    nn_width=128,
    nn_depth=3,
    observation_shape=None,
    output_size=None,

    quiet=False,
)

if __name__ == "__main__":
    alpha_zero.alpha_zero(config)
