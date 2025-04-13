import os
import random
import time
import numpy as np
import pyspiel
import tensorflow.compat.v1 as tf
from open_spiel.python.algorithms.nfsp import NFSP
from open_spiel.python import rl_environment

tf.disable_v2_behavior()
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Training time in minutes
train_time = 30
board_size = 11

# Maximum training time in seconds
max_train_time = train_time * 60

# === Setup ===
game = pyspiel.load_game(f"hex(board_size={board_size})")
env = rl_environment.Environment(game, include_full_state=True)
num_players = env.num_players
info_spec = env.observation_spec()["info_state"]

# Output path for the NFSP checkpoint
checkpoint_dir = f"./weights/nfsp/nfsp_hex_{board_size}_{train_time}"
os.makedirs(checkpoint_dir, exist_ok=True)

sess = tf.Session()
agents = []

state_representation_size = game.observation_tensor_size()
num_actions = game.num_distinct_actions()

# Create two NFSP agents with fixed player_ids.
for pid in range(num_players):
    agent = NFSP(
        session=sess,
        player_id=pid,
        state_representation_size=state_representation_size,
        num_actions=num_actions,
        hidden_layers_sizes=[64, 64],
        reservoir_buffer_capacity=100000,
        anticipatory_param=0.1,
        batch_size=32,
        rl_learning_rate=0.01,
        sl_learning_rate=0.01,
        min_buffer_size_to_learn=1000,
        learn_every=64,
        optimizer_str="sgd",
    )
    agents.append(agent)

sess.run(tf.global_variables_initializer())

# --- Helper: Swap transformation for observations ---
def swap_observations(observations):
    swapped = observations.copy()  # shallow copy for safety
    info_state = np.array(observations["info_state"])
    half = info_state.shape[0] // 2
    # Swap the first and second halves.
    swapped_info_state = np.concatenate([info_state[half:], info_state[:half]])
    swapped["info_state"] = swapped_info_state.tolist()
    return swapped

# === Training loop based on elapsed time ===
start_time = time.time()
episode = 0

while True:
    episode += 1
    time_step = env.reset()
    # Decide whether to swap the perspective this episode.
    swap_flag = random.choice([True, False])
    
    while not time_step.last():
        current_player = time_step.observations["current_player"]
        if swap_flag:
            # Create a modified time_step using _replace() method.
            modified_time_step = time_step._replace(
                observations=swap_observations(time_step.observations)
            )
            action_output = agents[current_player].step(modified_time_step)
        else:
            action_output = agents[current_player].step(time_step)
        time_step = env.step([action_output.action])
    
    for agent in agents:
        agent.step(time_step)
    
    if episode % 100 == 0:
        elapsed = time.time() - start_time
        print(f"Episode {episode}: elapsed time = {elapsed:.2f}s")
    
    if time.time() - start_time >= max_train_time:
        print(f"{train_time} minutes have elapsed. Ending training at episode {episode}.")
        break

# === Save Weights ===
for agent in agents:
    agent.save(checkpoint_dir)

# ---- Export the meta graph for visualization ----
saver = tf.train.Saver()
meta_graph_path = os.path.join(checkpoint_dir, "nfsp_model.meta")
saver.export_meta_graph(meta_graph_path, as_text=True)

print("NFSP training complete.")
