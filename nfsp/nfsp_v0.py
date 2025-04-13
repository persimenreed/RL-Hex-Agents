import os
import pyspiel
import tensorflow.compat.v1 as tf
from open_spiel.python.algorithms.nfsp import NFSP
from open_spiel.python import rl_environment

tf.disable_v2_behavior()

episodes = 2000

# === Setup ===
game = pyspiel.load_game("hex(board_size=5)")
env = rl_environment.Environment(game, include_full_state=True)
num_players = env.num_players
info_spec = env.observation_spec()["info_state"]

# Output path for the NFSP checkpoint
checkpoint_dir = f"./weights/nfsp/nfsp_hex_5_{episodes}"
os.makedirs(checkpoint_dir, exist_ok=True)

sess = tf.Session()
agents = []

state_representation_size = game.observation_tensor_size()
num_actions = game.num_distinct_actions()

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
        min_buffer_size_to_learn=episodes,
        learn_every=64,
        optimizer_str="sgd",
    )
    agents.append(agent)

sess.run(tf.global_variables_initializer())

# === Training ===
for episode in range(1, episodes+1):
    time_step = env.reset()
    while not time_step.last():
        current_player = time_step.observations["current_player"]
        action_output = agents[current_player].step(time_step)
        time_step = env.step([action_output.action])
    for agent in agents:
        agent.step(time_step)

    if episode % 100 == 0 or episode == episodes:
        print(f"Episode {episode}/{episodes}")

# === Save Weights ===
for agent in agents:
    agent.save(checkpoint_dir)

# ---- NEW: Export the meta graph for easier visualization ----
saver = tf.train.Saver()
meta_graph_path = os.path.join(checkpoint_dir, "nfsp_model.meta")
saver.export_meta_graph(meta_graph_path, as_text=True)
print("Meta graph exported to:", meta_graph_path)

print("NFSP training complete.")