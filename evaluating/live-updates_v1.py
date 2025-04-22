#!/usr/bin/env python3
import os
import sys
import time
import random
import numpy as np
import pyspiel
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon

# Optional AlphaZero imports
try:
    from open_spiel.python.algorithms.alpha_zero import model as model_lib
    from open_spiel.python.algorithms.alpha_zero import evaluator as evaluator_lib
except ImportError:
    model_lib = None
    evaluator_lib = None

# PyTorch for IDDQN loading
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_board_from_state(state, size):
    board = np.zeros((size, size), dtype=int)
    for t, action in enumerate(state.history()):
        r, c = divmod(action, size)
        board[r, c] = (t % 2) + 1
    return board

def draw_hex_board(board, ax, move_number):
    size = board.shape[0]
    ax.clear()
    ax.set_aspect('equal')
    ax.set_title(f"Move {move_number}", fontsize=14)

    for row in range(size):
        for col in range(size):
            x = col + 0.5 * row
            y = -row
            value = board[row][col]

            color = 'white'
            if value == 1:
                color = 'red'
            elif value == 2:
                color = 'blue'

            hexagon = RegularPolygon(
                (x, y), numVertices=6, radius=0.5,
                orientation=np.radians(30),
                facecolor=color, edgecolor='gray'
            )
            ax.add_patch(hexagon)

    offset = 0.5
    for row in [0, size - 1]:
        x_start = 0 + 0.5 * row
        x_end = (size - 1) + 0.5 * row
        y = -row + (offset if row == 0 else -offset)
        ax.plot([x_start, x_end], [y, y], color='red', linewidth=3)

    for col in [0, size - 1]:
        x_start = col
        y_start = 0
        x_end = col + (0.5 * (size - 1))
        y_end = -(size - 1)
        shift = -0.5 if col == 0 else 0.5
        x_start += shift
        x_end += shift
        ax.plot([x_start, x_end], [y_start, y_end], color='blue', linewidth=3)

    ax.set_xlim(-1, size + size * 0.75)
    ax.set_ylim(-size - 1.5, 1.5)
    ax.axis('off')

## dummy data
def load_dummy_evaluator(game, *args):
    def prior_fn(state):
        legal = state.legal_actions()
        action = random.choice(legal)
        return [(action, 1.0)]
    return type("DummyEval", (), {"prior": prior_fn})



# ─────────── AlphaZero loader ───────────
def load_alpha_zero_evaluator(game, ckpt_dir, ckpt_name):
    if model_lib is None or evaluator_lib is None:
        raise RuntimeError("AlphaZero modules not available")
    prefix = os.path.join(ckpt_dir, ckpt_name)
    for ext in (".meta", ".index", ".data-00000-of-00001"):
        if not os.path.exists(prefix + ext):
            raise FileNotFoundError(f"Missing {prefix+ext}")
    obs_shape  = game.observation_tensor_shape()
    n_actions  = game.num_distinct_actions()
    model = model_lib.Model.build_model(
        "mlp", obs_shape, n_actions,
        nn_width=64, nn_depth=2,
        weight_decay=1e-4, learning_rate=0.01,
        path=ckpt_dir
    )
    model.load_checkpoint(prefix)
    return evaluator_lib.AlphaZeroEvaluator(game, model)

# ─────────── NFSP loader ───────────
def load_nfsp_evaluator(game, ckpt_dir, _):
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    from open_spiel.python.algorithms.nfsp import NFSP

    sess = tf.Session()
    agent = NFSP(
        session=sess,
        player_id=0,
        state_representation_size=game.observation_tensor_size(),
        num_actions=game.num_distinct_actions(),
        hidden_layers_sizes=[64, 64],
        reservoir_buffer_capacity=100_000,
        anticipatory_param=0.1,
        batch_size=32,
        rl_learning_rate=0.01,
        sl_learning_rate=0.01,
        min_buffer_size_to_learn=1_000,
        learn_every=64,
        optimizer_str="adam",
    )
    agent.restore(ckpt_dir)
    def prior_fn(state):
        pid   = state.current_player()
        obs   = np.array(state.observation_tensor(pid), dtype=np.float32)
        legal = state.legal_actions()
        # force average policy
        with agent.temp_mode_as(agent.MODE.average_policy):
            a, probs = agent._act(obs, legal)
        return [(act, float(probs[act])) for act in legal]
    return type("NFSPEval", (), {"prior": prior_fn})

# ─────────── IDDQN loader ───────────
class _QNet(nn.Module):
    def __init__(self, input_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),      nn.ReLU(),
            nn.Linear(128, n_actions)
        )
    def forward(self, x):
        return self.net(x)

def load_iddqn_evaluator(game, ckpt_dir, ckpt_name):
    # allow ckpt_dir to be full .pt path
    if ckpt_dir.endswith(".pt") and not ckpt_name:
        path = ckpt_dir
    else:
        path = os.path.join(ckpt_dir, ckpt_name)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    obs_dim   = game.observation_tensor_size()
    n_actions = game.num_distinct_actions()
    net = _QNet(obs_dim, n_actions).to(device)
    net.load_state_dict(torch.load(path, map_location=device))
    net.eval()
    def prior_fn(state):
        pid   = state.current_player()
        obs   = np.array(state.observation_tensor(pid), dtype=np.float32)
        legal = state.legal_actions()
        with torch.no_grad():
            q = net(torch.from_numpy(obs).to(device).unsqueeze(0)).cpu().numpy()[0]
        return [(a, float(q[a])) for a in legal]
    return type("IDDQNEval", (), {"prior": prior_fn})

# ─────────── Dispatch by algorithm ───────────
def load_evaluator(alg, game, ckpt_dir, ckpt_name):
    alg = alg.lower()
    if alg == "alpha_zero":
        return load_alpha_zero_evaluator(game, ckpt_dir, ckpt_name)
    if alg == "nfsp":
        return load_nfsp_evaluator(game, ckpt_dir, ckpt_name)
    if alg == "iddqn":
        return load_iddqn_evaluator(game, ckpt_dir, ckpt_name)
    if alg == "dummy":
        return load_dummy_evaluator(game)
    raise ValueError(f"Unknown alg: {alg}")

# ─────────── Play one match ───────────
def play_match(game, eval1, eval2, ax):
    state    = game.new_initial_state()
    size     = game.get_parameters()["board_size"]
    move_num = 0
    while not state.is_terminal():
        pid       = state.current_player()
        evaluator = eval1 if pid == 0 else eval2
        policy    = evaluator.prior(state)
        action    = max(policy, key=lambda x: x[1])[0]
        state.apply_action(action)
        move_num += 1

        board = get_board_from_state(state, size)
        draw_hex_board(board, ax, move_num)
        plt.pause(0.05)

    board = get_board_from_state(state, size)
    draw_hex_board(board, ax, move_num)
    plt.pause(0.05)

    return state


# ─────────── Main ───────────
if __name__ == "__main__":
    # Accept either 4 or 6 args after script name
    if len(sys.argv) == 5:
        # four‑arg: alg1 fullpath1 alg2 fullpath2
        alg1, full1, alg2, full2 = sys.argv[1:]
        ckpt_dir1, ckpt_name1 = os.path.split(full1)
        ckpt_dir2, ckpt_name2 = os.path.split(full2)
    elif len(sys.argv) == 7:
        _, alg1, ckpt_dir1, ckpt_name1, alg2, ckpt_dir2, ckpt_name2 = sys.argv
    else:
        print("Usage (six‑arg): python live‑updates.py",
              "<alg1> <ckpt_dir1> <ckpt_name1>",
              "<alg2> <ckpt_dir2> <ckpt_name2>")
        print("   or (four‑arg): python live‑updates.py",
              "<alg1> <full_path_ckpt1> <alg2> <full_path_ckpt2>")
        sys.exit(1)

    game = pyspiel.load_game("hex(board_size=8)")
    print(f"Loading {alg1} from {ckpt_dir1}/{ckpt_name1}")
    eval1 = load_evaluator(alg1, game, ckpt_dir1, ckpt_name1)
    print(f"Loading {alg2} from {ckpt_dir2}/{ckpt_name2}")
    eval2 = load_evaluator(alg2, game, ckpt_dir2, ckpt_name2)

    plt.ion()
    fig, ax = plt.subplots(figsize=(6,6))

    for i in range(5):
        print(f"\n=== Match {i+1}/5 ===")
        final = play_match(game, eval1, eval2, ax)
        print("Result returns:", final.returns())
        time.sleep(3)

    plt.ioff()
    plt.close(fig)
    sys.exit(0)

