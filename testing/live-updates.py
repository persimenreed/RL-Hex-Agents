#python3 testing/live-updates.py ./weights/alpha_zero/alpha_zero_hex_5_100 checkpoint-100 dummy

import pyspiel
import random
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
from matplotlib.patches import RegularPolygon

# Optional AlphaZero imports
try:
    from open_spiel.python.algorithms.alpha_zero import model as model_lib
    from open_spiel.python.algorithms.alpha_zero import evaluator as evaluator_lib
except ImportError:
    model_lib = None
    evaluator_lib = None


def get_board_from_state(state, size):
    board = np.zeros((size, size), dtype=int)
    history = state.history()
    current = 0
    for action in history:
        row = action // size
        col = action % size
        player = (current % 2) + 1
        board[row][col] = player
        current += 1
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

    offset = 0.6
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
        shift = -0.6 if col == 0 else 0.6
        x_start += shift
        x_end += shift
        ax.plot([x_start, x_end], [y_start, y_end], color='blue', linewidth=3)

    ax.set_xlim(-1, size + size * 0.75)
    ax.set_ylim(-size - 1.5, 1.5)
    ax.axis('off')


def load_model_evaluator(game, checkpoint_dir, checkpoint_name, model_type="mlp", width=64, depth=2):
    if model_lib is None or evaluator_lib is None:
        print("AlphaZero modules not available.")
        return None

    import os

    full_prefix = os.path.join(checkpoint_dir, checkpoint_name)
    required_files = [f"{full_prefix}.meta", f"{full_prefix}.index", f"{full_prefix}.data-00000-of-00001"]

    if not all(os.path.exists(f) for f in required_files):
        print(f"Checkpoint files not found for: {checkpoint_name}")
        print("Expected files:")
        for f in required_files:
            print(" -", f)
        return None

    shape = game.observation_tensor_shape()
    num_actions = game.num_distinct_actions()

    model = model_lib.Model.build_model(
        model_type, shape, num_actions,
        nn_width=width, nn_depth=depth,
        weight_decay=1e-4, learning_rate=0.01,
        path=checkpoint_dir
    )

    print(f"Loading checkpoint: {checkpoint_name}")
    model.load_checkpoint(os.path.join(checkpoint_dir, checkpoint_name))

    return evaluator_lib.AlphaZeroEvaluator(game, model)


def test_hex_visual(checkpoint_dir, checkpoint_name_1, checkpoint_name_2, fig, ax):
    game = pyspiel.load_game("hex(board_size=5)")
    state = game.new_initial_state()
    board_size = game.get_parameters()["board_size"]

    evaluator_1 = load_model_evaluator(game, checkpoint_dir, checkpoint_name_1)
    evaluator_2 = load_model_evaluator(game, checkpoint_dir, checkpoint_name_2)
    evaluators = [evaluator_1, evaluator_2]

    print("Starting Hex game...")
    move_count = 0
    while not state.is_terminal():
        current_player = state.current_player()
        legal_actions = state.legal_actions()

        evaluator = evaluators[current_player]
        if evaluator:
            policy = evaluator.prior(state)
            policy.sort(key=lambda x: -x[1])
            action = policy[0][0]
        else:
            action = random.choice(legal_actions)

        state.apply_action(action)

        board = get_board_from_state(state, board_size)
        draw_hex_board(board, ax, move_count + 1)
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.05)

        move_count += 1

    print("\nGame finished!")
    print("Returns:", state.returns())
    print("Final state (text view):")
    print(state)

    time.sleep(2)


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python3 live-updates.py <checkpoint_dir> <checkpoint_name_1> <checkpoint_name_2>")
        sys.exit(1)

    checkpoint_dir = sys.argv[1]
    checkpoint_name_1 = sys.argv[2]
    checkpoint_name_2 = sys.argv[3]

    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.show()
    plt.pause(0.1)

    for match in range(5):
        print(f"\n=== Match {match + 1} of 5 ===")
        test_hex_visual(checkpoint_dir, checkpoint_name_1, checkpoint_name_2, fig, ax)
        if match < 4:
            print("Waiting 2 seconds before next match...\n")

    print("All matches complete!")
    plt.ioff()
    plt.show()