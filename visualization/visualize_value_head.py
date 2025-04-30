#!/usr/bin/env python3

import os
import sys
import time
import random

import pyspiel
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon

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
        x0 = 0 + 0.5 * row
        x1 = (size - 1) + 0.5 * row
        y  = -row + (offset if row == 0 else -offset)
        ax.plot([x0, x1], [y, y], color='red', linewidth=3)
    for col in [0, size - 1]:
        x0 = col + (-0.6 if col == 0 else 0.6)
        y0 = 0
        x1 = col + (0.5 * (size - 1)) + (-0.6 if col == 0 else 0.6)
        y1 = -(size - 1)
        ax.plot([x0, x1], [y0, y1], color='blue', linewidth=3)

    ax.set_xlim(-1, size + size * 0.75)
    ax.set_ylim(-size - 1.5, 1.5)
    ax.axis('off')


def load_model_evaluator(game, checkpoint_dir, checkpoint_name,
                         model_type="mlp", width=64, depth=2):
    if model_lib is None or evaluator_lib is None:
        print("AlphaZero modules not available.")
        return None

    full_prefix = os.path.join(checkpoint_dir, checkpoint_name)
    required = [f"{full_prefix}.meta",
                f"{full_prefix}.index",
                f"{full_prefix}.data-00000-of-00001"]
    if not all(os.path.exists(f) for f in required):
        print(f"Checkpoint files not found for: {checkpoint_name}")
        return None

    obs_shape   = game.observation_tensor_shape()
    num_actions = game.num_distinct_actions()

    model = model_lib.Model.build_model(
        model_type, obs_shape, num_actions,
        nn_width=width, nn_depth=depth,
        weight_decay=1e-4, learning_rate=0.01,
        path=checkpoint_dir
    )
    print(f"Loading checkpoint: {checkpoint_name}")
    model.load_checkpoint(full_prefix)
    return evaluator_lib.AlphaZeroEvaluator(game, model)


def draw_board_with_values(state, evaluator, board_size, ax):
    board = get_board_from_state(state, board_size)
    draw_hex_board(board, ax, len(state.history()))

    if evaluator is None:
        plt.draw()
        plt.pause(0.001)
        return

    for action in state.legal_actions():
        next_state = state.child(action)

        if next_state.is_terminal():
            v = next_state.returns()[0]
        else:
            vals = evaluator.evaluate(next_state)
            v    = vals[0] 

        row, col = divmod(action, board_size)
        x = col + 0.5 * row
        y = -row

        ax.text(
            x, y,
            f"{v:+.2f}",
            ha="center", va="center",
            color="black", fontsize=8, fontweight="bold"
        )

    ax.set_xlim(-1, board_size + board_size * 0.75)
    ax.set_ylim(-board_size - 1.5, 1.5)
    plt.draw()
    plt.pause(0.001)



def test_hex_visual(checkpoint_dir, ckpt1, ckpt2, fig, ax, board_size):
    game = pyspiel.load_game(f"hex(board_size={board_size})")
    state = game.new_initial_state()
    ev1 = load_model_evaluator(game, checkpoint_dir, ckpt1)
    ev2 = load_model_evaluator(game, checkpoint_dir, ckpt2)
    evaluators = [ev1, ev2]
    
    draw_board_with_values(state, ev1, board_size, ax)
    plt.pause(3)

    print("Starting Hex game with live value visualization...")
    move_count = 0
    while not state.is_terminal():
        cur = state.current_player()
        if evaluators[cur]:
            policy = evaluators[cur].prior(state)
            policy.sort(key=lambda x: -x[1])
            action = policy[0][0]
        else:
            action = random.choice(state.legal_actions())

        state.apply_action(action)
        move_count += 1

        draw_board_with_values(state, ev1, board_size, ax)
        plt.pause(0.1)

    draw_board_with_values(state, ev1, board_size, ax)
    print("Game finished, returns:", state.returns())
    plt.pause(1.0)


def main():
    if len(sys.argv) < 4:
        print("Usage: python3 visualize_state_values.py"
              " <checkpoint_dir> <ckpt1> <ckpt2>")
        sys.exit(1)

    checkpoint_dir = sys.argv[1]
    ckpt1          = sys.argv[2]
    ckpt2          = sys.argv[3]
    board_size = 8

    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.show()
    plt.pause(0.1)

    for match in range(5):
        print(f"\n=== Match {match+1}/5 ===")
        test_hex_visual(checkpoint_dir, ckpt1, ckpt2, fig, ax, board_size)
        if match < 4:
            print("Waiting 2s before next match...\n")
            time.sleep(2)

    plt.ioff()
    plt.show()


if __name__ == '__main__':
    main()
