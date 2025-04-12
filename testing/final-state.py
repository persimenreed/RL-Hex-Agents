#python3 testing/live-updates.py ./weights/alpha_zero/alpha_zero_hex_5 checkpoint-100 dummy

import pyspiel
import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import RegularPolygon

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

def draw_hex_board(board, ax):
    size = board.shape[0]
    ax.clear()
    ax.set_aspect('equal')
    ax.set_title("Final Board", fontsize=14)

    # Draw hexagons
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

    # Draw player goal lines
    offset = 0.6

    # Red (Player 1): top and bottom horizontal
    for row in [0, size - 1]:
        x_start = 0 + 0.5 * row
        x_end = (size - 1) + 0.5 * row
        y = -row + (offset if row == 0 else -offset)
        ax.plot([x_start, x_end], [y, y], color='red', linewidth=3)

    # Blue (Player 2): left and right diagonals
    for col in [0, size - 1]:
        x_start = col + (0.5 * 0)
        y_start = -0
        x_end = col + (0.5 * (size - 1))
        y_end = -(size - 1)

        shift = -0.6 if col == 0 else 0.6
        x_start += shift
        x_end += shift

        ax.plot([x_start, x_end], [y_start, y_end], color='blue', linewidth=3)

    ax.set_xlim(-1, size + size * 0.75)
    ax.set_ylim(-size - 1.5, 1.5)
    ax.axis('off')

def run_final_view():
    game = pyspiel.load_game("hex", {"board_size": 7})
    state = game.new_initial_state()

    # Play full game randomly
    while not state.is_terminal():
        legal_actions = state.legal_actions()
        action = random.choice(legal_actions)
        state.apply_action(action)

    print("Final state (text view):")
    print(state)
    print("Returns:", state.returns())

    board_size = game.get_parameters()["board_size"]
    board = get_board_from_state(state, board_size)

    # Plot final board
    fig, ax = plt.subplots(figsize=(6, 6))
    draw_hex_board(board, ax)
    plt.show()

if __name__ == '__main__':
    run_final_view()
