import pyspiel
import random
import matplotlib.pyplot as plt
import numpy as np
import time
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

def draw_hex_board(board, ax, move_number):
    size = board.shape[0]
    ax.clear()
    ax.set_aspect('equal')
    ax.set_title(f"Move {move_number}", fontsize=14)

    # --- Draw hex tiles ---
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

    # --- Draw player goal lines ---

        # --- Draw player goal lines ---

    offset = 0.6  # visual spacing

    # Red (Player 1): top and bottom (horizontal)
    for row in [0, size - 1]:
        x_start = 0 + 0.5 * row
        x_end = (size - 1) + 0.5 * row
        y = -row + (offset if row == 0 else -offset)
        ax.plot([x_start, x_end], [y, y], color='red', linewidth=3)

    # Blue (Player 2): left and right (diagonals)
    for col in [0, size - 1]:
        x_start = col + (0.5 * 0)
        y_start = -0
        x_end = col + (0.5 * (size - 1))
        y_end = -(size - 1)

        # Shift line slightly outside the board
        shift = -0.6 if col == 0 else 0.6
        x_start += shift
        x_end += shift

        ax.plot([x_start, x_end], [y_start, y_end], color='blue', linewidth=3)


    ax.set_xlim(-1, size + size * 0.75)
    ax.set_ylim(-size - 1.5, 1.5)
    ax.axis('off')

def test_hex_visual():
    game = pyspiel.load_game("hex", {"board_size": 7})
    state = game.new_initial_state()
    board_size = game.get_parameters()["board_size"]

    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.show()
    plt.pause(0.1)

    print("Starting Hex game...")

    move_count = 0
    while not state.is_terminal():
        legal_actions = state.legal_actions()
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

    input("\nPress Enter to close the board...")

if __name__ == '__main__':
    test_hex_visual()
