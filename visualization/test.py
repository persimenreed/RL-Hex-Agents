#!/usr/bin/env python3
"""
visualize_state_values.py

Live‐update Hex board with:
  - Top bar: Player 1 win‐prob (value head).
  - Board: hex‐cells colored by occupancy,
     plus a policy‐heat overlay on empty cells (policy head).
Usage:
    python3 visualize_state_values.py <checkpoint_dir> <ckpt1> <ckpt2>
"""
import os
import sys
import time
import random

import pyspiel
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib.colors import Normalize
from matplotlib import cm

# Attempt to import AlphaZero
try:
    from open_spiel.python.algorithms.alpha_zero import model as model_lib
    from open_spiel.python.algorithms.alpha_zero import evaluator as evaluator_lib
except ImportError:
    model_lib = None
    evaluator_lib = None


def load_model_evaluator(game, checkpoint_dir, checkpoint_name,
                         model_type="mlp", width=64, depth=2):
    """Load an AlphaZeroEvaluator from a TensorFlow checkpoint."""
    full = os.path.join(checkpoint_dir, checkpoint_name)
    for ext in (".meta", ".index", ".data-00000-of-00001"):
        if not os.path.exists(full + ext):
            print(f"[!] missing {full+ext}")
            return None
    obs_shape   = game.observation_tensor_shape()
    num_actions = game.num_distinct_actions()
    model = model_lib.Model.build_model(
        model_type, obs_shape, num_actions,
        nn_width=width, nn_depth=depth,
        weight_decay=1e-4, learning_rate=0.01,
        path=checkpoint_dir
    )
    print(f"Loading AlphaZero ckpt: {checkpoint_name}")
    model.load_checkpoint(full)
    return evaluator_lib.AlphaZeroEvaluator(game, model)


def draw_value_bar(ax, state, evaluator):
    """Draw the top horizontal bar showing Player 1's win‐prob."""
    ax.clear()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    if evaluator is None:
        return
    if state.is_terminal():
        p1 = 0.5 * ( state.returns()[0] + 1.0 )
    else:
        vals = evaluator.evaluate(state)
        p1   = 0.5 * ( vals[0] + 1.0 )
    # get value‐head from Player1's POV at current state
    
    # draw two segments
    ax.barh(0, p1, left=0,   height=1, color="tab:red")
    ax.barh(0, 1-p1, left=p1, height=1, color="tab:blue")
    ax.text(0.5, 0.5, f"P1 win {int(p1*100)}%", 
            ha="center", va="center", color="white", weight="bold")
    ax.axis("off")


def draw_policy_hex(ax, state, evaluator, board_size):
    """Draw the hex board with occupancy and policy‐heat on empties."""
    ax.clear()
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])

    # build policy map, action→prob
    policy_map = {}
    if evaluator is not None and not state.is_terminal():
        prior = evaluator.prior(state)      # list of (action, prob)
        tot   = sum(p for _,p in prior)
        for a,p in prior:
            policy_map[a] = p/tot

    # a colormap for policy
    norm = Normalize(vmin=0, vmax=1)
    cmap = cm.get_cmap("Greens")

    history = state.history()
    occupied = set(history)

    for row in range(board_size):
        for col in range(board_size):
            x = col + 0.5*row
            y = -row
            idx = row*board_size + col
            if idx in occupied:
                # which player?
                turn = history.index(idx)
                player = (turn%2)+1
                face = "red" if player==1 else "blue"
            else:
                # empty: heat by policy
                p = policy_map.get(idx, 0.0)
                face = cmap(norm(p))
            hexagon = RegularPolygon(
                (x,y), numVertices=6, radius=0.5,
                orientation=np.radians(30),
                facecolor=face, edgecolor="gray"
            )
            ax.add_patch(hexagon)

    # goal lines
    offset = 0.6
    # red goal (top/bottom)
    for r in (0, board_size-1):
        x0 = 0 + 0.5*r
        x1 = (board_size-1) + 0.5*r
        y  = -r + (offset if r==0 else -offset)
        ax.plot([x0,x1],[y,y], color="red", lw=3)
    # blue goal (left/right)
    for c in (0, board_size-1):
        x0 = c + (-0.6 if c==0 else 0.6)
        y0 = 0
        x1 = c + 0.5*(board_size-1) + (-0.6 if c==0 else 0.6)
        y1 = -(board_size-1)
        ax.plot([x0,x1],[y0,y1], color="blue", lw=3)

    ax.set_xlim(-1, board_size + board_size*0.75)
    ax.set_ylim(-board_size-1.5, 1.5)


def test_hex_visual(checkpoint_dir, ck1, ck2, fig, ax_bar, ax_board, board_size):
    game = pyspiel.load_game(f"hex(board_size={board_size})")
    state = game.new_initial_state()
    ev1 = load_model_evaluator(game, checkpoint_dir, ck1)
    ev2 = load_model_evaluator(game, checkpoint_dir, ck2)
    evaluators = [ev1, ev2]

    # initial pause so you see the empty policy‐heat + value‐bar
    draw_value_bar(ax_bar,   state, ev1)
    draw_policy_hex(ax_board, state, ev1, board_size)
    plt.pause(3.0)

    print("Starting live Hex demo…")
    while not state.is_terminal():
        cur = state.current_player()
        if evaluators[cur]:
            # greedy on policy‐head
            pr = evaluators[cur].prior(state)
            pr.sort(key=lambda x:-x[1])
            action = pr[0][0]
        else:
            action = random.choice(state.legal_actions())

        state.apply_action(action)
        # redraw both
        draw_value_bar(ax_bar,   state, ev1)
        draw_policy_hex(ax_board, state, ev1, board_size)
        plt.pause(0.1)

    print("Game over:", state.returns())
    plt.pause(1.0)


def main():
    if len(sys.argv)<4:
        print("Usage: python3 visualize_state_values.py"
              " <checkpoint_dir> <ckpt1> <ckpt2>")
        sys.exit(1)
    checkpoint_dir, ck1, ck2 = sys.argv[1:4]
    board_size = 8   # hard-coded or parse from ckpt name

    plt.ion()
    fig = plt.figure(figsize=(6,7))
    gs  = fig.add_gridspec(2,1, height_ratios=[1,5], hspace=0.1)
    ax_bar   = fig.add_subplot(gs[0,0])
    ax_board = fig.add_subplot(gs[1,0])

    for idx in range(5):
        print(f"=== Match {idx+1}/5 ===")
        test_hex_visual(checkpoint_dir, ck1, ck2, fig, ax_bar,ax_board, board_size)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()