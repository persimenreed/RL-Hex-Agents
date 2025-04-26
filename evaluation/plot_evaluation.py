#!/usr/bin/env python3

"""
plot_evaluation.py

Load evaluation CSVs from output/ and plot the training progress 
for PPO, AlphaZero, and NFSP agents on 5x5, 8x8, and 11x11 boards.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# Setup
SCRIPT_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Which files to plot
files = {
    "5x5": "eval_5.csv",
    "8x8": "eval_8.csv",
    "11x11": "eval_11.csv"
}

# Plotting settings
colors = {
    "ppo_mcts_win": "blue",
    "az_mcts_win": "green",
    "nfsp0_mcts_win": "orange",
    "nfsp1_mcts_win": "red"
}
labels = {
    "ppo_mcts_win": "PPO vs MCTS",
    "az_mcts_win": "AlphaZero vs MCTS",
    "nfsp0_mcts_win": "NFSP pid0 vs MCTS",
    "nfsp1_mcts_win": "NFSP pid1 vs MCTS"
}

# Main plotting
for board, filename in files.items():
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        print(f"Warning: {path} not found, skipping.")
        continue

    df = pd.read_csv(path)
    
    plt.figure(figsize=(10,6))
    for col in ["ppo_mcts_win", "az_mcts_win", "nfsp0_mcts_win", "nfsp1_mcts_win"]:
        if col in df.columns:
            plt.plot(df["timestamp_s"], df[col], label=labels[col], color=colors[col])

    plt.xlabel("Training Time (seconds)")
    plt.ylabel("Win Rate vs MCTS")
    plt.title(f"Agent Performance vs MCTS ({board} board)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f"evaluation_plot_{board}.png")
    plt.savefig(save_path)
    print(f"Saved plot: {save_path}")

print("Done.")
