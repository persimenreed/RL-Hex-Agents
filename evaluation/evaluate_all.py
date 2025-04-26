#!/usr/bin/env python3
import os
import sys

SCRIPT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import evaluate_ppo, evaluate_nfsp, evaluate_alphazero

def main():
    print("=== Running PPO evaluation ===")
    evaluate_ppo.main()

    print("\n=== Running NFSP evaluation ===")
    evaluate_nfsp.main()

    print("\n=== Running AlphaZero evaluation ===")
    evaluate_alphazero.main()

if __name__ == "__main__":
    main()
