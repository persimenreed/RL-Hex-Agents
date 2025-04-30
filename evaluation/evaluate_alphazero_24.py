#!/usr/bin/env python3
import os
import sys
import tensorflow.compat.v1 as tf
import numpy as np
import pyspiel

tf.disable_v2_behavior()

SCRIPT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
sys.path.insert(0, PROJECT_ROOT)

from open_spiel.python.algorithms import mcts

def evaluate_vs_mcts(game, agent, mcts_bot, pid=0, num_games=100):
    wins = 0
    for i in range(num_games):
        state = game.new_initial_state()
        swap  = (i >= num_games // 2)
        while not state.is_terminal():
            cur = state.current_player()
            is_agent = (cur == pid and not swap) or (cur != pid and swap)
            move = agent.act(state) if is_agent else mcts_bot.act(state)
            state.apply_action(move)
        winner = state.returns()[pid] if not swap else state.returns()[1-pid]
        if winner > 0:
            wins += 1
    return wins / num_games

class EvalMCTSBot:
    def __init__(self, game, sims=50):
        self.bot = mcts.MCTSBot(
            game, 1.0, sims,
            mcts.RandomRolloutEvaluator(),
            solve=False,
            child_selection_fn=mcts.SearchNode.puct_value,
            verbose=False,
            dont_return_chance_node=True
        )
    def act(self, state):
        return self.bot.mcts_search(state).best_child().action

class EvalAlphaZeroAgent:
    def __init__(self, sess, ckpt_prefix):
        saver = tf.train.import_meta_graph(ckpt_prefix + ".meta", clear_devices=True)
        saver.restore(sess, ckpt_prefix)
        g = tf.get_default_graph()
        self.obs_ph = g.get_tensor_by_name("input:0")
        self.legals_ph = g.get_tensor_by_name("legals_mask:0")
        self.train_ph = g.get_tensor_by_name("training:0")
        self.probs_t = g.get_tensor_by_name("policy_softmax:0")
        self.sess = sess

    def act(self, state):
        obs = state.observation_tensor()
        mask = np.array(state.legal_actions_mask(), dtype=np.float32)
        feed = {
            self.obs_ph: [obs],
            self.legals_ph: [mask],
            self.train_ph: False,
        }
        probs = self.sess.run(self.probs_t, feed_dict=feed)[0]
        probs = probs * mask
        return int(np.argmax(probs))

def main():
    board_size = 8
    checkpoint_number = 200
    sims = {5: 50, 8: 50, 11: 50}[board_size]

    print(f"\nEvaluating AlphaZero checkpoint--{checkpoint_number} vs MCTS on {board_size}x{board_size} board (10 runs)")

    game = pyspiel.load_game(f"hex(board_size={board_size})")
    mcts_bot = EvalMCTSBot(game, sims)

    ckpt_dir = os.path.join(PROJECT_ROOT, "weights", "alpha_zero", f"alpha_zero_hex_{board_size}_24h")
    ckpt_prefix = os.path.join(ckpt_dir, f"checkpoint-{checkpoint_number}")

    if not os.path.exists(ckpt_prefix + ".meta"):
        raise FileNotFoundError(f"Checkpoint {ckpt_prefix}.meta not found.")

    tf.reset_default_graph()
    with tf.Session() as sess:
        agent = EvalAlphaZeroAgent(sess, ckpt_prefix)

        results = []
        for i in range(10):
            print(f"\nRun {i+1}/10:")
            win_rate = evaluate_vs_mcts(game, agent, mcts_bot, pid=0)
            print(f"  Win rate vs MCTS: {win_rate:.3f}")
            results.append(win_rate)

        avg = sum(results) / len(results)
        print(f"\nAverage win rate over 10 runs: {avg:.3f}")


if __name__ == "__main__":
    main()
