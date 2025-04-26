#!/usr/bin/env python3
import os
import random

from flask import Flask, render_template, request, session, jsonify
import pyspiel

# AlphaZero imports
from open_spiel.python.algorithms.alpha_zero import model as model_lib
from open_spiel.python.algorithms.alpha_zero import evaluator as evaluator_lib

app = Flask(__name__)
app.secret_key = 'hex_game_secret_key'

# Game configuration
BOARD_SIZE = 8
game = pyspiel.load_game(f"hex(board_size={BOARD_SIZE})")

# Locate your checkpoint folder
BASE_DIR       = os.path.dirname(__file__)       # .../rl_hex_agents/testing
weights_root   = os.path.abspath(
    os.path.join(BASE_DIR, '..', 'weights', 'alpha_zero')
)
checkpoint_dir = os.path.join(weights_root, 'alpha_zero_hex_8_4h')
checkpoint_name= 'checkpoint-58'

def load_model_evaluator(game, checkpoint_dir, checkpoint_name,
                         model_type="mlp", width=64, depth=2):
    full = os.path.join(checkpoint_dir, checkpoint_name)
    for ext in (".meta", ".index", ".data-00000-of-00001"):
        if not os.path.exists(full + ext):
            print(f"[!] missing {full+ext}")
            return None
    model = model_lib.Model.build_model(
        model_type,
        game.observation_tensor_shape(),
        game.num_distinct_actions(),
        nn_width=width, nn_depth=depth,
        weight_decay=1e-4, learning_rate=0.01,
        path=checkpoint_dir
    )
    print(f"Loading AlphaZero ckpt: {checkpoint_name}")
    model.load_checkpoint(full)
    return evaluator_lib.AlphaZeroEvaluator(game, model)

# Try to load once at startup
evaluator = load_model_evaluator(game, checkpoint_dir, checkpoint_name)
if evaluator is None:
    print("Failed to load AlphaZero model; falling back to random moves.")

@app.route('/')
def index():
    if 'moves' not in session:
        session['moves'] = []
        session.modified = True
    return render_template('index.html')

@app.route('/make_move', methods=['POST'])
def make_move():
    # 1) Parse the move
    try:
        user_move = int(request.form.get('move', -1))
    except (TypeError, ValueError):
        return jsonify(error='Invalid move format'), 400

    # 2) Handle reset
    if user_move < 0:
        session['moves'] = []
        session.modified = True
        state     = game.new_initial_state()
        occupancy = get_occupancy(state)
        if evaluator:
            value    = float(evaluator.evaluate(state)[0])
            raw      = 0.5 * (value * 0.8 + 1.0)
            p1       = max(0.0, min(1.0, raw))
            win_probs = [int(p1 * 100), 100 - int(p1 * 100)]
            policy    = { str(a): float(p) for a, p in evaluator.prior(state) }
        else:
            win_probs = [50, 50]
            policy    = {}
        return jsonify(
            occupancy=occupancy,
            win_probs=win_probs,
            policy=policy,
            game_over=False
        )

    # 3) Rebuild state from history
    state = game.new_initial_state()
    for m in session.get('moves', []):
        state.apply_action(m)

    # 4) Validate user’s move
    if user_move not in state.legal_actions():
        return jsonify(error='Illegal move'), 400

    # 5) Apply user’s move
    state.apply_action(user_move)
    session['moves'].append(user_move)
    session.modified = True

    # 6) If terminal, just return
    if state.is_terminal():
        occupancy = get_occupancy(state)
        # compute win_probs & policy at terminal if you like...
        return jsonify(
            occupancy=occupancy,
            win_probs=[50,50],
            policy={},
            game_over=True
        )

    # 7) AI move (player 2)
    ai_move = random.choice(state.legal_actions())
    state.apply_action(ai_move)
    session['moves'].append(ai_move)
    session.modified = True

    # 8) Build response
    occupancy = get_occupancy(state)

    if evaluator:
        value = float(evaluator.evaluate(state)[0])
        raw   = 0.5 * (value * 0.8 + 1.0)
        p1    = max(0.0, min(1.0, raw))
        win_probs = [int(p1*100), 100-int(p1*100)]
        policy    = { str(a): float(p) for a, p in evaluator.prior(state) }
    else:
        win_probs = [50, 50]
        policy    = {}

    return jsonify(
        occupancy=occupancy,
        win_probs=win_probs,
        policy=policy,
        game_over=state.is_terminal()
    )


def get_occupancy(state):
    hist = state.history()
    occ  = []
    for turn, action in enumerate(hist):
        r, c   = divmod(action, BOARD_SIZE)
        player = 1 if (turn % 2)==0 else 2
        occ.append([r, c, player])
    return occ

if __name__ == '__main__':
    app.run(debug=True)
