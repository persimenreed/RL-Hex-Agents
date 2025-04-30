#!/usr/bin/env python3
import os
import random

from flask import Flask, render_template, request, session, jsonify
import pyspiel

from open_spiel.python.algorithms.alpha_zero import model as model_lib
from open_spiel.python.algorithms.alpha_zero import evaluator as evaluator_lib

app = Flask(__name__)
app.secret_key = 'hex_game_secret_key'

BOARD_SIZE = 8
game = pyspiel.load_game(f"hex(board_size={BOARD_SIZE})")

BASE_DIR = os.path.dirname(__file__)
weights_root = os.path.abspath(
    os.path.join(BASE_DIR, '..', 'evaluation', 'best_weight', 'alpha_zero')
)
checkpoint_dir = os.path.join(weights_root, 'hex_8_24h')
checkpoint_name = 'checkpoint-200'

def load_model_evaluator(game, checkpoint_dir, checkpoint_name,
                         model_type="mlp", width=128, depth=3):
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
        weight_decay=0.0001, learning_rate=0.001,
        path=checkpoint_dir
    )
    print(f"Loading AlphaZero ckpt: {checkpoint_name}")
    model.load_checkpoint(full)
    return evaluator_lib.AlphaZeroEvaluator(game, model)

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
    try:
        user_move = int(request.form.get('move', -1))
    except (TypeError, ValueError):
        return jsonify(error='Invalid move format'), 400

    # RESET / NEW GAME
    if user_move < 0:
        session['moves'] = []
        session.modified = True
        state = game.new_initial_state()
        occupancy = get_occupancy(state)

        if evaluator:
            # global win-prob
            v0  = float(evaluator.evaluate(state)[0])
            raw = max(0.0, min(1.0, (v0 + 1.0) / 2.0))
            win_probs = [int(raw*100), 100-int(raw*100)]
            policy    = { str(a): float(p) for a,p in evaluator.prior(state) }

            # per-action *raw* value in [-1,1]
            value_map = {}
            for a in state.legal_actions():
                child = state.clone()
                child.apply_action(a)
                if child.is_terminal():
                    rets = child.returns()
                    v = rets[0]
                else:
                    v = float(evaluator.evaluate(child)[0])
                value_map[str(a)] = v
        else:
            win_probs = [50,50]
            policy    = {}
            value_map = {}

        return jsonify(
            occupancy=occupancy,
            win_probs=win_probs,
            policy=policy,
            value_map=value_map,
            game_over=False
        )

    # REPLAY HISTORY
    state = game.new_initial_state()
    for m in session['moves']:
        state.apply_action(m)

    if user_move not in state.legal_actions():
        return jsonify(error='Illegal move'), 400

    # APPLY USER MOVE
    state.apply_action(user_move)
    session['moves'].append(user_move)
    session.modified = True

    # IF USER WINS
    if state.is_terminal():
        occupancy = get_occupancy(state)
        rets = state.returns()
        # rets[0] > rets[1] means first player won
        if rets[0] > rets[1]:
            win_probs = [100, 0]
        else:
            win_probs = [0, 100]
        return jsonify(
            occupancy=occupancy,
            win_probs=win_probs,
            policy={},
            value_map={},
            game_over=True
        )

    # AI MOVE
    if evaluator:
        pr = evaluator.prior(state)
        pr.sort(key=lambda x: -x[1])
        ai_move = pr[0][0]
    else:
        ai_move = random.choice(state.legal_actions())

    state.apply_action(ai_move)
    session['moves'].append(ai_move)
    session.modified = True

    occupancy = get_occupancy(state)

    # AFTER AI MOVE
    if state.is_terminal():
        # AI just won
        rets = state.returns()
        if rets[0] > rets[1]:
            win_probs = [100, 0]
        else:
            win_probs = [0, 100]
        return jsonify(
            occupancy=occupancy,
            win_probs=win_probs,
            policy={},
            value_map={},
            game_over=True
        )

    # non-terminal after AI
    if evaluator:
        v0  = float(evaluator.evaluate(state)[0])
        raw = max(0.0, min(1.0, (v0 + 1.0) / 2.0))
        win_probs = [int(raw*100), 100-int(raw*100)]
        policy    = { str(a): float(p) for a,p in evaluator.prior(state) }

        value_map = {}
        for a in state.legal_actions():
            child = state.clone()
            child.apply_action(a)
            if child.is_terminal():
                rets = child.returns()
                v = rets[0]
            else:
                v = float(evaluator.evaluate(child)[0])
            value_map[str(a)] = v
    else:
        win_probs = [50,50]
        policy    = {}
        value_map = {}

    return jsonify(
        occupancy=occupancy,
        win_probs=win_probs,
        policy=policy,
        value_map=value_map,
        game_over=False
    )


def get_occupancy(state):
    hist = state.history()
    occ = []
    for turn, action in enumerate(hist):
        r, c = divmod(action, BOARD_SIZE)
        player = 1 if (turn % 2) == 0 else 2
        occ.append([r, c, player])
    return occ


if __name__ == '__main__':
    app.run(debug=True)
