# Hex RL algorithm comparison
In this project the three Reinforcement Algorithms AlphaZero, NFSP and PPO are tested on the board game Hex using the three board sizes 5x5, 8x8 and 11x11.
AlphaZero is the algorithm that produced the overall best results

![image](https://github.com/user-attachments/assets/93d27940-2970-4c60-8125-51e33cf1204b)

# How to Run

Download OpenSpiel following this tutorial: https://github.com/google-deepmind/open_spiel/blob/master/docs/install.md

```bash
# Clone the repository
git clone https://github.com/<your-username>/RL-Hex-Agents.git
cd RL-Hex-Agents

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

# Visualize the project
```bash
# Visualize 
python visualization/visualize_policy_head.py ./evaluation/best_weight/alpha_zero/alpha_zero_hex_8_24h/ checkpoint-200 dummy
python visualization/visualize_value_head.py ./evaluation/best_weight/alpha_zero/alpha_zero_hex_8_24h/ checkpoint-200 dummy
# Loading a flask application on port 5000:
python visualization/hex_web_app.py
```

Screenshot from hex_web_app.py:

![image](https://github.com/user-attachments/assets/de5eb7e8-639f-4d7a-83fe-abf3e8cedb6e)

GIF from visualize_policy_head.py:

![Skjermopptak2025-04-30154305-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/9ec675a7-d06e-4a19-b1e5-c54dd7572bcc)

# Train and evaluate
The configuration is set to 4 hours, override if you want.
```bash
python models/alpha_zero.py
python models/nfsp.py
python models/ppo.py
```

This will play all stored weights againt MCTSBot and extract the best performing weight.
```bash
python evaluation/evaluate_all.py
```
This file will use the best performing weight in a Round Robin tournament where all algorithms compete against each other.
```bash
python evaluation/round_robin.py
```
# Result from Round Robin
![image](https://github.com/user-attachments/assets/93d27940-2970-4c60-8125-51e33cf1204b)


