import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(BASE_DIR, 'weights', 'alpha_zero')
OUTPUT_DIR = os.path.join(BASE_DIR, 'evaluation', 'output')

# Create output directory if not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Pattern to extract losses
loss_pattern = re.compile(
    r"Losses\(total: ([\d\.]+), policy: ([\d\.]+), value: ([\d\.]+), l2: ([\d\.]+)\)"
)

# Find all alpha_zero_hex folders
folders = [f for f in os.listdir(WEIGHTS_DIR) if f.startswith('alpha_zero_hex')]

for folder in folders:
    log_path = os.path.join(WEIGHTS_DIR, folder, 'log-learner.txt')
    if not os.path.exists(log_path):
        print(f"Log file not found for {folder}")
        continue

    # Read file
    with open(log_path, 'r') as f:
        lines = f.readlines()

    # Extract losses
    losses = []
    for line in lines:
        match = loss_pattern.search(line)
        if match:
            total_loss, policy_loss, value_loss, l2_loss = map(float, match.groups())
            losses.append({
                'total_loss': total_loss,
                'policy_loss': policy_loss,
                'value_loss': value_loss,
                'l2_loss': l2_loss,
            })

    # Save to CSV
    df = pd.DataFrame(losses)
    csv_path = os.path.join(OUTPUT_DIR, f"{folder}_losses.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    # Plot (policy, value, l2 only)
    plt.figure(figsize=(10, 6))
    plt.plot(df['policy_loss'], label='Policy Loss')
    plt.plot(df['value_loss'], label='Value Loss')
    plt.plot(df['l2_loss'], label='L2 Loss')
    #plt.title(f"Losses over Steps for {folder}")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(OUTPUT_DIR, f"{folder}_losses_plot.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved Plot: {plot_path}")

print("All done!")
