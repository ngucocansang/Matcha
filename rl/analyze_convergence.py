# ==========================================
# Matcha PPO Training Convergence Analyzer
# Author: Team Matcha (Fulbright University)
# ==========================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === CONFIG: chỉnh đúng đường dẫn ===
TRAIN_LOG = r"D:\FULBRIGHT\FALL 2025\git_robot\matcha\Matcha\logs_ppo\run_20251023-171447\monitor.monitor.csv"
EVAL_LOG  = r"D:\FULBRIGHT\FALL 2025\git_robot\matcha\Matcha\logs_ppo\run_20251023-171447\eval_monitor.monitor.csv"

# === CHECK FILES ===
for f in [TRAIN_LOG, EVAL_LOG]:
    if not os.path.isfile(f):
        raise FileNotFoundError(f"❌ File not found: {f}")
print("✅ Found both monitor files.")

# === LOAD CSVs ===
train_df = pd.read_csv(TRAIN_LOG, skiprows=1)
eval_df  = pd.read_csv(EVAL_LOG, skiprows=1)

# Cột quan trọng:
# r = reward mỗi episode, l = episode length, t = thời gian (s)
train_rewards = train_df["r"]
eval_rewards  = eval_df["r"]

# === BASIC STATS ===
def analyze(name, rewards):
    mean_r = rewards.tail(50).mean()
    std_r  = rewards.tail(50).std()
    rel_std = std_r / abs(mean_r + 1e-8)
    print(f"\n📊 {name.upper()} RESULTS (last 50 episodes):")
    print(f"   Mean reward  : {mean_r:.3f}")
    print(f"   Std reward   : {std_r:.3f}")
    print(f"   Rel. std (%) : {rel_std*100:.2f}%")
    stable = rel_std < 0.1
    if stable:
        print("   ✅ Stable learning achieved.")
    else:
        print("   ⚠️ Still unstable or improving.")
    return mean_r, std_r, stable

mean_train, std_train, stable_train = analyze("Training", train_rewards)
mean_eval, std_eval, stable_eval = analyze("Evaluation", eval_rewards)

# === OVERALL SUMMARY ===
if stable_train and stable_eval:
    print("\n✅ PPO appears to be CONVERGED and STABLE for Matcha robot.")
else:
    print("\n🚧 PPO not fully converged yet — consider longer training or tune reward scaling.")

# === PLOTTING ===
plt.figure(figsize=(12, 6))

# 1️⃣ Training Reward Curve
plt.subplot(2, 1, 1)
plt.plot(train_rewards, color='tab:green', label='Training Reward')
plt.title("Matcha PPO – Training Rewards")
plt.ylabel("Reward per episode")
plt.grid(True, alpha=0.3)
plt.legend()

# 2️⃣ Evaluation Reward Curve
plt.subplot(2, 1, 2)
plt.plot(eval_rewards, color='tab:blue', label='Evaluation Reward')
plt.title("Matcha PPO – Evaluation Rewards")
plt.xlabel("Episode")
plt.ylabel("Reward per eval episode")
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()

# === EXTRA METRICS ===
improvement = eval_rewards.mean() - train_rewards.mean()
print(f"\n📈 Average reward improvement (Eval - Train): {improvement:.3f}")
