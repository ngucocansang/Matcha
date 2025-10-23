# ==========================================================
# Matcha Robot Evaluation Comparison (Auto Version)
# ----------------------------------------------------------
# Automatically detect the 2 most recent PPO training runs
# and compare their performance (stability, survival, reward)
#
# Authors: Team Matcha — Fulbright University Vietnam
# PM: Đinh Hồng Ngọc | HW: Phương Quỳnh | SW: Alex
# Instructor: Prof. Dương Phùng
# ==========================================================

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from matcha_env import MatchaBalanceEnv


# ===================== CONFIG =====================
LOG_ROOT = "../logs_ppo"       # Thư mục gốc chứa các run_xxx
N_EPISODES = 5                 # Số episode đánh giá mỗi model
MAX_STEPS = 3000               # Giới hạn bước mô phỏng
RENDER = True                  # Bật PyBullet GUI
TIME_SCALE = 1.0               # 1.0 = realtime, <1 = nhanh hơn


# ===================== HELPER: FIND 2 LATEST RUNS =====================
def get_latest_two_runs(log_root):
    """Tự động tìm 2 thư mục run_xxx mới nhất."""
    runs = [os.path.join(log_root, d) for d in os.listdir(log_root)
            if os.path.isdir(os.path.join(log_root, d)) and d.startswith("run_")]
    runs.sort(key=os.path.getmtime, reverse=True)
    if len(runs) < 2:
        raise RuntimeError("❌ Need at least 2 runs in logs_ppo/ to compare.")
    return runs[1], runs[0]  # (older, newer)


# ===================== EVALUATION FUNCTION =====================
def evaluate_model(run_path, name):
    print(f"\n🎯 Evaluating {name} at: {run_path}")

    # Locate model and normalization files
    model_path = os.path.join(run_path, "best_model.zip")
    if not os.path.isfile(model_path):
        # fallback: use ckpt if no best_model
        ckpts = [f for f in os.listdir(run_path) if f.endswith(".zip")]
        if not ckpts:
            raise FileNotFoundError(f"❌ No model found in {run_path}")
        model_path = os.path.join(run_path, ckpts[-1])
        print(f"⚠️ Using fallback checkpoint: {model_path}")

    # Make environment
    def make_env():
        env = MatchaBalanceEnv(
            urdf_path="../data/balance_robot.urdf",
            render=RENDER,
            max_episode_steps=MAX_STEPS,
        )
        return env

    env = DummyVecEnv([make_env])
    env = VecNormalize(env, training=False, norm_obs=True, norm_reward=False)

    # Try loading VecNormalize stats
    try:
        env.load_running_average(run_path)
        print("✅ Loaded VecNormalize stats.")
    except Exception as e:
        print(f"⚠️ No VecNormalize found: {e}")

    # Load PPO model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PPO.load(model_path, env=env, device=device)

    stats = {"survival": [], "pitch_mean": [], "x_drift": [], "reward": []}

    for ep in range(N_EPISODES):
        obs, _ = env.reset()
        done = False
        total_reward, total_pitch, total_x_drift, steps = 0.0, 0.0, 0.0, 0
        start_t = time.time()

        while not done and steps < MAX_STEPS:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)

            pitch = obs[0][0]
            pitch_deg = np.degrees(pitch)
            total_pitch += abs(pitch_deg)

            x_drift = abs(obs[0][2])
            total_x_drift += x_drift

            total_reward += reward
            steps += 1
            if RENDER:
                time.sleep(TIME_SCALE * env.get_attr("time_step")[0])

        elapsed = time.time() - start_t
        stats["survival"].append(elapsed)
        stats["pitch_mean"].append(total_pitch / steps)
        stats["x_drift"].append(total_x_drift / steps)
        stats["reward"].append(total_reward / steps)

        print(f"Ep {ep+1}/{N_EPISODES} — time={elapsed:.2f}s, "
              f"mean_pitch={stats['pitch_mean'][-1]:.3f}°, "
              f"mean_x_drift={stats['x_drift'][-1]:.3f}, "
              f"mean_reward={stats['reward'][-1]:.3f}")

    env.close()
    results = {k: np.mean(v) for k, v in stats.items()}
    print(f"\n📊 {name} Summary:")
    for k, v in results.items():
        print(f"  {k:12s}: {v:.4f}")

    return results


# ===================== MAIN =====================
if __name__ == "__main__":
    run_old, run_new = get_latest_two_runs(LOG_ROOT)
    print(f"📂 Comparing:\n  1️⃣ {run_old}\n  2️⃣ {run_new}")

    result1 = evaluate_model(run_old, "Older PPO Run")
    result2 = evaluate_model(run_new, "Newer PPO Run")

    # ===================== PLOT =====================
    labels = ["Survival Time (s)", "Pitch Deviation (°)", "X Drift", "Reward/Step"]
    keys = ["survival", "pitch_mean", "x_drift", "reward"]

    x = np.arange(len(labels))
    width = 0.35

    v1_vals = [result1[k] for k in keys]
    v2_vals = [result2[k] for k in keys]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, v1_vals, width, label="Older Run", color="#6baed6")
    ax.bar(x + width/2, v2_vals, width, label="Newer Run", color="#fd8d3c")

    ax.set_ylabel("Value")
    ax.set_title("🏗️ Matcha PPO Model Comparison (Auto Latest 2 Runs)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()
