# ==========================================================
# Matcha PPO - Cross-Version Evaluation Analyzer
# ----------------------------------------------------------
# Compare PPO training performance across versions
# ==========================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

LOG_ROOT = "./logs_ppo"
MIN_EPISODES = 20  # bỏ qua những run ngắn

def find_versions(root):
    """Find all version subfolders under logs_ppo/"""
    versions = [v for v in os.listdir(root) if os.path.isdir(os.path.join(root, v))]
    return sorted(versions)

def collect_csvs(version_folder):
    """Collect all eval_monitor files in a version folder"""
    pattern = os.path.join(version_folder, "run_*/eval_monitor.monitor.csv")
    return sorted(glob(pattern))

def read_monitor_csv(file_path):
    """Read a stable-baselines3 monitor CSV"""
    try:
        df = pd.read_csv(file_path, skiprows=1)  # skip comment line
        df.columns = ["r", "l", "t"]  # reward, length, time
        return df
    except Exception as e:
        print(f"⚠️ Could not read {file_path}: {e}")
        return None

def summarize_version(version_name, csv_paths):
    """Compute metrics for one version"""
    all_rewards = []
    for path in csv_paths:
        df = read_monitor_csv(path)
        if df is not None and len(df) > MIN_EPISODES:
            all_rewards.append(df["r"].values)
    if not all_rewards:
        return None
    rewards = np.concatenate(all_rewards)
    mean_r, std_r = np.mean(rewards), np.std(rewards)
    last_mean = np.mean(rewards[-50:]) if len(rewards) >= 50 else mean_r
    best_r = np.max(rewards)
    rel_std = (std_r / abs(mean_r + 1e-8)) * 100
    stability = "✅ Stable" if rel_std < 10 else ("⚠️ Medium" if rel_std < 30 else "❌ Unstable")

    return {
        "version": version_name,
        "mean_reward": round(mean_r, 2),
        "std_reward": round(std_r, 2),
        "rel_std_%": round(rel_std, 2),
        "best_reward": round(best_r, 2),
        "last50_mean": round(last_mean, 2),
        "stability": stability,
    }

def plot_comparison(results, save_path=None):
    plt.figure(figsize=(10, 6))
    for res in results:
        version = res["version"]
        csv_paths = collect_csvs(os.path.join(LOG_ROOT, version))
        rewards_all = []
        for pth in csv_paths:
            df = read_monitor_csv(pth)
            if df is not None:
                rewards_all.append(df["r"].rolling(10).mean())
        if rewards_all:
            concat = pd.concat(rewards_all, axis=0).reset_index(drop=True)
            plt.plot(concat, label=version)
    plt.title("📈 Matcha PPO - Version Comparison")
    plt.xlabel("Episode")
    plt.ylabel("Rolling Reward (mean over 10)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"📊 Saved comparison plot to {save_path}")
    plt.show()

if __name__ == "__main__":
    print("🔍 Scanning logs_ppo for versions...\n")
    versions = find_versions(LOG_ROOT)
    all_results = []

    for ver in versions:
        csvs = collect_csvs(os.path.join(LOG_ROOT, ver))
        summary = summarize_version(ver, csvs)
        if summary:
            all_results.append(summary)

    if not all_results:
        print("❌ No valid monitor files found.")
        exit(0)

    df_summary = pd.DataFrame(all_results)
    df_summary = df_summary.sort_values("mean_reward", ascending=False)
    print("\n📊 PPO VERSION COMPARISON:")
    print(df_summary.to_string(index=False))

    save_path = os.path.join(LOG_ROOT, "version_comparison.png")
    plot_comparison(all_results, save_path=save_path)

    df_summary.to_csv(os.path.join(LOG_ROOT, "version_summary.csv"), index=False)
    print(f"✅ Summary saved to {LOG_ROOT}/version_summary.csv")
