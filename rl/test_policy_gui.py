import os
import sys
import time
import numpy as np
from stable_baselines3 import PPO

# ============================================================
# 🔧 CONFIGURATION
# ============================================================
VERSION = "v2"  # "v1" or "v2"
SEEDS = [42, 123, 999]
EPISODES_PER_SEED = 1
MAX_STEPS = 2000
RENDER = True

# === 🧭 Absolute paths ===
ROOT_DIR = r"D:\FULBRIGHT\FALL 2025\git_robot\matcha\Matcha"
URDF_PATH = r"D:\FULBRIGHT\FALL 2025\git_robot\matcha\Matcha\data\balance_robot.urdf"

# ⚠️ 👉 Modify this line to point exactly to your model zip file
MODEL_PATH = r"D:\FULBRIGHT\FALL 2025\git_robot\matcha\Matcha\logs_ppo\v2_reward_tuned\run_20251023-184644\ppo_matcha_final.zip"

# Add rl folder to sys.path so Python can find v1/v2
sys.path.append(os.path.join(ROOT_DIR, "rl"))

# ============================================================
# 🧠 Dynamic import based on VERSION
# ============================================================
if VERSION == "v1":
    from v1.matcha_env import MatchaBalanceEnv
elif VERSION == "v2":
    from v2.matcha_env_v2 import MatchaBalanceEnvTuned as MatchaBalanceEnv
else:
    raise ValueError(f"❌ Unknown VERSION '{VERSION}' (choose 'v1' or 'v2')")

# ============================================================
# 🚀 Run GUI test
# ============================================================
def run_test(seed):
    print(f"\n=== Testing {VERSION} with seed {seed} ===")
    env = MatchaBalanceEnv(
        urdf_path=URDF_PATH,
        render=RENDER,
        time_step=1.0 / 480.0,
        max_episode_steps=MAX_STEPS,
        torque_limit=2.0,
        pitch_limit_deg=35.0,
        debug_joints=False,
        symmetric_action=True,
    )

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Model not found: {MODEL_PATH}")

    model = PPO.load(MODEL_PATH)
    durations, rewards = [], []

    for ep in range(EPISODES_PER_SEED):
        obs, _ = env.reset(seed=seed)
        start_time = time.time()
        total_reward = 0
        for step in range(MAX_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            time.sleep(1 / 240.0)
            if terminated or truncated:
                break
        duration = time.time() - start_time
        durations.append(duration)
        rewards.append(total_reward)
        print(f"Episode {ep+1}: lasted {duration:.2f}s | reward={total_reward:.1f}")
    env.close()
    return durations, rewards


# ============================================================
# 📊 MAIN
# ============================================================
if __name__ == "__main__":
    all_durations, all_rewards = [], []
    for seed in SEEDS:
        durations, rewards = run_test(seed)
        all_durations.extend(durations)
        all_rewards.extend(rewards)

    print("\n=== 📈 SUMMARY ===")
    print(f"Version         : {VERSION}")
    print(f"Model tested    : {os.path.basename(MODEL_PATH)}")
    print(f"Avg survival    : {np.mean(all_durations):.2f}s")
    print(f"Max survival    : {np.max(all_durations):.2f}s")
    print(f"Avg reward      : {np.mean(all_rewards):.1f}")
    print(f"Reward std dev  : {np.std(all_rewards):.1f}")
    print(f"Stability check : {'✅ Stable' if np.std(all_rewards) < 50 else '⚠️ Unstable'}")
