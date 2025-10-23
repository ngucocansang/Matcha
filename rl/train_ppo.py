# ==========================================================
# PPO Training Script for Matcha Robot - Stable v2 (Windows-safe)
# ----------------------------------------------------------
# Authors: Team Matcha (PM: Đinh Hồng Ngọc, Eng: Phương Quỳnh, SW: Alex)
# Instructor: Prof. Dương Phùng – Fulbright University Vietnam
# ==========================================================

import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from matcha_env import MatchaBalanceEnv


# ========================= CONFIG =========================
SEED = 42
N_ENVS = 8
TOTAL_STEPS = 500_000
LOG_DIR = "../logs_ppo/run_v2"
MODEL_PATH = os.path.join(LOG_DIR, "ppo_matcha_v2.zip")

os.makedirs(LOG_DIR, exist_ok=True)
print(f"📂 Logging to {LOG_DIR}")


# ===================== ENV FACTORY ========================
def make_env(rank: int, render: bool = False):
    """Factory to create independent Matcha environments."""
    def _init():
        env = MatchaBalanceEnv(
            urdf_path= r"D:\FULBRIGHT\FALL 2025\git_robot\matcha\Matcha\data\balance_robot.urdf",
            render=render,
            debug_joints=False,
        )
        env.reset(seed=SEED + rank)
        return env
    return _init


# ========================= MAIN ===========================
if __name__ == "__main__":
    # Vectorized environments (parallel training)
    venv = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    venv = VecMonitor(venv, filename=os.path.join(LOG_DIR, "monitor"))
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True,
                        gamma=0.99, clip_obs=10.0)

    # Evaluation environment
    eval_env = SubprocVecEnv([make_env(999)])
    eval_env = VecMonitor(eval_env, filename=os.path.join(LOG_DIR, "eval_monitor"))
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)

    eval_norm_path = os.path.join(LOG_DIR, "vecnormalize.pkl")
    if os.path.exists(eval_norm_path):
        try:
            eval_env.load_running_average(LOG_DIR)
            print("✅ Loaded previous normalization stats.")
        except Exception:
            print("⚠️ Could not load previous normalization stats, starting fresh.")

    # ================== CALLBACKS ==========================
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=LOG_DIR,
        log_path=LOG_DIR,
        eval_freq=10_000,
        deterministic=True,
        render=False,
        verbose=1,
    )

    ckpt_cb = CheckpointCallback(
        save_freq=100_000,
        save_path=LOG_DIR,
        name_prefix="ckpt"
    )

    # ================== PPO MODEL ==========================
    model = PPO(
        "MlpPolicy",
        venv,
        learning_rate=1e-4,          # smaller for smoother learning
        n_steps=2048,                # rollout length
        batch_size=1024,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,              # slight entropy to encourage exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=LOG_DIR,
        policy_kwargs=dict(net_arch=[256, 256]),
        seed=SEED,
        verbose=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # =================== TRAINING ==========================
    print("🚀 Training Matcha PPO v2 (stable, Windows-safe)...")
    model.learn(
        total_timesteps=TOTAL_STEPS,
        callback=[eval_cb, ckpt_cb],
        progress_bar=True,
    )

    # =================== SAVE ==============================
    model.save(MODEL_PATH)
    venv.save(eval_norm_path)

    print("\n✅ Training complete.")
    print(f"📁 Model saved to: {MODEL_PATH}")
    print(f"📊 Logs & normalization saved in: {LOG_DIR}")
