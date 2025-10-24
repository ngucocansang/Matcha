# ==========================================================
# Matcha PPO - v3_upright_balance
# ----------------------------------------------------------
# Strict upright balancing with tuned reward shaping
# ==========================================================

import os, time, json
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from matcha_env_v2 import MatchaBalanceEnvUpright

# ========== GLOBAL CONFIG ==========
VERSION_TAG = "v2_upright_balance"
SEED = 42
URDF_PATH = r"D:\FULBRIGHT\FALL 2025\git_robot\matcha\Matcha\data\balance_robot.urdf"

LOG_ROOT = f"./logs_ppo/{VERSION_TAG}"
os.makedirs(LOG_ROOT, exist_ok=True)

RUN_ID = time.strftime("%Y%m%d-%H%M%S")
LOG_DIR = os.path.join(LOG_ROOT, f"run_{RUN_ID}")
os.makedirs(LOG_DIR, exist_ok=True)

# ========== TRAINING CONFIG ==========
TOTAL_STEPS = 800_000  # nhiều hơn vì reward nghiêm ngặt hơn
N_ENVS = 8
MAX_EPISODE_STEPS = 1500
LEARNING_RATE = 3e-4

def make_env(rank: int, render: bool = False):
    def _thunk():
        env = MatchaBalanceEnvUpright(
            urdf_path=URDF_PATH,
            render=render,
            time_step=1.0 / 240.0,
            max_episode_steps=MAX_EPISODE_STEPS,
            torque_limit=2.0,
            pitch_limit_deg=35.0,
            debug_joints=False,
            symmetric_action=True,
        )
        return env
    return _thunk


if __name__ == "__main__":
    print(f"📂 Logging to {LOG_DIR}")

    # ---------- Parallel training environments ----------
    venv = SubprocVecEnv([make_env(i, render=False) for i in range(N_ENVS)])
    venv = VecMonitor(venv, filename=os.path.join(LOG_DIR, "monitor"))

    eval_env = DummyVecEnv([make_env(999, render=False)])
    eval_env = VecMonitor(eval_env, filename=os.path.join(LOG_DIR, "eval_monitor"))

    # ---------- PPO model setup ----------
    model = PPO(
        policy="MlpPolicy",
        env=venv,
        seed=SEED,
        verbose=1,
        learning_rate=LEARNING_RATE,
        n_steps=1024,
        batch_size=512,
        n_epochs=10,
        gamma=0.995,           # giữ nguyên để ưu tiên sống lâu
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,         # tăng exploration vì reward nghiêm ngặt hơn
        vf_coef=0.6,
        max_grad_norm=0.5,
        tensorboard_log=os.path.join(LOG_DIR, "tb"),
        policy_kwargs=dict(net_arch=[256, 256]),
    )

    # ---------- Callbacks ----------
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=LOG_DIR,
        log_path=LOG_DIR,
        eval_freq=10_000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )
    ckpt_cb = CheckpointCallback(save_freq=50_000, save_path=LOG_DIR, name_prefix="ckpt")

    # ---------- Metadata ----------
    metadata = {
        "version": VERSION_TAG,
        "run_id": RUN_ID,
        "urdf_path": URDF_PATH,
        "seed": SEED,
        "n_envs": N_ENVS,
        "total_steps": TOTAL_STEPS,
        "learning_rate": LEARNING_RATE,
        "policy_arch": [256, 256],
        "max_episode_steps": MAX_EPISODE_STEPS,
        "notes": "Strict upright balance; penalize lean offset, x drift, velocity; reward only for vertical balance.",
    }
    with open(os.path.join(LOG_DIR, "metadata.txt"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    # ---------- Train ----------
    print(f"🚀 Starting PPO training for version: {VERSION_TAG}")
    model.learn(total_timesteps=TOTAL_STEPS, callback=[eval_cb, ckpt_cb], progress_bar=True)

    model.save(os.path.join(LOG_DIR, "ppo_matcha_upright_final.zip"))
    print(f"✅ Training complete! Model saved to {LOG_DIR}")

    venv.close()
    eval_env.close()
