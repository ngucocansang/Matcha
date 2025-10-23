# ==========================================================
# PPO Training Script - Vision-only RL (v3)
# ==========================================================
import os, time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from matcha_env_v3 import MatchaBalanceCamEnv

VERSION_TAG = "v3_cam_vision_only"
SEED = 42
set_random_seed(SEED)

LOG_ROOT = f"./logs_ppo/{VERSION_TAG}"
URDF_PATH = r"D:\FULBRIGHT\FALL 2025\git_robot\matcha\Matcha\hardware\balance_robot.urdf"

RUN_ID = time.strftime("%Y%m%d-%H%M%S")
VERSION = "v3_vision_only"
LOG_DIR = os.path.join(LOG_ROOT, f"{VERSION}_run_{RUN_ID}")
os.makedirs(LOG_DIR, exist_ok=True)

# ==========================================================
def make_env(rank: int, render=False):
    def _thunk():
        env = MatchaBalanceCamEnv(
            urdf_path=URDF_PATH,
            render=render,
            image_size=(84, 84),
            time_step=1.0 / 240.0,
            max_episode_steps=1500,
            torque_limit=2.0,
            pitch_limit_deg=35.0,
            symmetric_action=True,
        )
        return env
    return _thunk

# ==========================================================
if __name__ == "__main__":
    N_ENVS = 2  # reduce if CPU weak
    venv = SubprocVecEnv([make_env(i, render=False) for i in range(N_ENVS)])
    venv = VecMonitor(venv, filename=os.path.join(LOG_DIR, "monitor"))

    model = PPO(
        policy="CnnPolicy",       # <—— switch from MLP to CNN
        env=venv,
        seed=SEED,
        verbose=1,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=512,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=LOG_DIR,
        policy_kwargs=dict(features_extractor_kwargs=dict(features_dim=256)),
    )

    # Eval env
    eval_env = make_env(999, render=False)()
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=LOG_DIR,
        log_path=LOG_DIR,
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
    )
    ckpt_cb = CheckpointCallback(save_freq=50_000, save_path=LOG_DIR, name_prefix="ckpt")

    TOTAL_STEPS = 300_000
    print(f"🚀 Training Vision-only PPO for {TOTAL_STEPS} steps...")
    model.learn(total_timesteps=TOTAL_STEPS, callback=[eval_cb, ckpt_cb], progress_bar=True)

    model.save(os.path.join(LOG_DIR, "ppo_matcha_v3_final.zip"))
    print(f"✅ Model saved to {LOG_DIR}")

    venv.close()
    eval_env.close()
