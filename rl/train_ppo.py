# rl/train_ppo.py
import os, time
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from matcha_env import MatchaBalanceEnv

SEED = 42
set_random_seed(SEED)

URDF_PATH = "/mnt/data/balance_robot.urdf"  # <-- path URDF của bạn
LOG_ROOT = "./logs_ppo"
RUN_ID = time.strftime("%Y%m%d-%H%M%S")
LOG_DIR = os.path.join(LOG_ROOT, f"run_{RUN_ID}")
os.makedirs(LOG_DIR, exist_ok=True)

def make_env(rank:int, render:bool=False):
    def _thunk():
        env = MatchaBalanceEnv(
            urdf_path=URDF_PATH,
            render=render,
            time_step=1.0/240.0,
            max_episode_steps=1500,
            torque_limit=2.0,
            pitch_limit_deg=35.0,
            debug_joints=False,
            symmetric_action=True,  # bắt đầu bài toán 1D dễ hơn
        )
        return env
    return _thunk

if __name__ == "__main__":
    N_ENVS = 8  # nếu máy yếu có thể hạ xuống 4
    venv = SubprocVecEnv([make_env(i, render=False) for i in range(N_ENVS)])
    venv = VecMonitor(venv, filename=os.path.join(LOG_DIR, "monitor"))

    model = PPO(
        policy="MlpPolicy",
        env=venv,
        seed=SEED,
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=512,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=LOG_DIR,
        policy_kwargs=dict(net_arch=[256, 256]),
    )

    eval_env = DummyVecEnv([make_env(999, render=False)])
    eval_env = VecMonitor(eval_env, filename=os.path.join(LOG_DIR, "eval_monitor"))

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

    TOTAL_STEPS = 500_000
    model.learn(total_timesteps=TOTAL_STEPS, callback=[eval_cb, ckpt_cb], progress_bar=True)

    model.save(os.path.join(LOG_DIR, "ppo_matcha_final.zip"))
    venv.close()
    eval_env.close()
