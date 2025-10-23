# rl/eval_ppo.py
import os
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from matcha_env import MatchaBalanceEnv

URDF_PATH = "/mnt/data/balance_robot.urdf"
MODEL_PATH = "./logs_ppo/<THAY-BANG-THU-MUC-RUN>/ppo_matcha_final.zip"

def make_env(render=False):
    def _thunk():
        return MatchaBalanceEnv(
            urdf_path=URDF_PATH, render=render,
            max_episode_steps=1500, symmetric_action=True
        )
    return _thunk

if __name__ == "__main__":
    eval_env = DummyVecEnv([make_env(render=False)])
    eval_env = VecMonitor(eval_env)

    model = PPO.load(MODEL_PATH)
    mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=20, deterministic=True)
    print(f"[EVAL] Mean reward over 20 eps: {mean_r:.2f} ± {std_r:.2f}")
    eval_env.close()

    # Demo 1 episode có GUI
    demo_env = make_env(render=True)()
    obs, info = demo_env.reset()
    ep_r, ep_len = 0.0, 0
    for t in range(2000):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, _ = demo_env.step(action)
        ep_r += r; ep_len += 1
        if term or trunc:
            print(f"[DEMO] Return={ep_r:.2f}, len={ep_len}")
            break
    demo_env.close()
