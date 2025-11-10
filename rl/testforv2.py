# rl/eval_ppo.py
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
import os, time

from v1.matcha_env import MatchaBalanceEnv as MatchaBalanceEnv1
from v2.matcha_env_v2 import MatchaBalanceEnvTuned as MatchaBalanceEnv2

URDF_PATH = r"D:\FALL\PJ\Matcha\hardware\balance_robot.urdf"
MODEL_PATH = r"logs_ppo\v2_reward_tuned\run_20251026-004447\ppo_matcha_final.zip"

assert os.path.exists(URDF_PATH), f"URDF not found: {URDF_PATH}"
assert os.path.exists(MODEL_PATH), f"Model not found: {MODEL_PATH}"

def make_env(render=False):
    def _thunk():
        return MatchaBalanceEnv2(
            urdf_path=URDF_PATH, render=render,
            max_episode_steps=1500, symmetric_action=True
        )
    return _thunk

if __name__ == "__main__":
    # --- Evaluation ---
    eval_env = DummyVecEnv([make_env(render=False)])
    eval_env = VecMonitor(eval_env)

    model = PPO.load(MODEL_PATH)
    mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=20, deterministic=True)
    print(f"[EVAL] Mean reward over 20 eps: {mean_r:.2f} ± {std_r:.2f}")
    eval_env.close()

    # --- Demo with GUI ---
    demo_env = make_env(render=True)()
    try:
        obs = demo_env.reset()[0]  # works for Gymnasium
    except Exception:
        obs = demo_env.reset()

    ep_r, ep_len = 0.0, 0
    real_start = time.time()

    for t in range(5000):
        action, _ = model.predict(obs, deterministic=True)
        try:
            obs, r, term, trunc, _ = demo_env.step(action)
            done = term or trunc
        except Exception:
            obs, r, done, _ = demo_env.step(action)
        ep_r += r
        ep_len += 1
        if done:
            break
        time.sleep(1/240)

    real_elapsed = time.time() - real_start
    sim_elapsed = ep_len * demo_env.time_step

    print(f"\n[RESULT] Simulated time = {sim_elapsed:.2f}s  |  Real elapsed = {real_elapsed:.2f}s")
    print(f"[RESULT] Steps before fall = {ep_len}  |  Return = {ep_r:.2f}")

    demo_env.close()
