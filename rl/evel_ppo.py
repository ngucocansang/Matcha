# rl/eval_ppo.py
from stable_baselines3 import PPO
import os 
import time 
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from v1.matcha_env import MatchaBalanceEnv as MatchaBalanceEnv1


URDF_PATH = r"D:\FALL\PJ\Matcha\hardware\balance_robot.urdf"
MODEL_PATH = r"logs_ppo\v2_reward_tuned\run_20251027-023310\ppo_matcha_final.zip"


def make_env(render=False):
    def _thunk():
        return MatchaBalanceEnv1(
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

# --- DEMO with GUI + balance time measurement ---
demo_env = make_env(render=True)()
obs, info = demo_env.reset()
ep_r, ep_len = 0.0, 0

real_start = time.time()

for t in range(5000):  # ~20 seconds at 240Hz
    action, _ = model.predict(obs, deterministic=True)
    obs, r, term, trunc, _ = demo_env.step(action)
    ep_r += r
    ep_len += 1
    time.sleep(1/240)  # realtime speed (optional)
    if term or trunc:
        break

real_elapsed = time.time() - real_start
sim_elapsed = ep_len * demo_env.time_step

print(f"\n[RESULT] Simulated time = {sim_elapsed:.2f}s  |  Real elapsed = {real_elapsed:.2f}s")
print(f"[RESULT] Steps before fall = {ep_len}  |  Return = {ep_r:.2f}")

demo_env.close()
