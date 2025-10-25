
# ==========================================================
# eval_ppo.py - Evaluation & Demo for MatchaBalanceEnvTuned (v2)
# ==========================================================
import os
import time
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from v2.matcha_env_v2 import MatchaBalanceEnvTuned as MatchaBalanceEnv

# ---------- Paths ----------
URDF_PATH = r"D:\FALL\PJ\Matcha\hardware\balance_robot.urdf"
MODEL_PATH = r"D:\FALL\PJ\Matcha\logs_ppo\v2_reward_tuned\run_20251025-180428\ppo_matcha_final.zip"


# ---------- Environment Factory ----------
def make_env(render=False):
    """
    Creates a single instance of the environment.
    Note: For GUI demo, use the raw env (not DummyVecEnv).
    """
    def _thunk():
        return MatchaBalanceEnv(
            urdf_path=URDF_PATH,
            render=render,
            max_episode_steps=1500,
            symmetric_action=True,
        )
    return _thunk


# ---------- Main ----------
def main(render_demo: bool = False):
    # Load trained PPO model
    assert os.path.isfile(MODEL_PATH), f"Model not found: {MODEL_PATH}"
    print(f"[INFO] Loading PPO model from: {MODEL_PATH}")
    model = PPO.load(MODEL_PATH)

    # ----- Headless Evaluation -----
    print("\n[STEP 1] Running evaluation (no render)...")
    eval_env = DummyVecEnv([make_env(render=False)])
    eval_env = VecMonitor(eval_env)

    mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=20, deterministic=True)
    print(f"[EVAL] Mean reward over 20 episodes: {mean_r:.2f} ± {std_r:.2f}")
    eval_env.close()

    # ----- Optional GUI Demo -----
    if render_demo:
        print("\n[STEP 2] Starting GUI demo...")
        demo_env = make_env(render=True)()  # Direct environment, not wrapped
        obs, info = demo_env.reset()
        ep_r, ep_len = 0.0, 0

        real_start = time.time()
        for t in range(5000):  # about 20s at 240Hz
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = demo_env.step(action)
            ep_r += r
            ep_len += 1
            time.sleep(1/240)  # realtime playback
            if term or trunc:
                break

        real_elapsed = time.time() - real_start
        sim_elapsed = ep_len * demo_env.time_step

        print(f"\n[RESULT] Simulated time = {sim_elapsed:.2f}s  |  Real elapsed = {real_elapsed:.2f}s")
        print(f"[RESULT] Steps before fall = {ep_len}  |  Return = {ep_r:.2f}")

        demo_env.close()
    else:
        print("\n[INFO] Demo skipped (use --render to show PyBullet GUI).")


# ---------- CLI Entry ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PPO on MatchaBalanceEnvTuned")
    parser.add_argument("--render", action="store_true", help="Show PyBullet GUI demo after evaluation")
    args = parser.parse_args()

    main(render_demo=args.render)
