import time
import numpy as np
import pybullet as p
from stable_baselines3 import PPO
from v3.matcha_env_v3 import MatchaBalanceCamEnv

URDF_PATH = r"D:\FALL\PJ\Matcha\hardware\balance_robot.urdf"
MODEL_PATH = r"D:\FALL\PJ\Matcha\logs_ppo\v2_reward_tuned\run_20251025-170214\ppo_matcha_final.zip"

# GUI env
env = MatchaBalanceCamEnv(urdf_path=URDF_PATH, render=True, image_size=(84,84))
model = PPO.load(MODEL_PATH)

N_EPISODES = 5
durations = []
rewards = []
step = 0

for ep in range(N_EPISODES):
    obs, info = env.reset()
    done = False
    total_reward = 0
    t0 = time.time()
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        # p.stepSimulation()
        time.sleep(1.0 / 120.0)  # real-time playback
        sim_seconds = env.step_count * env.time_step
        print(f"Ep {ep+1}: survived {sim_seconds:.2f}s (sim), reward={total_reward:.1f}")
                # 💥 Push test: apply external force at step 300
        if step == 300:
            print("Applying external force!")
            p.applyExternalForce(
                objectUniqueId=env.robot_id,   # robot ID
                linkIndex=-1,                  # apply to base link
                forceObj=[20, 0, 0],           # push forward (X direction)
                posObj=[0, 0, 0.1],            # apply slightly above base
                flags=p.WORLD_FRAME
            )


    duration = time.time() - t0
    durations.append(duration)
    rewards.append(total_reward)
    print(f"Ep {ep+1}: survived {duration:.2f}s, reward={total_reward:.1f}")

env.close()

print("\n=== 📊 Summary ===")
print(f"Avg survival: {np.mean(durations):.2f}s | Max: {np.max(durations):.2f}s")
print(f"Avg reward  : {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
