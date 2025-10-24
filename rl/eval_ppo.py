# ==========================================================
# Matcha PPO Evaluation (GUI) - Upright Balance (v3)
# ----------------------------------------------------------
# Render and visualize upright balance behavior in PyBullet GUI
# ==========================================================

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from v2.matcha_env_v2 import MatchaBalanceEnvUpright
import pybullet as p

# ---------- CONFIG ----------
MODEL_PATH = r"D:\FULBRIGHT\FALL 2025\git_robot\matcha\Matcha\logs_ppo\v2_reward_tuned\run_20251023-184644\ppo_matcha_final.zip"
URDF_PATH = r"D:\FULBRIGHT\FALL 2025\git_robot\matcha\Matcha\data\balance_robot.urdf"
STEPS = 3000            # tăng số bước để xem robot lâu hơn
REALTIME = True        # nếu True → chạy realtime (240Hz ~ 1s thực)
# CAMERA_TRACK = True    # camera tự động bám theo robot

# ---------- Load Env ----------
env = MatchaBalanceEnvUpright(
    urdf_path=URDF_PATH,
    render=True,              # <== GUI bật ở đây
    time_step=1.0 / 240.0,
    max_episode_steps=2000,
    torque_limit=2.0,
    pitch_limit_deg=35.0,
    debug_joints=False,
    symmetric_action=True,
)

# ---------- Load Model ----------
print(f"📦 Loading model from {MODEL_PATH}")
model = PPO.load(MODEL_PATH, env=env)

# ---------- Reset ----------
obs, _ = env.reset()
pitch_list, x_list, xdot_list, rewards = [], [], [], []

# ---------- Optional: setup camera ----------
# if CAMERA_TRACK:
#     p.resetDebugVisualizerCamera(
#         cameraDistance=0.6,
#         cameraYaw=0,
#         cameraPitch=-20,
#         cameraTargetPosition=[0, 0, 0.15],
#     )

print("🎥 Starting GUI simulation... (Press Ctrl+C to exit)")

for i in range(STEPS):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    # Log
    pitch, pitch_rate, x_dot = obs
    pos, orn = env._get_state()[1:3]
    x = pos[0]
    pitch_deg = np.degrees(pitch)

    pitch_list.append(pitch_deg)
    x_list.append(x)
    xdot_list.append(x_dot)
    rewards.append(reward)

    # if CAMERA_TRACK:
    #     cam_target = [x, 0, 0.15]
    #     p.resetDebugVisualizerCamera(cameraDistance=0.6, cameraYaw=0, cameraPitch=-20, cameraTargetPosition=cam_target)

    # Slow motion (visualization)
    if REALTIME:
        time.sleep(1.0 / 240.0)
    else:
        time.sleep(0.002)  # khoảng 500 FPS nếu muốn tua nhanh

    if terminated or truncated:
        print(f"⚠️ Terminated at step {i} | pitch={pitch_deg:.2f}° | x={x:.3f} m | ẋ={x_dot:.3f} m/s")
        break

print(f"\n✅ Average reward: {np.mean(rewards):.3f}")
print(f"Mean |pitch|: {np.mean(np.abs(pitch_list)):.2f}° | Mean |ẋ|: {np.mean(np.abs(xdot_list)):.3f} m/s")

# ---------- Visualization ----------
plt.figure(figsize=(10, 5))
plt.subplot(3, 1, 1)
plt.plot(pitch_list, label="Pitch (°)")
plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
plt.ylabel("Pitch (°)")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(x_list, label="X position (m)")
plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
plt.ylabel("X (m)")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(xdot_list, label="X velocity (m/s)")
plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
plt.ylabel("X_dot (m/s)")
plt.xlabel("Step")
plt.legend()

plt.tight_layout()
plt.show()

env.close()
