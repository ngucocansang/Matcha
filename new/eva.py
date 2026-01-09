import time
from stable_baselines3 import PPO
from env import BalancingRobotEnv

env = BalancingRobotEnv(model_path='robot.xml')
# model = PPO.load("ppo_balancing_final")
model = PPO.load('checkpoints\ppo_robot_450000_steps.zip')

obs, _ = env.reset()
print("Đang chạy thử nghiệm model đã train...")

for _ in range(1000):
    # Predict hành động từ bộ não đã học
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    env.render() # Hiển thị 3D
    
    if terminated or truncated:
        time.sleep(0.5)
        obs, _ = env.reset()
    
    time.sleep(0.01)