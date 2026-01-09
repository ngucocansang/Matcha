import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import os

# Import class môi trường đã viết ở session trước
# Giả sử file trước bạn lưu là balancing_env.py
from env import BalancingRobotEnv

# 1. Khởi tạo môi trường
env = BalancingRobotEnv(model_path='robot.xml')

# 2. Thiết lập TensorBoard log và Checkpoint
log_dir = "./ppo_balancing_tensorboard/"
os.makedirs(log_dir, exist_ok=True)

checkpoint_callback = CheckpointCallback(
  save_freq=5000,
  save_path="./checkpoints/",
  name_prefix="ppo_robot"
)

# 3. Khởi tạo Model PPO
# Hyperparameters: learning_rate có thể chỉnh ở đây (Homework)
model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1, 
    tensorboard_log=log_dir,
    learning_rate=0.0003, # Giá trị mặc định hoặc tùy chỉnh
    n_steps=2048,
    batch_size=64
)

# 4. Huấn luyện (Train)
print("Bắt đầu huấn luyện...")
model.learn(
    total_timesteps=1000000, # Tăng lên 50k-100k để có kết quả tốt hơn 10k
    callback=checkpoint_callback,
    progress_bar=True
)

# 5. Lưu model cuối cùng
model.save("ppo_balancing_final")
print("Đã lưu model!")