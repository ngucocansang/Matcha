import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from env import BalancingRobotEnv

if __name__ == "__main__":
    # 1. Khởi tạo 4 môi trường song song (VecEnv)
    # n_envs=4 giúp tăng tốc độ train và tăng tính đa dạng của dữ liệu
    env = make_vec_env(lambda: BalancingRobotEnv(training=True), n_envs=4)

    log_dir = "./ppo_sim2real_tensorboard/"
    os.makedirs(log_dir, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=10000, # Lưu mỗi 10k steps (chia cho 4 env thực tế là 2500 lượt update)
        save_path="./checkpoints_sim2real/",
        name_prefix="robust_bot"
    )

    # 2. Khởi tạo PPO với tham số ổn định
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.00025, # Giảm nhẹ LR để tránh sụp đổ Reward như PPO_3
        n_steps=2048,
        batch_size=128,        # Tăng batch_size để gradient ổn định hơn
        tensorboard_log=log_dir,
        device="auto"
    )

    print("Bắt đầu huấn luyện Sim-to-Real...")
    # Với randomization, bạn chỉ cần khoảng 300k - 500k steps là đủ hội tụ bền vững
    model.learn(total_timesteps=500000, callback=checkpoint_callback, progress_bar=True)

    model.save("ppo_balancing_sim2real_final")
    print("Huấn luyện hoàn tất!")