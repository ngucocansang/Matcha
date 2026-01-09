import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

# Import class môi trường từ file env.py
from env import BalancingRobotEnv

def make_env(rank, seed=0):
    """
    Hàm tạo môi trường hỗ trợ đa luồng
    """
    def _init():
        # Khởi tạo env với chế độ training=True để bật Domain Randomization
        env = BalancingRobotEnv(model_path='robot.xml', training=True)
        # Không dùng env.seed(seed + rank) vì Gymnasium dùng np.random trực tiếp
        return env
    return _init

if __name__ == "__main__":
    # 1. Thiết lập các đường dẫn lưu trữ
    log_dir = "./ppo_sim2real_tensorboard/"
    checkpoint_dir = "./checkpoints_sim2real/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 2. Khởi tạo môi trường song song (Vectorized Environment)
    # n_envs=4 giúp robot học nhanh gấp 4 lần và đối mặt với 4 cấu hình vật lý khác nhau cùng lúc
    num_cpu = 4 
    env = make_vec_env(lambda: BalancingRobotEnv(training=True), n_envs=num_cpu, seed=42)

    # 3. Thiết lập Callback để lưu model định kỳ
    checkpoint_callback = CheckpointCallback(
        save_freq=25000,           # Lưu mỗi 25k steps (tổng của các env)
        save_path=checkpoint_dir,
        name_prefix="robust_bot"
    )

    # 4. Khởi tạo thuật toán PPO với các tham số cho Sim-to-Real
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=0.00025,    # Giảm nhẹ LR để học ổn định trong môi trường biến động
        n_steps=2048,             # Số bước thu thập dữ liệu trước khi cập nhật
        batch_size=128,           # Tăng batch size để gradient mượt hơn
        n_epochs=10,              # Số lần học lại trên một tập dữ liệu
        gamma=0.99,               # Hệ số chiết khấu
        gae_lambda=0.95,
        clip_range=0.2,           # Giới hạn cập nhật policy
        ent_coef=0.01,            # Khuyến khích khám phá (Exploration)
        device="auto"             # Tự động chọn GPU nếu có
    )

    # 5. Bắt đầu huấn luyện
    print("--- Bắt đầu huấn luyện Robust Policy (Session 5) ---")
    print(f"Chạy song song {num_cpu} môi trường với Domain Randomization...")
    
    try:
        model.learn(
            total_timesteps=500000, # 500k steps là đủ để hội tụ bền vững
            callback=checkpoint_callback,
            progress_bar=True
        )
        
        # 6. Lưu model cuối cùng
        model.save("ppo_balancing_robust_final")
        print("--- Huấn luyện hoàn tất và đã lưu model ---")

    except KeyboardInterrupt:
        print("--- Đã dừng huấn luyện thủ công. Đang lưu checkpoint hiện tại... ---")
        model.save("ppo_balancing_interrupted")