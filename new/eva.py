import time
import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO
from env import BalancingRobotEnv

def evaluate():
    # 1. Khởi tạo môi trường (Tắt training để không randomize thông số vật lý lúc test)
    env = BalancingRobotEnv(model_path='robot.xml', training=False)
    
    # 2. Load model Robust nhất (Bạn hãy thay tên file model của bạn vào đây)
    model_path = "checkpoints_sim2real/robust_bot_500000_steps.zip"
    try:
        model = PPO.load(model_path)
        print(f"Đã load model: {model_path}")
    except:
        print("Không tìm thấy file model! Vui lòng kiểm tra lại tên file.")
        return

    obs, _ = env.reset()
    
    # Biến điều khiển Stress Test (đẩy robot bằng lực ảo)
    stress_test = True 
    force_timer = 0

    print("--- Đang chạy đánh giá Model Robust ---")
    print("Mẹo: Bạn có thể dùng chuột kéo robot trong cửa sổ MuJoCo để test độ bền.")

    try:
        while True:
            # Predict hành động (Dùng deterministic=True để điều khiển mượt nhất)
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            # --- LOGIC STRESS TEST (Tự động đẩy robot mỗi 3 giây) ---
            if stress_test:
                force_timer += 1
                if force_timer > 300: # Khoảng mỗi 3 giây (với 100Hz)
                    # Tác động một lực đẩy ngẫu nhiên vào thân robot
                    env.data.qvel[0] += np.random.uniform(-0.5, 0.5) # Đẩy theo trục X
                    print("--> Tác động lực đẩy ngẫu nhiên!")
                    force_timer = 0

            # Hiển thị
            env.render()
            
            # In chỉ số góc Pitch (độ) để theo dõi
            pitch_deg = np.degrees(obs[0])
            if abs(pitch_deg) > 10:
                print(f"Cảnh báo: Góc nghiêng lớn! {pitch_deg:.2f}°")

            if terminated or truncated:
                print("Robot đã ngã! Đang reset...")
                time.sleep(0.5)
                obs, _ = env.reset()
            
            # Khớp với thời gian thực (100Hz)
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("Đã dừng đánh giá.")
        env.close()

if __name__ == "__main__":
    evaluate()