import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import numpy as np
import os
import time

class BalancingRobotEnv(gym.Env):
    def __init__(self, model_path='robot.xml', training=True):
        super().__init__()
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không tìm thấy file mô hình tại: {model_path}")
            
        # 1. KHỞI TẠO MUJOCO TRƯỚC (Quan trọng: Phải có trước khi lấy mass/friction)
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        self.training = training
        self.frame_skip = 5 # 100Hz control (0.002s * 5)
        
        # 2. LƯU GIÁ TRỊ GỐC (Để làm mốc Randomize)
        self.orig_mass = np.copy(self.model.body_mass)
        self.orig_friction = np.copy(self.model.geom_friction)
        self.orig_gear = np.copy(self.model.actuator_gear)
        
        # 3. ĐỊNH NGHĨA KHÔNG GIAN (Obs: pitch, pitch_vel, l_vel, r_vel)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        self.viewer = None

    def _get_obs(self):
        # Lấy ma trận xoay của thân robot để tính góc Pitch
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'chassis')
        xmat = self.data.xmat[body_id].reshape(3, 3)
        pitch = np.arctan2(-xmat[2, 0], np.sqrt(xmat[2, 1]**2 + xmat[2, 2]**2))
        
        # Vận tốc: 4 là pitch_vel, 6 & 7 là vận tốc 2 bánh
        obs = np.array([
            pitch, 
            self.data.qvel[4], 
            self.data.qvel[6], 
            self.data.qvel[7]
        ], dtype=np.float32)
        
        # Thêm nhiễu nhẹ vào cảm biến khi training để tăng tính Robust (Session 4)
        if self.training:
            noise = np.random.normal(0, 0.002, size=obs.shape)
            obs += noise
            
        return np.nan_to_num(obs)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        
        # DOMAIN RANDOMIZATION (Session 5)
        if self.training:
            # Random khối lượng các bộ phận +/- 15%
            self.model.body_mass[:] = self.orig_mass * np.random.uniform(0.85, 1.15, size=self.model.nbody)
            # Random ma sát sàn nhà +/- 30%
            self.model.geom_friction[0, 0] = self.orig_friction[0, 0] * np.random.uniform(0.7, 1.3)
            # Random lực motor +/- 10% (giả lập pin yếu/khỏe)
            self.model.actuator_gear[:, 0] = self.orig_gear[:, 0] * np.random.uniform(0.9, 1.1)

        # Đặt vị trí đứng thẳng
        self.data.qpos[0:3] = [0, 0, 0.05] # Z=0.05 để bánh xe chạm đất
        self.data.qpos[3:7] = [1, 0, 0, 0] # Quaternion thẳng đứng
        self.data.qvel[:] = 0
        self.data.qvel[4] = np.random.uniform(-0.02, 0.02) # Nhiễu vận tốc góc ban đầu
        
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        # Bảo vệ chống lỗi số học
        action = np.nan_to_num(action)
        self.data.ctrl[:] = action
        
        # Frame Skipping (Vật lý chạy nhanh hơn AI)
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
            
        obs = self._get_obs()
        pitch = obs[0]
        pitch_vel = obs[1]
        
        # ENHANCED REWARD (Session 4)
        # Khuyến khích đứng thẳng (cos), phạt dùng quá nhiều lực, phạt rung lắc (jerk)
        reward = np.cos(pitch) - 0.01 * np.sum(np.square(action)) - 0.05 * np.abs(pitch_vel)
        
        # Dừng nếu ngã quá 45 độ hoặc lỗi NaN
        terminated = bool(np.abs(pitch) > 0.78 or np.isnan(pitch))
        
        return obs, reward, terminated, False, {}

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        if self.viewer:
            self.viewer.close()

if __name__ == "__main__":
    # Test thử môi trường
    env = BalancingRobotEnv(training=False)
    obs, _ = env.reset()
    try:
        while True:
            env.render()
            action = env.action_space.sample()
            obs, reward, done, _, _ = env.step(action)
            if done:
                obs, _ = env.reset()
            time.sleep(0.01)
    except KeyboardInterrupt:
        env.close()