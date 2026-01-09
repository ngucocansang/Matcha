import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import numpy as np
import time
import os

class BalancingRobotEnv(gym.Env):
    def __init__(self, model_path='robot.xml'):
        super().__init__()
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing {model_path}")
            
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Obs: [pitch, pitch_vel, l_wheel_vel, r_wheel_vel]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        self.viewer = None
        self.frame_skip = 5 # 100Hz Control (0.002s * 5 = 0.01s)

    def _get_obs(self):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'chassis')
        xmat = self.data.xmat[body_id].reshape(3, 3)
        # Tính Pitch Angle
        pitch = np.arctan2(-xmat[2, 0], np.sqrt(xmat[2, 1]**2 + xmat[2, 2]**2))
        
        # Qvel mapping: 0-2: translation, 3-5: rotation, 6-7: wheels
        pitch_vel = self.data.qvel[4]
        l_vel = self.data.qvel[6]
        r_vel = self.data.qvel[7]
        
        obs = np.array([pitch, pitch_vel, l_vel, r_vel], dtype=np.float32)
        return np.nan_to_num(obs) # Bảo vệ chống NaN

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        
        # SPAWN POSITION: Đặt robot đứng thẳng trên sàn
        self.data.qpos[0:3] = [0, 0, 0.05]  # X, Y, Z
        self.data.qpos[3:7] = [1, 0, 0, 0] # Quaternion (Identity)
        
        # Nhiễu cực nhỏ để robot không bị "đóng băng"
        self.data.qvel[4] = np.random.uniform(-0.02, 0.02)
        
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        action = np.nan_to_num(action)
        self.data.ctrl[:] = action
        
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
            
        obs = self._get_obs()
        pitch = obs[0]
        
        # Reward: 1.0 (alive) - phạt góc nghiêng
        reward = 1.0 - np.abs(pitch)
        
        # Termination: Ngã quá 45 độ hoặc bị văng (NaN)
        terminated = bool(np.abs(pitch) > 0.78 or np.isnan(pitch))
        
        return obs, reward, terminated, False, {}

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

if __name__ == "__main__":
    env = BalancingRobotEnv()
    obs, _ = env.reset()
    print("Môi trường đã sẵn sàng. Chạy thử nghiệm...")
    
    try:
        while True:
            env.render()
            action = env.action_space.sample() # Random action
            obs, reward, done, _, _ = env.step(action)
            
            if done:
                print(f"Robot ngã! Pitch: {np.degrees(obs[0]):.2f}°")
                obs, _ = env.reset()
            time.sleep(0.01)
    except KeyboardInterrupt:
        if env.viewer: env.viewer.close()