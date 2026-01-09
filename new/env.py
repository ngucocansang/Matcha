import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import numpy as np
import os

class BalancingRobotEnv(gym.Env):
    def __init__(self, model_path='robot.xml', training=True):
        super().__init__()
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing {model_path}")
            
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.training = training # Chế độ train sẽ bật randomization
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        self.viewer = None
        self.frame_skip = 5
        
        # Lưu lại giá trị mặc định để randomize dựa trên gốc
        self.original_mass = np.copy(self.model.body_mass)
        self.original_friction = np.copy(self.model.geom_friction)
        self.original_gear = np.copy(self.model.actuator_gear)

    def _get_obs(self):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'chassis')
        xmat = self.data.xmat[body_id].reshape(3, 3)
        pitch = np.arctan2(-xmat[2, 0], np.sqrt(xmat[2, 1]**2 + xmat[2, 2]**2))
        
        # Thêm nhiễu nhẹ vào observation khi training để AI không bị "học vẹt" số liệu ảo
        noise = np.random.normal(0, 0.005, size=4) if self.training else 0
        
        obs = np.array([pitch, self.data.qvel[4], self.data.qvel[6], self.data.qvel[7]], dtype=np.float32)
        return np.nan_to_num(obs + noise)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        
        if self.training:
            # --- DOMAIN RANDOMIZATION (SESSION 5) ---
            # Random khối lượng +/- 15%
            self.model.body_mass[:] = self.original_mass * np.random.uniform(0.85, 1.15, size=self.model.nbody)
            # Random ma sát sàn nhà +/- 30%
            self.model.geom_friction[0, 0] = self.original_friction[0, 0] * np.random.uniform(0.7, 1.3)
            # Random hiệu suất motor (giả lập pin yếu)
            self.model.actuator_gear[:, 0] = self.original_gear[:, 0] * np.random.uniform(0.9, 1.1)

        self.data.qpos[0:3] = [0, 0, 0.05]
        self.data.qpos[3:7] = [1, 0, 0, 0]
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
        
        # --- ENHANCED REWARD (SESSION 4) ---
        # cos(theta) giúp reward mượt hơn + phạt năng lượng (action^2) + phạt giật (pitch_vel)
        reward = np.cos(pitch) - 0.01 * np.sum(np.square(action)) - 0.05 * np.abs(obs[1])
        
        terminated = bool(np.abs(pitch) > 0.78 or np.isnan(pitch))
        return obs, reward, terminated, False, {}

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()