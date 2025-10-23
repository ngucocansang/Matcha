# ==========================================================
# MatchaBalanceEnv v3 - Vision-only (camera-based) version
# ==========================================================
import os
import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces

class MatchaBalanceCamEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        urdf_path: str,
        render: bool = False,
        image_size=(64, 64),
        time_step: float = 1.0 / 240.0,
        max_episode_steps: int = 2000,
        torque_limit: float = 2.0,
        pitch_limit_deg: float = 35.0,
        symmetric_action: bool = True,
    ):
        super().__init__()
        self.urdf_path = urdf_path
        self.render_mode = "human" if render else None
        self.img_w, self.img_h = image_size
        self.time_step = time_step
        self.max_episode_steps = max_episode_steps
        self.torque_limit = torque_limit
        self.pitch_limit = np.deg2rad(pitch_limit_deg)
        self.symmetric_action = symmetric_action

        # ===== Observation: RGB camera image =====
        obs_shape = (self.img_h, self.img_w, 3)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8
        )

        # ===== Action: 1D or 2D torque =====
        if self.symmetric_action:
            self.action_space = spaces.Box(
                low=np.array([-1.0], dtype=np.float32),
                high=np.array([1.0], dtype=np.float32),
                dtype=np.float32,
            )
        else:
            self.action_space = spaces.Box(
                low=np.array([-1.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )

        self.physics_client = None
        self.robot_id = None
        self.step_count = 0

    # --------------------------------------------------------
    def _connect(self):
        if self.render_mode == "human":
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.time_step)

    def _load_world(self):
        p.resetSimulation()
        p.setTimeStep(self.time_step)
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        start_pos = [0, 0, 0.1]
        start_orn = p.getQuaternionFromEuler([0, 0, 0])
        assert os.path.isfile(self.urdf_path), f"URDF not found: {self.urdf_path}"
        self.robot_id = p.loadURDF(self.urdf_path, start_pos, start_orn, useFixedBase=False)

        # Get joints
        n_j = p.getNumJoints(self.robot_id)
        for j in range(n_j):
            info = p.getJointInfo(self.robot_id, j)
            if info[1].decode() == "base_to_left_wheel":
                self.wheel_left_joint = j
            elif info[1].decode() == "base_to_right_wheel":
                self.wheel_right_joint = j

        # Disable default motor control
        for j in [self.wheel_left_joint, self.wheel_right_joint]:
            p.setJointMotorControl2(self.robot_id, j, p.VELOCITY_CONTROL, force=0)

    # --------------------------------------------------------
    def _apply_action(self, action):
        if self.symmetric_action:
            torque = float(np.clip(action[0], -1, 1)) * self.torque_limit
            tau_l, tau_r = torque, torque
        else:
            a = np.clip(action, -1, 1)
            tau_l = float(a[0]) * self.torque_limit
            tau_r = float(a[1]) * self.torque_limit

        p.setJointMotorControl2(self.robot_id, self.wheel_left_joint, p.TORQUE_CONTROL, force=tau_l)
        p.setJointMotorControl2(self.robot_id, self.wheel_right_joint, p.TORQUE_CONTROL, force=tau_r)

    # --------------------------------------------------------
    def _get_camera_obs(self):
        """Render RGB image (84x84) from a fixed camera."""
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        cam_target = [pos[0] + 0.3, pos[1], pos[2]]
        cam_pos = [pos[0] - 0.3, pos[1], pos[2] + 0.2]
        view = p.computeViewMatrix(
            cameraEyePosition=cam_pos,
            cameraTargetPosition=cam_target,
            cameraUpVector=[0, 0, 1],
        )
        proj = p.computeProjectionMatrixFOV(
            fov=60, aspect=self.img_w / self.img_h, nearVal=0.01, farVal=2.0
        )
        img = p.getCameraImage(
            self.img_w, self.img_h, viewMatrix=view, projectionMatrix=proj
        )[2]

        # ✅ PyBullet returns RGBA → convert to RGB
        img = np.array(img, dtype=np.uint8)[..., :3]   # <-- FIX HERE

        return img

    # --------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.physics_client is None:
            self._connect()
        self._load_world()
        self.step_count = 0

        # Random small tilt
        init_pitch = self.np_random.uniform(low=-0.05, high=0.05)
        cur_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        new_orn = p.getQuaternionFromEuler([0, init_pitch, 0])
        p.resetBasePositionAndOrientation(self.robot_id, cur_pos, new_orn)

        obs = self._get_camera_obs()
        info = {}
        return obs, info

    def step(self, action):
        self._apply_action(action)
        p.stepSimulation()
        self.step_count += 1

        obs = self._get_camera_obs()

        # Reward from base pitch angle
        _, orn = p.getBasePositionAndOrientation(self.robot_id)
        roll, pitch, yaw = p.getEulerFromQuaternion(orn)
        alive_bonus = 1.0
        w_th = 2.0
        reward = alive_bonus - (w_th * pitch * pitch)

        terminated = abs(pitch) > self.pitch_limit
        truncated = self.step_count >= self.max_episode_steps

        info = {"pitch": pitch}
        return obs, reward, terminated, truncated, info

    def close(self):
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None
