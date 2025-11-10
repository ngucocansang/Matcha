# ==========================================================
# MatchaBalanceEnvUpright - Strict upright balance environment (v3)
# ==========================================================
import os
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces, Env


class MatchaBalanceEnvTuned(Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        urdf_path: str,
        render: bool = False,
        time_step: float = 1.0 / 240.0,
        max_episode_steps: int = 2000,
        torque_limit: float = 2.0,
        pitch_limit_deg: float = 35.0,
        debug_joints: bool = False,
        symmetric_action: bool = True,
    ):
        super().__init__()
        self.urdf_path = urdf_path
        self.render_mode = "human" if render else None
        self.time_step = time_step
        self.max_episode_steps = max_episode_steps
        self.torque_limit = torque_limit
        self.pitch_limit = np.deg2rad(pitch_limit_deg)
        self.debug_joints = debug_joints
        self.symmetric_action = symmetric_action

        # ---------- Observation space ----------
        # [roll, pitch, roll_rate, pitch_rate, x_dot]
        obs_high = np.array([np.pi, np.pi, 50.0, 50.0, 10.0], dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

        # ---------- Action space ----------
        if self.symmetric_action:
            # One torque value applied symmetrically to both wheels
            self.action_space = spaces.Box(
                low=np.array([-1.0], dtype=np.float32),
                high=np.array([1.0], dtype=np.float32),
                dtype=np.float32,
            )
        else:
            # Separate torques for each wheel
            self.action_space = spaces.Box(
                low=np.array([-1.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )

        self.physics_client = None
        self.robot_id = None
        self.step_count = 0
        self.pitch_prev = 0.0
        self.pitch_ema = 0.0
        self.x_origin = 0.0
        self.ema_alpha = 0.98  # low-pass filter

    # ---------- Setup ----------

    # ---------- PyBullet Setup ----------
    def _connect(self):
        self.physics_client = p.connect(p.GUI if self.render_mode else p.DIRECT)
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.time_step)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

    def _load_world(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.time_step)
        p.loadURDF("plane.urdf")

        assert os.path.isfile(self.urdf_path), f"URDF not found: {self.urdf_path}"
        start_pos = [0, 0, 0.05]
        start_orn = p.getQuaternionFromEuler([0, 0, 0])
        self.robot_id = p.loadURDF(
            self.urdf_path, start_pos, start_orn, useFixedBase=False
        )

        self.wheel_left_joint = self._find_joint("base_to_left_wheel")
        self.wheel_right_joint = self._find_joint("base_to_right_wheel")

        for j in [self.wheel_left_joint, self.wheel_right_joint]:
            p.setJointMotorControl2(self.robot_id, j, p.VELOCITY_CONTROL, force=0)

        # Set friction for realistic wheel-ground interaction
        p.changeDynamics(self.robot_id, self.wheel_left_joint, lateralFriction=1.0)
        p.changeDynamics(self.robot_id, self.wheel_right_joint, lateralFriction=1.0)

    def _find_joint(self, name: str):
        for j in range(p.getNumJoints(self.robot_id)):
            ji = p.getJointInfo(self.robot_id, j)
            if ji[1].decode() == name:
                return j
        return None

    # ---------- State ----------
    def _get_state(self):
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        lin_vel, ang_vel = p.getBaseVelocity(self.robot_id)
        roll, pitch, yaw = p.getEulerFromQuaternion(orn)

        roll_rate = ang_vel[0]
        pitch_rate = ang_vel[1]
        x_dot = lin_vel[0]

        obs = np.array([roll, pitch, roll_rate, pitch_rate, x_dot], dtype=np.float32)
        return obs, pos, (roll, pitch, yaw)

    # ---------- Action ----------
    def _apply_action(self, action):
        if self.symmetric_action:
            torque = float(np.clip(action[0], -1, 1)) * self.torque_limit
            tau_l = tau_r = torque
        else:
            a = np.clip(action, -1, 1)
            tau_l, tau_r = a[0] * self.torque_limit, a[1] * self.torque_limit
        p.setJointMotorControl2(self.robot_id, self.wheel_left_joint, p.TORQUE_CONTROL, force=tau_l)
        p.setJointMotorControl2(self.robot_id, self.wheel_right_joint, p.TORQUE_CONTROL, force=tau_r)

    # ---------- Gym interface ----------

        p.setJointMotorControl2(
            self.robot_id, self.wheel_left_joint, p.TORQUE_CONTROL, force=tau_l
        )
        p.setJointMotorControl2(
            self.robot_id, self.wheel_right_joint, p.TORQUE_CONTROL, force=tau_r
        )

    # ---------- Gym Interface ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.physics_client is None:
            self._connect()
        self._load_world()

        self.step_count = 0

        # Start nearly upright
        init_pitch = 0.0
        cur_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        new_orn = p.getQuaternionFromEuler([0, init_pitch, 0])
        p.resetBasePositionAndOrientation(self.robot_id, cur_pos, new_orn)

        obs, _, _ = self._get_state()
        self.pitch_prev = obs[1]
        return obs, {}

    def step(self, action):
        self._apply_action(action)
        p.stepSimulation()
        self.step_count += 1

        obs, pos, eul = self._get_state()
        pitch, pitch_rate, x_dot = obs
        x, _, _ = pos
        x_err = x - self.x_origin

        # update moving average to detect "leaned" posture
        self.pitch_ema = self.ema_alpha * self.pitch_ema + (1 - self.ema_alpha) * pitch

        # ------------- Reward shaping -------------
        alive_bonus = 0.2  # smaller survival bonus
        w_th, w_dth, w_vx, w_xpos, w_lean = 2.0, 0.05, 0.20, 0.10, 1.0

        # limit pitch penalty (Huber-like)
        pitch_pen = min(pitch * pitch, (0.2 ** 2))  # saturate beyond ~11°

        roll, pitch, roll_rate, pitch_rate, x_dot = obs

        # ---------- Reward ----------
        alive_bonus = 1.0
        w_pitch = 1.5
        w_roll = 0.8
        w_pitch_rate = 0.05
        w_roll_rate = 0.05
        w_x = 0.01

        # reward for moving pitch toward 0 (more upright)
        pitch_delta_reward = 0.5 * (abs(self.pitch_prev) - abs(pitch))
        self.pitch_prev = pitch

        reward = (
            alive_bonus
            - w_th * pitch_pen
            - w_dth * (pitch_rate ** 2)
            - w_vx * (x_dot ** 2)
            - w_xpos * (x_err ** 2)
            - w_lean * (self.pitch_ema ** 2)
            + pitch_delta_reward
        )

        # ------------- Termination -------------
        pitch_limit = self.pitch_limit
        x_limit, v_limit = 0.6, 2.0  # m, m/s

        terminated = (
            abs(pitch) > pitch_limit
            or abs(x_err) > x_limit
            or abs(x_dot) > v_limit
        )
        truncated = self.step_count >= self.max_episode_steps

        if terminated:
            reward -= 10.0  # heavy penalty for fall

            - w_pitch * (pitch ** 2)
            - w_roll * (roll ** 2)
            - w_pitch_rate * (pitch_rate ** 2)
            - w_roll_rate * (roll_rate ** 2)
            - w_x * (x_dot ** 2)
            + pitch_delta_reward
        )

        # ---------- Termination ----------
        terminated = (
            abs(pitch) > self.pitch_limit or abs(roll) > np.deg2rad(25)
        )
        truncated = self.step_count >= self.max_episode_steps

        return obs, float(reward), terminated, truncated, {}

    def close(self):
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None
