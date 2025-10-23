# rl/matcha_env.py
import os
import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces

class MatchaBalanceEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        urdf_path:str = "/mnt/data/balance_robot.urdf",  # <-- URDF của bạn
        render:bool = False,
        time_step:float = 1.0/240.0,
        max_episode_steps:int = 2000,
        torque_limit:float = 2.0,
        pitch_limit_deg:float = 35.0,
        debug_joints:bool = False,
        symmetric_action:bool = True,
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

        # Observation: [pitch, pitch_rate, x_dot]
        obs_high = np.array([np.pi, 50.0, 10.0], dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

        # Action: 1D torque (both wheels same) hoặc 2D nếu symmetric_action=False
        if self.symmetric_action:
            self.action_space = spaces.Box(
                low=np.array([-1.0], dtype=np.float32),
                high=np.array([ 1.0], dtype=np.float32),
                dtype=np.float32
            )
        else:
            self.action_space = spaces.Box(
                low=np.array([-1.0, -1.0], dtype=np.float32),
                high=np.array([ 1.0,  1.0], dtype=np.float32),
                dtype=np.float32
            )

        self.physics_client = None
        self.robot_id = None
        self.base_link = 0
        self.wheel_left_joint = None
        self.wheel_right_joint = None
        self.step_count = 0

    def _connect(self):
        if self.render_mode == "human":
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        p.setTimeStep(self.time_step)
        p.setGravity(0, 0, -9.81)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

    def _load_world(self):
        p.resetSimulation()
        p.setTimeStep(self.time_step)
        p.setGravity(0, 0, -9.81)
        self.plane_id = p.loadURDF("plane.urdf")
        start_pos = [0, 0, 0.1]
        start_orn = p.getQuaternionFromEuler([0, 0, 0])
        assert os.path.isfile(self.urdf_path), f"URDF not found: {self.urdf_path}"
        self.robot_id = p.loadURDF(self.urdf_path, start_pos, start_orn, useFixedBase=False)

        if self.debug_joints:
            n_j = p.getNumJoints(self.robot_id)
            print(f"[DEBUG] Num joints: {n_j}")
            for j in range(n_j):
                ji = p.getJointInfo(self.robot_id, j)
                print(j, ji[1].decode(), "type=", ji[2], "parentIndex=", ji[16])

        # ====== ĐÃ CHỈNH THEO URDF CỦA BẠN ======
        wheel_left_name = "base_to_left_wheel"
        wheel_right_name = "base_to_right_wheel"

        self.wheel_left_joint = self._find_joint_by_name(wheel_left_name)
        self.wheel_right_joint = self._find_joint_by_name(wheel_right_name)
        assert self.wheel_left_joint is not None, "Left wheel joint not found"
        assert self.wheel_right_joint is not None, "Right wheel joint not found"

        # Vô hiệu điều khiển mặc định
        for j in [self.wheel_left_joint, self.wheel_right_joint]:
            p.setJointMotorControl2(self.robot_id, j, p.VELOCITY_CONTROL, force=0)

    def _find_joint_by_name(self, name:str):
        n_j = p.getNumJoints(self.robot_id)
        for j in range(n_j):
            ji = p.getJointInfo(self.robot_id, j)
            if ji[1].decode() == name:
                return j
        return None

    def _get_state(self):
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        lin_vel, ang_vel = p.getBaseVelocity(self.robot_id)
        roll, pitch, yaw = p.getEulerFromQuaternion(orn)

        # Theo URDF, trục bánh là Y (0 1 0), nên pitch là quanh Y
        pitch_rate = ang_vel[1]  # rad/s quanh Y
        x_dot = lin_vel[0]       # m/s (giả sử X là hướng tiến)

        obs = np.array([pitch, pitch_rate, x_dot], dtype=np.float32)
        return obs, pos, (roll, pitch, yaw)

    def _apply_action(self, action):
        if self.symmetric_action:
            torque = float(np.clip(action[0], -1.0, 1.0)) * self.torque_limit
            tau_l, tau_r = torque, torque
        else:
            a = np.clip(action, -1.0, 1.0)
            tau_l = float(a[0]) * self.torque_limit
            tau_r = float(a[1]) * self.torque_limit

        p.setJointMotorControl2(self.robot_id, self.wheel_left_joint,
                                controlMode=p.TORQUE_CONTROL, force=tau_l)
        p.setJointMotorControl2(self.robot_id, self.wheel_right_joint,
                                controlMode=p.TORQUE_CONTROL, force=tau_r)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.physics_client is None:
            self._connect()
        self._load_world()
        self.step_count = 0

        # Random pitch nhỏ ban đầu để đa dạng hoá
        init_pitch = self.np_random.uniform(low=-0.05, high=0.05)
        cur_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        new_orn = p.getQuaternionFromEuler([0, init_pitch, 0])
        p.resetBasePositionAndOrientation(self.robot_id, cur_pos, new_orn)

        obs, _, _ = self._get_state()
        info = {}
        return obs, info

    def step(self, action):
        self._apply_action(action)
        p.stepSimulation()
        self.step_count += 1

        obs, pos, eul = self._get_state()
        pitch = obs[0]
        pitch_rate = obs[1]
        x = pos[0]

        alive_bonus = 1.0
        w_th, w_dth, w_x = 2.0, 0.1, 0.01
        reward = alive_bonus - (w_th * pitch * pitch) - (w_dth * pitch_rate * pitch_rate) - (w_x * x * x)

        terminated = (abs(pitch) > self.pitch_limit)
        truncated = (self.step_count >= self.max_episode_steps)

        info = {}
        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None
