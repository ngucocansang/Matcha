import pybullet as p 
import os 
import pybullet_data
import gymnasium as gym 
import numpy as np 

class Matcha_ppo_env(gym.Env):
    def __init__(
        self, 
        urdf_path:str,
        time_step: float = 1/200 , #200hz
        torque_limit: int =  0.13734, #theo 70
        max_episode_step: int = 20000,
        pitch_limit_deg: int = 35, #góc tính từ lý thuyết 72 độ thì tính góc limit thử thách sẽ 35 độ
        debug_joints: bool = False,
    ):
        super().__init__()
        self.urdf_path = urdf_path
        self.time_step = time_step
        self.torque_limit = torque_limit
        self.max_episode_step = max_episode_step
        self.pich_limi_deg = np.deg2rad(pitch_limit_deg)
        self.debug_joints = debug_joints
        
        #observation 
        obs_high = np.array([np.pi, 50.0, 10.0], dtype = np.float32)
        #chọn 50 limit cho pitch rate (tốc độ nghiêng)
        #chọn 10 cho x_dot - vận tốc robot theo trục ngang 
        self.observation_space = gym.spaces.Box(-obs_high, obs_high, dtype=np.float32)
        
        #action 
        if self.symmetric_acton: 
            self.action_space = gym.spaces.Box(
                low = np.array([-1.0], dtype=np.float32),
                high= np.array([1.0], dtype=np.float32),
                dtype=np.float32,
            )
        else:
            self.action_space = gym.spaces.Box(
                low = np.array([-1.0, -1.0], dtype = np.float32),
                high = np.array([1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )
        
        self.physics_client = None
        self.robot_id = None
        self.step_count = 0.0
        self.pich_prev = 0.0
        self.pitch_ema = 0.0
        self.x_origin = 0.0
        self.ema_aplpha = 0.9048 #Lấy dt/tổng thời gian trung bình(thường lấy dtx10)
        
        #setup
        def _connect(self):
            self.physics_client = p.connect(p.GUI) 
            p.setGravity(0,0,-9.81)
            p.setTimeStep(self.time_step)
            p.getSearchPath(pybullet_data.getDataPath())
        def load_world(self):
            p.resetSimulation()
            p.setGravity(0,0,-9.81)
            p.setTimeStep(self.time_step)
            p.loadURDF("plane.urdf")
            assert os.path.isfile(self.urdf_path), f"URDF not found: {self.urdf_path}"
            start_pos = [0,0,0.1]
            start_orn = p.getQuaternionFromEuler([0,0,0])
            self.robot_id = p.loadURDF(self.urdf_path, start_pos, start_orn)
            
            self.wheel_left_joint = self._find_joint("base_to_left_wheel")
            self.wheel_right_joint
                   
        
        