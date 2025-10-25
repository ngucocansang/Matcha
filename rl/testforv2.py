import pybullet as p
import pybullet_data
import time

# ===========================================
# 1️⃣ Connect to PyBullet and load environment
# ===========================================
p.connect(p.GUI)                   # Use GUI so you can see it
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# Optional: add a plane for ground
plane_id = p.loadURDF("plane.urdf")

# ===========================================
# 2️⃣ Load your robot URDF
# ===========================================
robot_urdf_path = "D:\FALL\PJ\Matcha\hardware\balance_robot.urdf"   # <-- change path if needed
start_pos = [0, 0, 0.05]
start_orn = p.getQuaternionFromEuler([0, 0, 0])

robot_id = p.loadURDF(robot_urdf_path, start_pos, start_orn, useFixedBase=False)

# ===========================================
# 3️⃣ Identify wheel joints (by name or index)
# ===========================================
# Get joint info to print names
num_joints = p.getNumJoints(robot_id)
for i in range(num_joints):
    print(i, p.getJointInfo(robot_id, i)[1])

# You’ll see joint names like:
# 5 base_to_left_wheel
# 6 base_to_right_wheel

left_wheel_joint = 5
right_wheel_joint = 6

# ===========================================
# 4️⃣ Simple forward motion
# ===========================================
target_velocity = 10.0    # rad/s wheel speed
max_force = 2.0           # motor torque limit

p.setRealTimeSimulation(0)

for _ in range(2000):  # Run for ~8 seconds (at 240 Hz)
    # Set both wheels to same angular velocity
    p.setJointMotorControl2(
        bodyUniqueId=robot_id,
        jointIndex=left_wheel_joint,
        controlMode=p.VELOCITY_CONTROL,
        targetVelocity=target_velocity,
        force=max_force,
    )
    p.setJointMotorControl2(
        bodyUniqueId=robot_id,
        jointIndex=right_wheel_joint,
        controlMode=p.VELOCITY_CONTROL,
        targetVelocity=target_velocity,
        force=max_force,
    )

    p.stepSimulation()
    time.sleep(1 / 240.0)

# ===========================================
# 5️⃣ Disconnect
# ===========================================
p.disconnect()
