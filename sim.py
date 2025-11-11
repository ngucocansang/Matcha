import pybullet as p
import pybullet_data
import time
import math

# === CONFIG ===
URDF_PATH = r"hardware\balance_robot.urdf"  # path to your URDF
GUI = True
STEP_DT = 1. / 240.
RUN_TIME = 10  # seconds
WHEEL_VELOCITY = 6.0  # rad/s
WHEEL_FORCE = 3.0  # max torque

# === INIT ===
physics_client = p.connect(p.GUI if GUI else p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setTimeStep(STEP_DT)

# === LOAD ENVIRONMENT ===
plane = p.loadURDF("plane.urdf")
robot_start_pos = [0, 0, 0.05]
robot_start_orn = p.getQuaternionFromEuler([0, 0, 0])
robot = p.loadURDF(URDF_PATH, robot_start_pos, robot_start_orn, useFixedBase=False)

# === GET JOINT INFO ===
wheel_indices = []
print("=== ROBOT JOINTS ===")
for i in range(p.getNumJoints(robot)):
    info = p.getJointInfo(robot, i)
    j_name = info[1].decode()
    print(f"[{i}] {j_name}, type={info[2]}, axis={info[13]}")
    if "wheel" in j_name:
        wheel_indices.append(i)
print("Wheel joint indices:", wheel_indices)

# === UTILITY FUNCTIONS ===
def get_base_state():
    pos, orn = p.getBasePositionAndOrientation(robot)
    lin_vel, ang_vel = p.getBaseVelocity(robot)
    roll, pitch, yaw = p.getEulerFromQuaternion(orn)
    return {
        "pos": pos,
        "rpy_deg": [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)],
        "lin_vel": lin_vel,
        "ang_vel": ang_vel
    }

def print_debug_state(step, state, contacts):
    print(f"\n--- Step {step} ---")
    print(f"Position: {state['pos']}")
    print(f"Orientation (deg): roll={state['rpy_deg'][0]:.2f}, pitch={state['rpy_deg'][1]:.2f}, yaw={state['rpy_deg'][2]:.2f}")
    print(f"Linear velocity: {state['lin_vel']}")
    print(f"Angular velocity: {state['ang_vel']}")
    if abs(state['rpy_deg'][0]) > 10:
        print("⚠️  WARNING: Robot tilting to side (roll > 10°)")
    if len(contacts) == 0:
        print("⚠️  No contact with ground!")
    else:
        normals = [c[7] for c in contacts]
        print(f"Contact points: {len(contacts)}, first normal={normals[0]}")

# === MAIN SIMULATION LOOP ===
print("\nStarting simulation... (Press Ctrl+C to stop)\n")
start_time = time.time()
step = 0

while time.time() - start_time < RUN_TIME:
    # simple forward motion
    for idx in wheel_indices:
        j_name = p.getJointInfo(robot, idx)[1].decode()
        target_vel = WHEEL_VELOCITY if "left" in j_name else WHEEL_VELOCITY
        p.setJointMotorControl2(
            robot, idx,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=target_vel,
            force=WHEEL_FORCE
        )

    p.stepSimulation()

    if step % 120 == 0:  # every ~0.5s
        state = get_base_state()
        contacts = p.getContactPoints()
        print_debug_state(step, state, contacts)

    time.sleep(STEP_DT)
    step += 1

# === FINAL STATE ===
state = get_base_state()
print("\n=== FINAL STATE ===")
print(f"Position: {state['pos']}")
print(f"Orientation (deg): {state['rpy_deg']}")
p.disconnect()
