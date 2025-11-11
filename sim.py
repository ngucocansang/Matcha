import pybullet as p
import pybullet_data
import time
import math

# === CONFIG ===
URDF_PATH = r"hardware\balance_robot.urdf"   # hoặc file URDF bạn muốn test
GUI = True
STEP_DT = 1. / 240.
RUN_TIME = 10
WHEEL_VELOCITY = 6.0

# === INIT ===
if GUI:
    p.connect(p.GUI)
else:
    p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setTimeStep(STEP_DT)

# === ENVIRONMENT ===
plane = p.loadURDF("plane.urdf")
robot = p.loadURDF(URDF_PATH, [0, 0, 0.05], p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=False)

print("=== ROBOT INFO ===")
print("Num joints:", p.getNumJoints(robot))
wheel_indices = []
for i in range(p.getNumJoints(robot)):
    jn = p.getJointInfo(robot, i)[1].decode()
    print(f"[{i}] {jn}")
    if "wheel" in jn:
        wheel_indices.append(i)
print("Detected wheel joints:", wheel_indices)


for i in range(p.getNumJoints(robot)):
    jinfo = p.getJointInfo(robot, i)
    axis = p.getJointInfo(robot, i)[13]  # axis local frame
    print(f"[{i}] {jinfo[1].decode()} axis={axis}")


# === CONTROL ===
p.resetBasePositionAndOrientation(robot, [0, 0, 0.05], p.getQuaternionFromEuler([0, 0, 1]))

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

def print_debug_state(step, state, contact_data):
    print(f"\n--- Step {step} ---")
    print(f"Pos (x,y,z): {state['pos']}")
    print(f"RPY (deg):   roll={state['rpy_deg'][0]:.2f}, pitch={state['rpy_deg'][1]:.2f}, yaw={state['rpy_deg'][2]:.2f}")
    print(f"Lin vel:     {state['lin_vel']}")
    print(f"Ang vel:     {state['ang_vel']}")
    if abs(state['rpy_deg'][0]) > 10:
        print("⚠️  WARNING: Robot tilting to side (roll > 10°)")

    # Contact info
    contacts = [c for c in contact_data if c[2] == robot]
    if len(contacts) == 0:
        print("⚠️  No contact with ground!")
    else:
        normals = [c[7] for c in contacts]
        print(f"Contact points: {len(contacts)}, first normal={normals[0]}")

# === MAIN LOOP ===
print("\nStarting simulation... (Press Ctrl+C to stop)\n")
start_time = time.time()
step = 0

while time.time() - start_time < RUN_TIME:
    t = time.time() - start_time

    # simple forward motion
    left_speed = WHEEL_VELOCITY
    right_speed = WHEEL_VELOCITY

    for idx in wheel_indices:
        name = p.getJointInfo(robot, idx)[1].decode()
        p.setJointMotorControl2(
            robot, idx,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=left_speed if "left" in name else right_speed,
            force=3.0
        )

    p.stepSimulation()

    if step % 120 == 0:  # print every ~0.5s
        state = get_base_state()
        contact_data = p.getContactPoints()
        print_debug_state(step, state, contact_data)

    time.sleep(STEP_DT)
    step += 1

# === FINAL ===
state = get_base_state()
print("\n=== FINAL STATE ===")
print(f"Final pos: {state['pos']}")
print(f"Final RPY (deg): {state['rpy_deg']}")
p.disconnect()
