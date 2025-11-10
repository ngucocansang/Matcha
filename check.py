import pybullet as p
import pybullet_data
import math

# ================= CONFIG =================
URDF_PATH = r"hardware\balance_robot.urdf"  # Path to your URDF
GUI = True

# ================= INIT =================
if GUI:
    physicsClient = p.connect(p.GUI)
else:
    physicsClient = p.connect(p.DIRECT)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# Load plane and robot
plane = p.loadURDF("plane.urdf")
robot = p.loadURDF(URDF_PATH, [0, 0, 0.05], p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=False)

print("\n=== ROBOT INFO ===")
num_joints = p.getNumJoints(robot)
print(f"Total joints: {num_joints}")

total_mass = 0.0
for i in range(num_joints):
    info = p.getJointInfo(robot, i)
    joint_name = info[1].decode()              # joint name
    joint_type = info[2]                       # joint type as int
    child_link_name = info[12].decode()        # child link name
    parent_index = info[16]                     # parent index (int)
    axis = info[13]                             # axis (tuple)
    print(f"[{i}] {joint_name} | type={joint_type} | parent_index={parent_index} -> child={child_link_name} | axis={axis}")

# Print all link masses and compute total mass
print("\n=== LINK MASS INFO ===")
for link_idx in range(-1, num_joints):  # -1 = base link
    dyn = p.getDynamicsInfo(robot, link_idx)
    mass = dyn[0]
    local_inertia_diag = dyn[2]
    link_name = "base_link" if link_idx == -1 else p.getJointInfo(robot, link_idx)[12].decode()
    print(f"{link_name}: mass={mass} kg, local_inertia_diag={local_inertia_diag}")
    total_mass += mass

print(f"\nTotal robot mass: {total_mass:.3f} kg")

# Optional: check base position
base_pos, _ = p.getBasePositionAndOrientation(robot)
print(f"Base position: {base_pos}")

p.disconnect()