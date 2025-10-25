import pybullet as p, pybullet_data, time
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
robot = p.loadURDF(r"D:\FALL\PJ\Matcha\hardware\balance_robot.urdf", [0,0,0.1])
print("Joints:", [p.getJointInfo(robot, j)[1] for j in range(p.getNumJoints(robot))])

# thử apply torque thủ công
wheel_L, wheel_R = 1, 2  # tạm, thay bằng ID bạn in ra
p.setJointMotorControl2(robot, wheel_L, p.VELOCITY_CONTROL, force=0)
p.setJointMotorControl2(robot, wheel_R, p.VELOCITY_CONTROL, force=0)
for _ in range(500):
    p.setJointMotorControl2(robot, wheel_L, p.TORQUE_CONTROL, force=1.0)
    p.setJointMotorControl2(robot, wheel_R, p.TORQUE_CONTROL, force=1.0)
    p.stepSimulation()
    time.sleep(1/240)
