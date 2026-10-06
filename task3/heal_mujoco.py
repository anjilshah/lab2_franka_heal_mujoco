import numpy as np
import mujoco


# ---------------------------------------------------------
# Load HEAL MuJoCo model
# ---------------------------------------------------------

xml_path = "robot_descriptions/single_arm_heal_effort_actuation_rs_mj.xml"

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)


# ---------------------------------------------------------
# Set the same joint configuration as Python FK
# ---------------------------------------------------------

q = np.array([
    0.0,   # joint_1
    0.0,   # joint_2
    0.0,   # joint_3
    0.0,   # joint_4
    0.0,   # joint_5
    0.0    # joint_6
])

data.qpos[:] = q


# ---------------------------------------------------------
# Run MuJoCo forward kinematics
# ---------------------------------------------------------

mujoco.mj_forward(model, data)


# ---------------------------------------------------------
# Get end-effector position
# ---------------------------------------------------------

ee_id = model.body("end_effector").id

mujoco_position = data.xpos[ee_id]


# ---------------------------------------------------------
# Print result
# ---------------------------------------------------------

np.set_printoptions(
    precision=3,
    suppress=True
)

print("HEAL MuJoCo Forward Kinematics")
print("==============================")

print("\nJoint configuration:")
print(q)

print("\nEnd-effector position from MuJoCo:")

print("x =", mujoco_position[0])
print("y =", mujoco_position[1])
print("z =", mujoco_position[2])