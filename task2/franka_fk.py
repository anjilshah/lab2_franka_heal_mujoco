import mujoco
import numpy as np

XML_PATH = "robot_descriptions/franka/scene.xml"
JOINT_NAMES = [f"joint{i}" for i in range(1, 8)]


def franka_mujoco_fk(q):
    """Calculate Franka hand pose using the MuJoCo model."""
    q = np.asarray(q, dtype=float)

    if q.shape != (7,):
        raise ValueError("Expected 7 joint angles in radians.")

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    data.qpos[:] = model.qpos0

    for name, angle in zip(JOINT_NAMES, q):
        jid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, name
        )

        if jid < 0:
            raise ValueError(f"Joint not found: {name}")

        data.qpos[model.jnt_qposadr[jid]] = angle

    mujoco.mj_forward(model, data)

    hand_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "hand"
    )

    if hand_id < 0:
        raise ValueError("Hand body not found in XML.")

    return data.xpos[hand_id].copy()


def main():
    q_deg = np.array([0, -30, 0, -100, 0, 70, 0], dtype=float)
    q = np.deg2rad(q_deg)

    position = franka_mujoco_fk(q)

    np.set_printoptions(precision=6, suppress=True)

    print("Franka FK — MuJoCo reference")
    print("Joint angles (degrees):", q_deg)
    print("Hand position (m):", position)
    print("Expected position (m): [-0.212, 0.000, 0.926]")

    expected = np.array([-0.212, 0.0, 0.926])
    print("Position error (m):", np.linalg.norm(position - expected))


if __name__ == "__main__":
    main()