
import os
import sys
import numpy as np
import mujoco

# Run from the project root.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)

HEAL_XML = "robot_descriptions/single_arm_heal_effort_actuation_rs_mj.xml"
FRANKA_XML = "robot_descriptions/franka/scene.xml"

np.set_printoptions(precision=3, suppress=True)


def mujoco_position(xml_path, body_name, joint_names):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Start at the model's default configuration.
    data.qpos[:] = model.qpos0

    for name in joint_names:
        jid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        if jid < 0:
            raise ValueError(f"Joint not found: {name}")
        data.qpos[model.jnt_qposadr[jid]] = 0.0

    mujoco.mj_forward(model, data)

    bid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, body_name
    )
    if bid < 0:
        raise ValueError(f"Body not found: {body_name}")

    return data.xpos[bid].copy()


def show_result(label, position):
    print(f"{label}: {np.array2string(position, precision=3, suppress_small=True)}")


def print_dh_table(name, rows):
    print(f"\n{name} DH table candidate")
    print("Columns: a (m), alpha (rad), d (m), theta offset (rad)")
    print("These are candidate parameters, not yet verified DH tables.")
    print(f"{'a':>10} {'alpha':>12} {'d':>10} {'theta offset':>15}")
    for a, alpha, d, offset in rows:
        print(f"{a:10.3f} {alpha:12.3f} {d:10.3f} {offset:15.3f}")


def main():
    print("=" * 60)
    print("ALL ROBOT RESULTS AT ZERO JOINT ANGLES")
    print("Positions shown in metres, rounded to 3 decimal places")
    print("=" * 60)

    # 1. HEAL MuJoCo
    heal_mj = mujoco_position(
        HEAL_XML, "end_effector",
        [f"joint_{i}" for i in range(1, 7)]
    )
    show_result("1. HEAL MuJoCo EE position", heal_mj)

    # 2. HEAL FK function from scripts/heal_fk.py
    from heal_fk import heal_fk
    heal_fk_T = heal_fk(np.zeros(6))
    heal_fk_pos = heal_fk_T[:3, 3]
    show_result("2. HEAL FK position", heal_fk_pos)
    show_result("   HEAL position difference", heal_fk_pos - heal_mj)

    # 3. Franka MuJoCo
    franka_mj = mujoco_position(
        FRANKA_XML, "hand",
        [f"joint{i}" for i in range(1, 8)]
    )
    show_result("3. Franka MuJoCo hand position", franka_mj)

        # 4. Franka FK function from scripts/franka_fk.py
    import franka_fk as franka_module

    if hasattr(franka_module, "franka_mujoco_fk"):
        franka_fk_pos = franka_module.franka_mujoco_fk(np.zeros(7))
        show_result("4. Franka FK position", franka_fk_pos)
        show_result("   Franka position difference", franka_fk_pos - franka_mj)
    else:
        print("4. Franka MuJoCo FK function not found in franka_fk.py")

    # Existing candidate tables, printed for inspection.
    # They must not be treated as verified standard-DH tables.
    heal_candidate = [
        [0.0, np.pi/2, 0.3208, 0.0],
        [0.3000, np.pi, 0.0, np.pi/2],
        [0.0, np.pi/2, 0.0, 0.0],
        [0.03185, 0.50951, 0.3204, -np.pi/2],
        [0.0, np.pi/2, 0.0654, np.pi],
        [0.0, np.pi, -0.1227, np.pi],
    ]

    franka_candidate = [
        [0.0, 0.0, 0.333, 0.0],
        [0.0, -np.pi/2, 0.0, 0.0],
        [-0.0825, np.pi/2, 0.316, 0.0],
        [-0.0825, np.pi/2, 0.0, 0.0],
        [0.0, -np.pi/2, 0.384, 0.0],
        [0.088, np.pi/2, 0.0, 0.0],
        [0.0, np.pi/2, 0.107, 0.0],
    ]

    print_dh_table("HEAL 6-DOF", heal_candidate)
    print_dh_table("Franka Panda 7-DOF", franka_candidate)

    print("\nFinished. A small position difference is expected from rounding.")
    print("The displayed DH tables are candidates and still need validation.")


if __name__ == "__main__":
    main()
