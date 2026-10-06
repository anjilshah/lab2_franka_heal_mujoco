
import time
from pathlib import Path
import tkinter as tk
from tkinter.scrolledtext import ScrolledText

import mujoco
import mujoco.viewer
import numpy as np

from task2.franka_fk import franka_mujoco_fk
from task2.heal_fk import heal_fk


ROOT = Path(__file__).resolve().parents[1]

FRANKA_XML = ROOT / "robot_descriptions/franka/scene.xml"
HEAL_XML = ROOT / "robot_descriptions/single_arm_heal_effort_actuation_rs_mj.xml"

FRANKA_JOINTS = [f"joint{i}" for i in range(1, 8)]
HEAL_JOINTS = [f"joint_{i}" for i in range(1, 7)]

FRANKA_DH = [
    [0,       0,          .333,  0],
    [0,      -np.pi/2,    0,     0],
    [-.0825,  np.pi/2,    .316,  0],
    [-.0825,  np.pi/2,    0,     0],
    [0,      -np.pi/2,    .384,  0],
    [.088,    np.pi/2,    0,     0],
    [0,       np.pi/2,    .107,  0],
]


def dh_transform(a, alpha, d, theta):
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)

    return np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0, sa, ca, d],
        [0, 0, 0, 1],
    ], dtype=float)


def franka_dh_fk(q):
    T = np.eye(4)

    for (a, alpha, d, offset), angle in zip(FRANKA_DH, q):
        T = T @ dh_transform(a, alpha, d, angle + offset)

    return T


def joint_id(model, name):
    jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    if jid < 0:
        raise ValueError(f"Joint '{name}' was not found in the XML.")

    return jid


def get_model(xml_path, joint_names):
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0

    info = []

    for name in joint_names:
        jid = joint_id(model, name)

        if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError(f"{name} is not a hinge joint.")

        axis = model.jnt_axis[jid]

        if model.jnt_limited[jid]:
            low, high = model.jnt_range[jid]
            low_deg, high_deg = np.rad2deg([low, high])
        else:
            low_deg, high_deg = -180.0, 180.0

        info.append({
            "name": name,
            "axis": axis.copy(),
            "limited": bool(model.jnt_limited[jid]),
            "low": float(low_deg),
            "high": float(high_deg),
        })

    return model, data, info


def ask_angles(robot_name, info):
    print("\n" + "=" * 64)
    print(f"{robot_name}: JOINT ANGLE INPUT")
    print("=" * 64)

    print("- Enter angles in degrees.")
    print("- The axis shown is the joint axis defined in the MuJoCo XML.")
    print("- Positive rotation follows the right-hand rule.")
    print("- Limits are read from the loaded model.")
    print("- Simulation only: do not send these values directly to hardware.")
    print()

    angles = []

    for i, item in enumerate(info, 1):
        print(f"Joint {i}: {item['name']}")
        print("  XML axis:", np.array2string(item["axis"], precision=3))

        low = item["low"] if item["limited"] else -180.0
        high = item["high"] if item["limited"] else 180.0

        if item["limited"]:
            print(f"  Model range: {low:.2f}° to {high:.2f}°")
        else:
            print("  Model range: unlimited in XML; input restricted to ±180°")

        while True:
            try:
                value = float(input("  Angle (degrees): "))

                if not np.isfinite(value) or not low <= value <= high:
                    print(
                        f"  Enter a finite angle between "
                        f"{low:.2f}° and {high:.2f}°."
                    )
                    continue

                angles.append(value)
                break

            except ValueError:
                print("  Enter a valid number, e.g. 30 or -15.5.")

    return np.deg2rad(np.asarray(angles, dtype=float))


def set_joint_angles(model, data, names, q):
    data.qpos[:] = model.qpos0

    for name, angle in zip(names, q):
        jid = joint_id(model, name)
        data.qpos[model.jnt_qposadr[jid]] = angle

    mujoco.mj_forward(model, data)


def body_pose(model, data, body_name):
    bid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        body_name,
    )

    if bid < 0:
        raise ValueError(f"End-effector body '{body_name}' not found.")

    return (
        data.xpos[bid].copy(),
        data.xmat[bid].reshape(3, 3).copy(),
    )


def show_viewer(model, data, title):
    import threading

    def run_viewer():
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.lookat[:] = data.xpos[1]

            while viewer.is_running():
                viewer.sync()
                time.sleep(1 / 60)

    thread = threading.Thread(
        target=run_viewer,
        daemon=True,
        name=title,
    )
    thread.start()

    time.sleep(1)

def show_dashboard(
    title,
    dh_description,
    dh_T,
    mujoco_pos,
    mujoco_rot,
    extra="",
):
    root = tk.Tk()
    root.title(title)
    root.geometry("760x700")

    text = ScrolledText(
        root,
        wrap=tk.WORD,
        font=("monospace", 10),
    )

    text.pack(
        fill=tk.BOTH,
        expand=True,
        padx=10,
        pady=10,
    )

    text.insert(tk.END, title + "\n")
    text.insert(tk.END, "=" * 64 + "\n\n")

    text.insert(tk.END, dh_description + "\n\n")

    text.insert(
        tk.END,
        "FK transformation:\n",
    )
    text.insert(
        tk.END,
        np.array2string(
            dh_T,
            precision=5,
            suppress_small=True,
        ),
    )

    text.insert(
        tk.END,
        "\n\nFK position (m):\n",
    )
    text.insert(
        tk.END,
        np.array2string(
            dh_T[:3, 3],
            precision=5,
        ),
    )

    text.insert(
        tk.END,
        "\n\nMuJoCo end-effector position (m):\n",
    )
    text.insert(
        tk.END,
        np.array2string(
            mujoco_pos,
            precision=5,
        ),
    )

    text.insert(
        tk.END,
        "\n\nMuJoCo end-effector rotation:\n",
    )
    text.insert(
        tk.END,
        np.array2string(
            mujoco_rot,
            precision=5,
        ),
    )

    error = np.linalg.norm(
        dh_T[:3, 3] - mujoco_pos
    )

    text.insert(
        tk.END,
        f"\n\nPosition difference: {error:.6f} m\n",
    )

    text.insert(
        tk.END,
        "\n" + extra,
    )

    text.configure(state=tk.DISABLED)

    print("\nDashboard opened.")
    print("Close the dashboard window to return to the robot menu.")

    root.mainloop()


def run_heal():
    print("\nLoading HEAL model...")

    model, data, info = get_model(
        HEAL_XML,
        HEAL_JOINTS,
    )

    q = ask_angles(
        "HEAL",
        info,
    )

    set_joint_angles(
        model,
        data,
        HEAL_JOINTS,
        q,
    )

    mujoco_pos, mujoco_rot = body_pose(
        model,
        data,
        "end_effector",
    )

    heal_T = heal_fk(q)

    error = np.linalg.norm(
        heal_T[:3, 3] - mujoco_pos
    )

    print("\n========== HEAL RESULT ==========")
    print("Joint angles (degrees):", np.rad2deg(q))
    print("FK position (m):", heal_T[:3, 3])
    print("MuJoCo position (m):", mujoco_pos)
    print(f"Position difference: {error:.9f} m")
    print("=================================")

    show_viewer(
        model,
        data,
        "HEAL MuJoCo Viewer",
    )

    show_dashboard(
        "HEAL — FK vs MuJoCo",
        "HEAL XML-transform-chain FK result.",
        heal_T,
        mujoco_pos,
        mujoco_rot,
        "The HEAL FK uses the existing XML-derived "
        "transform chain from task2/heal_fk.py.",
    )


def run_franka():
    print("\nLoading Franka Panda model...")

    model, data, info = get_model(
        FRANKA_XML,
        FRANKA_JOINTS,
    )

    q = ask_angles(
        "Franka Panda",
        info,
    )

    set_joint_angles(
        model,
        data,
        FRANKA_JOINTS,
        q,
    )

    mujoco_pos, mujoco_rot = body_pose(
        model,
        data,
        "hand",
    )

    # Use the existing Franka MuJoCo FK implementation.
    # This uses the actual MuJoCo robot model rather than
    # the previously unverified candidate DH table.
    fk_position = franka_mujoco_fk(q)

    error = np.linalg.norm(
        fk_position - mujoco_pos
    )

    fk_T = np.eye(4)
    fk_T[:3, :3] = mujoco_rot
    fk_T[:3, 3] = fk_position

    print("\n========== FRANKA RESULT ==========")
    print("Joint angles (degrees):", np.rad2deg(q))
    print("FK position (m):", fk_position)
    print("MuJoCo hand position (m):", mujoco_pos)
    print("Position difference (m):", fk_position - mujoco_pos)
    print(f"Position error: {error:.9f} m")
    print("===================================")

    show_viewer(
        model,
        data,
        "Franka MuJoCo Viewer",
    )

    show_dashboard(
        "Franka — FK vs MuJoCo",
        "Franka FK calculated from the loaded MuJoCo model.",
        fk_T,
        mujoco_pos,
        mujoco_rot,
        "The FK position is calculated using "
        "task2/franka_fk.py and compared against "
        "the hand position from the same loaded MuJoCo model.\n"
        "The previous candidate DH table was removed from "
        "this result because it was not validated against the XML.",
    )


def main():
    while True:
        print("\n")
        print("=" * 64)
        print("             DUAL ROBOT MUJOCO LAUNCHER")
        print("=" * 64)
        print("1. HEAL")
        print("2. Franka")
        print("3. Exit")
        print("=" * 64)

        choice = input("Choose robot: ").strip()

        if choice == "1":
            run_heal()

        elif choice == "2":
            run_franka()

        elif choice == "3":
            print("\nExiting launcher.")
            break

        else:
            print("\nInvalid choice. Enter 1, 2, or 3.")


if __name__ == "__main__":
    main()
