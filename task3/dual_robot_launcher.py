
import threading
import tkinter as tk
from tkinter.scrolledtext import ScrolledText

import mujoco
import mujoco.viewer
import numpy as np

from scripts.franka_fk import franka_mujoco_fk
from scripts.heal_fk import heal_fk


FRANKA_XML = "robot_descriptions/franka/scene.xml"
HEAL_XML = "robot_descriptions/single_arm_heal_effort_actuation_rs_mj.xml"

FRANKA_JOINTS = [f"joint{i}" for i in range(1, 8)]
HEAL_JOINTS = [f"joint_{i}" for i in range(1, 7)]


# Candidate standard-DH parameters: [a, alpha, d, theta_offset].
# Verify these against the robot's actual kinematic convention.
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
        model, mujoco.mjtObj.mjOBJ_JOINT, name
    )
    if jid < 0:
        raise ValueError(f"Joint '{name}' was not found in the XML.")
    return jid


def get_model(xml_path, joint_names):
    model = mujoco.MjModel.from_xml_path(xml_path)
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
    print("DISCLAIMER")
    print("- Enter angles in degrees.")
    print("- The axis shown is the joint axis defined in the MuJoCo XML.")
    print("- Positive rotation follows the right-hand rule around that axis.")
    print("- Limits are read from the loaded model when limits are defined.")
    print("- Unverified DH results are estimates, not proof of physical accuracy.")
    print("- Simulation only: do not send these values directly to hardware.\n")

    angles = []
    for i, item in enumerate(info, 1):
        print(f"Joint {i}: {item['name']}")
        print(f"  Associated joint: {item['name']}")
        print("  XML axis:", np.array2string(item["axis"], precision=3))

        if item["limited"]:
            print(f"  Model range: {item['low']:.2f}° to "
                  f"{item['high']:.2f}°")
        else:
            print("  Model range: unlimited in XML; input restricted to ±180°")

        low = item["low"] if item["limited"] else -180.0
        high = item["high"] if item["limited"] else 180.0

        while True:
            try:
                value = float(input("  Angle (degrees): "))
                if not np.isfinite(value) or not low <= value <= high:
                    print(f"  Enter a finite angle between {low:.2f}° "
                          f"and {high:.2f}°.")
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
        model, mujoco.mjtObj.mjOBJ_BODY, body_name
    )
    if bid < 0:
        raise ValueError(f"End-effector body '{body_name}' not found.")

    return data.xpos[bid].copy(), data.xmat[bid].reshape(3, 3).copy()


def launch_viewer(model, data, title):
    def run():
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.lookat[:] = data.xpos[1]
            while viewer.is_running():
                viewer.sync()
                import time
                time.sleep(1 / 60)

    threading.Thread(target=run, daemon=True, name=title).start()


def dashboard(root, title, dh_description, dh_T,
              mujoco_pos, mujoco_rot, extra=""):
    window = tk.Toplevel(root)
    window.title(title)
    window.geometry("760x700")

    text = ScrolledText(window, wrap=tk.WORD, font=("monospace", 10))
    text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    text.insert(tk.END, title + "\n")
    text.insert(tk.END, "=" * 64 + "\n\n")
    text.insert(tk.END, dh_description + "\n\n")
    text.insert(tk.END, "Candidate DH / FK transformation:\n")
    text.insert(tk.END, np.array2string(dh_T, precision=5, suppress_small=True))
    text.insert(tk.END, "\n\nCandidate FK position (m):\n")
    text.insert(tk.END, np.array2string(dh_T[:3, 3], precision=5))
    text.insert(tk.END, "\n\nMuJoCo end-effector position (m):\n")
    text.insert(tk.END, np.array2string(mujoco_pos, precision=5))
    text.insert(tk.END, "\n\nMuJoCo end-effector rotation:\n")
    text.insert(tk.END, np.array2string(mujoco_rot, precision=5))

    error = np.linalg.norm(dh_T[:3, 3] - mujoco_pos)
    text.insert(tk.END, f"\n\nRaw position difference: {error:.6f} m\n")
    text.insert(tk.END, "\nIMPORTANT:\n")
    text.insert(tk.END,
        "This difference is meaningful only when both methods use the "
        "same base frame, end-effector frame, joint zero offsets and "
        "kinematic convention. A large difference does not by itself "
        "prove either method is wrong.\n")
    text.insert(tk.END, extra)
    text.configure(state=tk.DISABLED)
    return window


def main():
    print("DUAL ROBOT FK LAB")
    print("Enter joint angles for each robot when prompted.")

    fm, fd, fi = get_model(FRANKA_XML, FRANKA_JOINTS)
    hm, hd, hi = get_model(HEAL_XML, HEAL_JOINTS)

    fq = ask_angles("Franka Panda", fi)
    hq = ask_angles("HEAL", hi)

    set_joint_angles(fm, fd, FRANKA_JOINTS, fq)
    set_joint_angles(hm, hd, HEAL_JOINTS, hq)

    # MuJoCo reference poses
    f_pos, f_rot = body_pose(fm, fd, "hand")
    h_pos, h_rot = body_pose(hm, hd, "end_effector")

    # Existing FK implementations
    franka_mujoco_position = franka_mujoco_fk(fq)
    franka_T = franka_dh_fk(fq)
    heal_T = heal_fk(hq)

    print("\nFranka MuJoCo FK function position:", franka_mujoco_position)
    print("Franka scene hand position:", f_pos)
    print("HEAL scene end-effector position:", h_pos)

    launch_viewer(fm, fd, "Franka MuJoCo Viewer")
    launch_viewer(hm, hd, "HEAL MuJoCo Viewer")

    root = tk.Tk()
    root.withdraw()

    dashboard(
        root,
        "Franka — DH vs MuJoCo",
        "Franka candidate DH table [a, alpha, d, theta offset]:\n"
        + np.array2string(np.asarray(FRANKA_DH), precision=5),
        franka_T, f_pos, f_rot,
        "\nYour existing franka_mujoco_fk() returns position only. "
        "It is printed in the terminal for comparison with the scene."
    )

    dashboard(
        root,
        "HEAL — Candidate FK vs MuJoCo",
        "HEAL uses your existing heal_fk() XML-transform-chain candidate. "
        "It is not independently verified DH output.",
        heal_T, h_pos, h_rot,
        "\nCheck the transform chain against the XML before treating it "
        "as a validated DH model."
    )

    print("\nBoth dashboards and viewer threads have been started.")
    print("Close the dashboard windows to end the launcher.")
    root.mainloop()


if __name__ == "__main__":
    main()
