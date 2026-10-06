
import threading
import tkinter as tk
from tkinter import ttk
from pathlib import Path
import sys

import mujoco
import mujoco.viewer
import numpy as np

# Allow importing franka_fk.py from the scripts folder
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from franka_fk import franka_mujoco_fk


XML_PATH = PROJECT_DIR / "robot_descriptions/franka/scene.xml"

# Seven Franka Panda arm joints, in degrees
TEST_DEG = [0, -30, 0, -100, 0, 70, 0]
JOINT_NAMES = [f"joint{i}" for i in range(1, 8)]
EE_BODY_NAME = "hand"


def main():
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    # Find all seven arm joints in MuJoCo
    joint_ids = []
    for name in JOINT_NAMES:
        jid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        if jid < 0:
            raise ValueError(f"Joint not found in XML: {name}")
        joint_ids.append(jid)

    # Find the Franka hand body
    ee_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY_NAME
    )
    if ee_id < 0:
        raise ValueError(f"Body not found in XML: {EE_BODY_NAME}")

    # Reset to the model's initial state, then apply test angles
    data.qpos[:] = model.qpos0
    q_rad = np.deg2rad(TEST_DEG)

    for jid, angle in zip(joint_ids, q_rad):
        data.qpos[model.jnt_qposadr[jid]] = angle

    mujoco.mj_forward(model, data)

    # Calculate DH forward kinematics
    T = franka_mujoco_fk(q_rad)
    fk_position = T

    # MuJoCo reference position
    mujoco_position = data.xpos[ee_id].copy()

    difference = fk_position - mujoco_position
    error_m = np.linalg.norm(difference)

    # ---------- Terminal output ----------
    np.set_printoptions(precision=6, suppress=True)

    print("\n========== FRANKA FK DASHBOARD ==========")
    print("Joint angles (degrees):", TEST_DEG)
    print("\nDH FK position (m):", fk_position)
    print("MuJoCo hand position (m):", mujoco_position)
    print("Position difference (m):", difference)
    print(f"Position error: {error_m:.6f} m")
    print("Note: DH parameters must be validated against the XML.")
    print("=========================================\n")

    # ---------- Tkinter dashboard ----------
    root = tk.Tk()
    root.title("Franka Panda - FK Dashboard")
    root.geometry("600x480")

    ttk.Label(
        root,
        text="FRANKA PANDA FORWARD KINEMATICS",
        font=("Arial", 15, "bold"),
    ).pack(pady=12)

    ttk.Label(
        root,
        text="Joint Angles (degrees)",
        font=("Arial", 11, "bold"),
    ).pack()

    angle_text = "    ".join(
        f"J{i}: {angle}°"
        for i, angle in enumerate(TEST_DEG, start=1)
    )
    ttk.Label(root, text=angle_text, wraplength=560).pack(pady=8)

    table = ttk.Treeview(
        root,
        columns=("axis", "dh", "mujoco", "diff"),
        show="headings",
        height=3,
    )

    for col, title in [
        ("axis", "Axis"),
        ("dh", "DH FK (m)"),
        ("mujoco", "MuJoCo (m)"),
        ("diff", "Difference (m)"),
    ]:
        table.heading(col, text=title)
        table.column(col, width=130, anchor="center")

    for i, axis in enumerate(["X", "Y", "Z"]):
        table.insert(
            "",
            "end",
            values=(
                axis,
                f"{fk_position[i]:.6f}",
                f"{mujoco_position[i]:.6f}",
                f"{difference[i]:.6f}",
            ),
        )

    table.pack(padx=12, pady=14, fill="x")

    ttk.Label(
        root,
        text=f"Position error: {error_m:.6f} m",
        font=("Arial", 12, "bold"),
    ).pack(pady=8)

    ttk.Label(
        root,
        text=(
            "A small error indicates agreement between the two models.\n"
            "A large error means the DH frames/parameters need checking."
        ),
        justify="center",
    ).pack(pady=6)

    ttk.Button(root, text="Close Dashboard", command=root.destroy).pack(
        pady=12
    )

    # Open the MuJoCo viewer separately
    def launch_viewer():
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                viewer.sync()

    threading.Thread(target=launch_viewer, daemon=True).start()

    root.mainloop()


if __name__ == "__main__":
    main()
