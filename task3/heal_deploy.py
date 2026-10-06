
import sys
import tkinter as tk
from tkinter import ttk
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from task2.heal_fk import heal_fk

XML_PATH = ROOT / "robot_descriptions/single_arm_heal_effort_actuation_rs_mj.xml"

JOINT_NAMES = [f"joint_{i}" for i in range(1, 7)]
TEST_DEG = np.array([30, 20, -15, 25, 10, -20], dtype=float)

# Existing candidate DH parameters: not yet verified against MuJoCo.
DH_TABLE = [
    [0.0, np.pi / 2, 0.3208],
    [0.3000, np.pi, 0.0],
    [0.0, np.pi / 2, 0.0],
    [0.03185, 0.50951, 0.3204],
    [0.0, np.pi / 2, 0.0654],
    [0.0, np.pi, -0.1227],
]


def main():
    np.set_printoptions(precision=6, suppress=True)

    # FK position from your existing DH-based function.
    T = heal_fk(np.deg2rad(TEST_DEG))
    fk_position = T[:3, 3]

    # Load MuJoCo and set the same joint angles.
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0

    for name, angle in zip(JOINT_NAMES, TEST_DEG):
        jid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        if jid < 0:
            raise RuntimeError(f"Joint not found: {name}")

        data.qpos[model.jnt_qposadr[jid]] = np.deg2rad(angle)

    mujoco.mj_forward(model, data)

    ee_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "end_effector"
    )
    if ee_id < 0:
        raise RuntimeError("Could not find end_effector body.")

    mujoco_position = data.xpos[ee_id].copy()
    difference = fk_position - mujoco_position
    error_m = np.linalg.norm(difference)

    # Terminal results.
    print("\n========== HEAL FK DASHBOARD ==========")
    print("Joint angles (degrees):", TEST_DEG)
    print("\nDH table (candidate; NOT verified)")
    print("Joint       a (m)      alpha (rad)     d (m)")
    for i, (a, alpha, d) in enumerate(DH_TABLE, start=1):
        print(f"J{i:<5} {a:>9.5f} {alpha:>14.5f} {d:>11.5f}")

    print("\nFK end-effector position (m):", fk_position)
    print("MuJoCo end-effector position (m):", mujoco_position)
    print("Difference FK - MuJoCo (m):", difference)
    print(f"Position error: {error_m:.9f} m ({error_m * 1000:.6f} mm)")
    print("NOTE: DH parameters are candidates, not verified.")
    print("=======================================\n")

    # Dashboard window.
    root = tk.Tk()
    root.title("HEAL Robot - FK Dashboard")
    root.geometry("720x650")

    frame = ttk.Frame(root, padding=18)
    frame.pack(fill="both", expand=True)

    ttk.Label(
        frame, text="HEAL ROBOT — FK DASHBOARD",
        font=("Arial", 18, "bold")
    ).pack(anchor="w", pady=(0, 8))

    ttk.Label(
        frame,
        text="Test angles: " + ", ".join(f"{x:g}°" for x in TEST_DEG)
    ).pack(anchor="w", pady=4)

    ttk.Label(
        frame,
        text="DH table: candidate values — not verified",
        foreground="darkorange"
    ).pack(anchor="w", pady=4)

    columns = ("joint", "a (m)", "alpha (rad)", "d (m)")
    table = ttk.Treeview(frame, columns=columns, show="headings", height=6)
    for col in columns:
        table.heading(col, text=col)
        table.column(col, width=130, anchor="center")
    for i, (a, alpha, d) in enumerate(DH_TABLE, start=1):
        table.insert("", "end", values=(
            f"Joint {i}", f"{a:.5f}", f"{alpha:.5f}", f"{d:.5f}"
        ))
    table.pack(fill="x", pady=8)

    def add_result(title, values):
        ttk.Label(frame, text=title, font=("Arial", 11, "bold")).pack(
            anchor="w", pady=(10, 2)
        )
        ttk.Label(
            frame,
            text=f"X: {values[0]: .6f} m    "
                 f"Y: {values[1]: .6f} m    "
                 f"Z: {values[2]: .6f} m",
            font=("Courier", 10)
        ).pack(anchor="w")

    add_result("Forward kinematics position", fk_position)
    add_result("MuJoCo end-effector position", mujoco_position)
    add_result("Difference (FK - MuJoCo)", difference)

    status = (
        f"Position error: {error_m:.9f} m  |  "
        f"{error_m * 1000:.6f} mm"
    )
    ttk.Label(
        frame, text=status, font=("Arial", 12, "bold")
    ).pack(anchor="w", pady=12)

    ttk.Label(
        frame,
        text="Note: the FK uses the candidate DH table in heal_fk.py.",
        foreground="gray"
    ).pack(anchor="w", pady=4)

    # Launch viewer while keeping the dashboard responsive.
    def open_viewer():
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                viewer.sync()
        root.after(0, root.destroy)

    import threading
    threading.Thread(target=open_viewer, daemon=True).start()
    root.mainloop()


if __name__ == "__main__":
    main()
