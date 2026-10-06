
import numpy as np

# HEAL XML-derived fixed transforms and joint axes.
# Each joint rotates about its local z-axis.
def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)

def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)

def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)

def transform(pos, rot):
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = pos
    return T

def translate(x, y, z):
    return transform([x, y, z], np.eye(3))

def joint(q, axis_sign=1):
    return transform([0, 0, 0], rot_z(axis_sign * q))

def heal_fk(q):
    q = np.asarray(q, dtype=float)
    if q.shape != (6,):
        raise ValueError("Expected 6 joint angles in radians.")

    # Transform sequence follows the body positions and orientations
    # in the supplied HEAL MuJoCo XML.
    T = np.eye(4)
    T = T @ translate(0, 0, 0.171) @ joint(q[0])
    T = T @ translate(0, 0.0875, 0.1498) @ transform(
        [0, 0, 0], rot_x(np.pi / 2)
    ) @ joint(-q[1])
    T = T @ translate(0, 0.3, 0) @ transform(
        [0, 0, 0], rot_z(-1.57)
    ) @ joint(q[2])
    T = T @ translate(0, 0.1593, 0.0875) @ transform(
        [0, 0, 0], rot_x(-1.57)
    ) @ joint(q[3])
    T = T @ translate(0, 0.03185, 0.16105) @ transform(
        [0, 0, 0], rot_x(0.50951) @ rot_z(1.57)
    ) @ joint(q[4])
    T = T @ translate(0, -0.1227, 0.0654) @ transform(
        [0, 0, 0], rot_x(np.pi / 2)
    ) @ joint(-q[5])

    return T

def main():
    q_deg = np.array([30, 20, -15, 25, 10, -20], dtype=float)
    T = heal_fk(np.deg2rad(q_deg))

    np.set_printoptions(precision=6, suppress=True)
    print("HEAL FK (XML transform-chain candidate)")
    print("Joint angles (degrees):", q_deg)
    print("\nTransform:")
    print(T)
    print("\nEnd-effector position (m):", T[:3, 3])
    print("\nEnd-effector rotation:")
    print(T[:3, :3])
    print("\nMuJoCo reference position (m):",
          [0.27986306, 0.23967140, 0.29712868])
    print("Position error (m):",
          np.linalg.norm(T[:3, 3] -
                         [0.27986306, 0.23967140, 0.29712868]))

if __name__ == "__main__":
    main()
