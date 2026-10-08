# Lab 2 — Forward Kinematics and MuJoCo Simulation

## Overview

This repository contains the complete implementation for **Lab 2: Forward Kinematics and MuJoCo Simulation**.

The laboratory focuses on understanding and implementing forward kinematics for robotic manipulators and validating the results using the **MuJoCo physics simulation environment**.

The project is divided into three tasks:

- **Task 1:** Forward kinematics of a 2-DOF planar manipulator
- **Task 2:** Python forward-kinematics implementation for the 6-DOF HEAL robot and 7-DOF Franka Panda robot
- **Task 3:** MuJoCo simulation, interactive joint configuration, end-effector visualization, and FK validation

The project combines robotics mathematics, Python programming, numerical validation, robot modeling, and physics-based simulation.

---

# Task 1 — 2-DOF Manipulator Forward Kinematics

Task 1 focuses on the mathematical derivation of forward kinematics for a **2-DOF planar manipulator**.

The task establishes the relationship between:

- Joint variables
- Link lengths
- Transformation matrices
- Joint configuration
- End-effector position

For a planar two-link manipulator, the general forward-kinematics equations are:

\[
x = L_1\cos(\theta_1) + L_2\cos(\theta_1+\theta_2)
\]

\[
y = L_1\sin(\theta_1) + L_2\sin(\theta_1+\theta_2)
\]

where:

- \(L_1\) is the first link length
- \(L_2\) is the second link length
- \(\theta_1\) is the first joint angle
- \(\theta_2\) is the second joint angle
- \(x,y\) are the end-effector coordinates

The completed Task 1 solution is stored as:

```text
task1/Lab2_Task1.pdf
```

**[View Task 1 PDF](task1/Lab2_Task1.pdf)**

---

# Task 2 — Python Forward Kinematics

Task 2 extends the forward-kinematics concept to two complete robot models:

1. **HEAL — 6 DOF**
2. **Franka Panda — 7 DOF**

Python implementations were created to calculate robot end-effector positions for specified joint configurations.

---

## Task 2.1 — HEAL Forward Kinematics

The HEAL robot has **6 degrees of freedom**.

Implementation:

```text
task2/heal_fk.py
```

The implementation uses the robot transformation chain to calculate the end-effector position for a given joint configuration.

The tested configuration was:

```text
[30, 20, -15, 25, 10, -20] degrees
```

The calculated end-effector position was:

```text
[0.279863 0.239671 0.297129] m
```

The corresponding MuJoCo reference position was:

```text
[0.27986306 0.2396714  0.29712868] m
```

The resulting position error was approximately:

```text
1.3 × 10⁻⁷ m
```

This demonstrates close agreement between the implemented HEAL forward-kinematics calculation and the MuJoCo reference position for the tested configuration.

---

## Task 2.2 — Franka Panda Forward Kinematics

The Franka Panda robot has **7 degrees of freedom**.

Implementation:

```text
task2/franka_fk.py
```

The implementation calculates the Franka end-effector position for a specified seven-joint configuration.

The Franka implementation is also used by the Task 3 interactive MuJoCo launcher for validation.

The calculated FK position is compared with the `hand` body position obtained from the loaded MuJoCo Franka model.

For one tested configuration:

```text
FK position:
[0.034072 0.       0.749774]

MuJoCo hand position:
[0.034072 0.       0.749774]

Position difference:
[0. 0. 0.]

Position error:
0.000000 m
```

The implementation was also tested with other valid Franka joint configurations through the interactive launcher.

---

# Task 3 — MuJoCo Robot Simulation

Task 3 integrates the forward-kinematics implementations with the **MuJoCo simulation environment**.

The objective of Task 3 is to:

- Load the robot models into MuJoCo
- Accept user-defined joint configurations
- Apply joint angles to the robot
- Calculate forward kinematics
- Obtain end-effector positions from MuJoCo
- Compare FK and MuJoCo positions
- Calculate numerical position errors
- Visualize the robot configuration
- Display the FK and MuJoCo results

The supported robots are:

- **HEAL — 6 DOF**
- **Franka Panda — 7 DOF**

---

## Interactive Dual-Robot Launcher

The main Task 3 program is:

```text
task3/dual_robot_launcher.py
```

The launcher provides the following menu:

```text
1. HEAL
2. Franka
3. Exit
```

Run it from the project root with:

```bash
python3 -m task3.dual_robot_launcher
```

The user selects a robot and enters the required joint angles.

After the selected simulation windows are closed, the program returns to the robot-selection menu.

---

## HEAL MuJoCo Simulation

When **HEAL** is selected, the program:

1. Loads the HEAL MuJoCo model.
2. Requests six joint angles.
3. Accepts the angles in degrees.
4. Converts the angles to radians internally.
5. Applies the joint configuration to the MuJoCo model.
6. Calculates the forward-kinematics position.
7. Obtains the end-effector position from MuJoCo.
8. Calculates the position difference.
9. Calculates the numerical position error.
10. Opens the MuJoCo viewer.
11. Displays a result dashboard.

This provides both visual and numerical validation of the HEAL forward-kinematics implementation.

---

## Franka Panda MuJoCo Simulation

When **Franka** is selected, the program:

1. Loads the Franka Panda MuJoCo model.
2. Requests seven joint angles.
3. Accepts the angles in degrees.
4. Converts the angles to radians internally.
5. Applies the joint configuration to the MuJoCo model.
6. Calculates the FK position.
7. Obtains the `hand` body position from MuJoCo.
8. Compares the FK position with the MuJoCo hand position.
9. Calculates the numerical position error.
10. Opens the Franka MuJoCo viewer.
11. Displays a result dashboard.

The launcher can be tested using different valid Franka joint configurations.

---

# Task 3 Validation

The validation process compares the forward-kinematics position with the corresponding position obtained from the MuJoCo model.

The position error is calculated using the Euclidean distance:

\[
e = \left\|p_{FK} - p_{MuJoCo}\right\|
\]

where:

- \(p_{FK}\) is the calculated FK position
- \(p_{MuJoCo}\) is the position obtained from the MuJoCo model
- \(e\) is the numerical position error

### HEAL

For the tested HEAL configuration:

```text
Position error ≈ 1.3 × 10⁻⁷ m
```

### Franka

For the tested Franka configuration:

```text
Position error = 0.000000 m
```

The calculated FK position agreed with the MuJoCo `hand` body position.

---

# Task 3 Supporting Programs

The Task 3 directory contains:

```text
task3/
├── __init__.py
├── dual_robot_launcher.py
├── franka_deploy.py
├── heal_deploy.py
├── heal_mujoco.py
└── run_all_zero.py
```

### `dual_robot_launcher.py`

Interactive launcher for selecting and testing the HEAL or Franka robot.

### `heal_mujoco.py`

Loads the HEAL robot into MuJoCo and provides a zero-configuration simulation test.

### `heal_deploy.py`

Provides HEAL deployment and forward-kinematics validation.

### `franka_deploy.py`

Provides Franka forward-kinematics and MuJoCo validation.

### `run_all_zero.py`

Provides a zero-configuration test for the available robot models.

---

# Zero Configuration Test

A zero-configuration test was also implemented using:

```text
task3/run_all_zero.py
```

For the HEAL model, the zero configuration produced an end-effector position of approximately:

```text
x = 0.3774928841 m
y = -0.0003085475 m
z = 0.4984002352 m
```

This test was useful for verifying that the robot model could be loaded and that the end-effector position could be obtained before testing arbitrary joint configurations.

---

# Robot Descriptions

The MuJoCo XML robot descriptions are stored in:

```text
robot_descriptions/
```

These XML files contain the robot model definitions required by MuJoCo.

They define the robot structures used for simulation and forward-kinematics validation.

---

# Python Package Organization

The project is organized into Python packages:

```text
task2/
    __init__.py

task3/
    __init__.py
```

This allows the Task 3 launcher to import the FK implementations from Task 2.

For example:

```python
from task2.franka_fk import franka_mujoco_fk
from task2.heal_fk import heal_fk
```

The interactive launcher is therefore executed using:

```bash
python3 -m task3.dual_robot_launcher
```

---

# Demo Video

A demonstration video of the completed Lab 2 implementation is provided below.

The demonstration includes:

- Interactive robot selection
- HEAL joint-angle input
- HEAL MuJoCo visualization
- HEAL FK validation
- Franka joint-angle input
- Franka MuJoCo visualization
- Franka FK validation

**[Lab 2 Demonstration Video](PASTE_YOUR_VIDEO_LINK_HERE)**

> Replace `PASTE_YOUR_VIDEO_LINK_HERE` with the shareable video link.

---

# Project Structure

```text
lab2_franka_heal_mujoco/
│
├── README.md
├── environment.yml
├── requirements.txt
│
├── robot_descriptions/
│
├── task1/
│   └── Lab2_Task1.pdf
│
├── task2/
│   ├── __init__.py
│   ├── franka_fk.py
│   └── heal_fk.py
│
└── task3/
    ├── __init__.py
    ├── dual_robot_launcher.py
    ├── franka_deploy.py
    ├── heal_deploy.py
    ├── heal_mujoco.py
    └── run_all_zero.py
```

---

# Environment and Dependencies

The project was developed using:

- Python 3.10
- MuJoCo
- NumPy
- Tkinter
- Git
- GitHub

The required Python dependencies are listed in:

```text
requirements.txt
```

An environment configuration is also provided:

```text
environment.yml
```

Install the required Python dependencies using:

```bash
pip install -r requirements.txt
```

---

# Running the Project

Navigate to the project directory:

```bash
cd /home/anjil/Downloads/ITR_mujoco_fk_lab
```

Activate the virtual environment if required:

```bash
source .venv/bin/activate
```

Run the Task 3 interactive launcher:

```bash
python3 -m task3.dual_robot_launcher
```

The program displays:

```text
1. HEAL
2. Franka
3. Exit
```

Select the desired robot and enter the requested joint angles in degrees.

The corresponding MuJoCo viewer and result dashboard will open.

---

# Joint Angle Input

The interactive launcher accepts joint angles in **degrees**.

For the HEAL robot, six joint angles are required:

```text
Joint 1
Joint 2
Joint 3
Joint 4
Joint 5
Joint 6
```

For the Franka Panda robot, seven joint angles are required:

```text
Joint 1
Joint 2
Joint 3
Joint 4
Joint 5
Joint 6
Joint 7
```

The program converts the input angles to radians internally before applying them to the MuJoCo model.

Joint values should remain within the valid limits of the corresponding robot model.

---

# Testing and Validation

Testing was performed at multiple levels.

### HEAL FK Testing

The HEAL FK implementation was compared against the MuJoCo reference position.

Result:

```text
Position error ≈ 1.3 × 10⁻⁷ m
```

### Franka FK Testing

The Franka FK implementation was compared with the MuJoCo `hand` body position.

For the tested configuration:

```text
Position error = 0.000000 m
```

### Interactive Testing

The interactive launcher was tested with:

- HEAL joint configurations
- Franka joint configurations
- Zero configuration
- Different valid joint angles
- Repeated robot selection

The launcher successfully returns to the robot-selection menu after the simulation windows are closed.

---

# Technologies Used

### Python

Used for forward-kinematics calculations, robot configuration, MuJoCo integration, numerical calculations, and the interactive launcher.

### MuJoCo

Used to load, simulate, and visualize the HEAL and Franka Panda robot models.

### NumPy

Used for vectors, matrices, transformations, rotations, and numerical position-error calculations.

### Tkinter

Used for the result dashboard displaying FK and MuJoCo results.

### Git and GitHub

Used for version control, project organization, and submission.

---

# Learning Outcomes

This laboratory provided practical experience with:

- Forward kinematics
- Homogeneous transformation matrices
- Robot coordinate frames
- Multi-DOF robotic manipulators
- Joint-angle configuration
- Python robotics programming
- MuJoCo robot modeling
- MuJoCo simulation
- End-effector position extraction
- Numerical error calculation
- FK validation
- Interactive robot simulation
- Robot visualization
- Git and GitHub project management

---

# Results

The laboratory successfully demonstrated forward kinematics for different robotic systems.

### Task 1

The forward kinematics of a 2-DOF manipulator was derived and documented in the Task 1 PDF.

### Task 2 — HEAL

The Python FK implementation produced:

```text
Position error ≈ 1.3 × 10⁻⁷ m
```

when compared with the MuJoCo reference position for the tested configuration.

### Task 2 — Franka

The Franka implementation was successfully integrated with the MuJoCo model and tested using seven joint angles.

### Task 3

An interactive MuJoCo application was successfully developed for both robots.

The application allows:

- Robot selection
- Joint-angle input
- MuJoCo visualization
- FK calculation
- MuJoCo position extraction
- Numerical position comparison
- Position-error calculation
- Result visualization

---
## 🎥 Lab 2 Demonstration Videos

All Lab 2 demonstration videos are available in the following Google Drive folder:

[Watch Lab 2 Demonstration Videos](https://drive.google.com/drive/folders/1CIvtz5rspxNjfDOI7hQLfJB23owUt10u?usp=drive_link)

The folder contains:

- **Task 2 — Forward Kinematics**
  - HEAL FK calculation and MuJoCo validation
  - Franka FK demonstration

- **Task 3 — HEAL Deployment**
  - HEAL robot deployed in MuJoCo
  - Joint-angle input and end-effector visualization

- **Task 3 — Franka Deployment**
  - Franka Panda deployed in MuJoCo
  - Joint-angle input and end-effector visualization
# Conclusion

This laboratory demonstrated the complete workflow for implementing and validating forward kinematics for robotic manipulators.

**Task 1** established the mathematical foundation using a 2-DOF manipulator.

**Task 2** extended the forward-kinematics concept to the 6-DOF HEAL robot and 7-DOF Franka Panda robot using Python.

**Task 3** integrated the implementations with MuJoCo and provided an interactive environment where different joint configurations could be tested and visualized.

The final system allows the user to select a robot, enter its joint angles, visualize the resulting configuration in MuJoCo, and compare the forward-kinematics result with the corresponding MuJoCo end-effector position.

The project successfully combines **robotics mathematics, Python programming, numerical validation, robot modeling, MuJoCo simulation, and interactive visualization** into one complete laboratory implementation.

---

# Author

**Anjil Shah**

**Lab 2 — Forward Kinematics and MuJoCo Simulation**
