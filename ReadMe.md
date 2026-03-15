# CogNav — Vision-Based Collision Avoidance

CogNav is a decentralized collision avoidance framework that uses onboard RGB images from a robot's 360° camera to detect and reactively avoid collisions. It combines a fine-tuned monocular depth estimator (Depth Anything V2) and a reactive behaviour tree for real-time navigation in Webots simulation with optional ROS 2 / RViz2 visualisation.

---

## Environment Setup

### Prerequisites
- CUDA-capable GPU (CUDA 11.8 recommended)
- ROS 2 Humble *(only required for RViz2 visualisation)*

### Create the Conda Environment

```bash
conda env create -f environment.yml
conda activate cognav
```

## 1. Training

Trains DA2 on the 360° cylindrical Webots images to produce metric per-slice distances:

```bash
python train_depth_da2.py
```

**Offline evaluation** of a saved checkpoint:

```bash
python train_depth_da2.py --eval

# Quick check on a small subset
python train_depth_da2.py --eval --max-scenes 50
```

The report prints MAE,  Pearson r.
---

## 2. Data

> **Dataset will be provided via a separate download link.**

Once downloaded, place the data as follows:

```
CogNav/
└── dataset_360fov/
    └── dataset/
        ├── rgb_robot_000/    # 360° cylindrical RGB images per robot view
        ├── rgb_robot_001/
        │   ...
        ├── mask/             # Semantic segmentation masks (obstacle = 1)
        └── metadata/         # Per-scene JSON files with obstacle geometry
```

---

## 3. Building the ROS 2 Workspace

> Requires ROS 2 Humble installed at `/opt/ros/humble`.

```bash
# Source ROS 2
source /opt/ros/humble/setup.bash

# Navigate to the ROS 2 workspace
cd SingleRobotCA/ros2_ws

# Build with colcon
colcon build --symlink-install

# Source the workspace overlay
source install/setup.bash
```

Verify the package is found:

```bash
ros2 pkg list | grep webots_ca_bridge
```

---

## 4. Running Webots with RViz2

You need **two terminals**.

### Terminal 1 — Start Webots

```bash
# Open the simulation world in Webots
webots SingleRobotCA/worlds/singlerobotCA.wbt
```

Leave Webots running. The RobotController and SupervisorSM controllers start automatically when the simulation plays.

### Terminal 2 — Launch ROS 2 Bridge + RViz2

```bash
source /opt/ros/humble/setup.bash
source SingleRobotCA/ros2_ws/install/setup.bash

ros2 launch webots_ca_bridge ca_sim.launch.py
```

This starts:
- **`bridge_node`** — reads UDP state packets from the Webots RobotController and re-publishes as ROS 2 messages (`Odometry`, `Path`, `LaserScan`, `Markers`, `TF`)
- **`rviz2`** — opens with the pre-configured `config/ca_rviz.rviz` layout

Press **Play** in Webots. The robot pose, navigation path, depth detections, and obstacle markers will appear in RViz2.
---

## Navigation Metrics

The supervisor computes and prints the following metrics per scene and averaged over N scenes:

| Metric | Definition |
|---|---|
| **Path Efficiency %** | `min(D₀ / D_travel, 1) × 100` — straight-line distance vs actual path length |
| **Collision Count** | Distinct AABB penetration events (debounced) |

---

## Depth Estimator Metrics

| Metric | Definition |
|---|---|
| **MAE** | `mean |d_pred − d_gt|` in metres |
| **Pearson r** | Linear correlation between predicted and GT distance vectors |

> Full offline evaluation (including RMSE, AbsRel, SqRel, δ<1.25, per-slice MAE) is available via `python train_depth_da2.py --eval`.

---

## Visualisation Frames

At every inference step the supervisor saves 4 separate PNG files to `nav_viz/scene_<id>/`:

| File | Contents |
|---|---|
| `step_NNNNNN_1_cam_raw.png` | Raw 360° camera image before alignment |
| `step_NNNNNN_2_cam_aligned.png` | Aligned image (DA2 input) |
| `step_NNNNNN_3_overhead.png` | 2D map: obstacles, reference path (purple), avoidance path (blue), targets (green), DA2 detections (orange) |
| `step_NNNNNN_4_polar.png` | Polar depth view: EMA distance bars, raw DA2 (blue dashes), reference/detour arrows |
