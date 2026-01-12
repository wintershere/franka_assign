# Franka Assignment

A demonstration of Franka Panda robot kinematics using NVIDIA Isaac Sim. This extension samples random poses within a defined task space, visualizes reachable and unreachable poses, and automatically cycles through targets using inverse kinematics.

## Features

- **Task Space Sampling**: Generates random poses within a defined cuboid workspace
- **IK Validation**: Tests each sampled pose for reachability using Lula kinematics solver
- **Visual Feedback**: 
  - Green frames indicate successful IK solutions (reachable poses)
  - Red frames indicate failed IK solutions (unreachable poses)
  - Yellow wireframe shows the task space boundary
- **Automatic Target Cycling**: Robot automatically moves through all sampled targets
  - Prioritizes unreachable poses first
  - Resets to default pose between targets
  - Loops infinitely through all samples

## Requirements

- NVIDIA Isaac Sim 2023.1 or later
- Required Isaac Sim extensions:
  - `isaacsim.robot_motion.motion_generation`
  - `isaacsim.util.debug_draw`
  - `isaacsim.core.utils`

## Installation

1. Clone this repository into your Isaac Sim `extsUser` directory:
   ```bash
   cd <isaac-sim-path>/extsUser
   git clone <repository-url> franka_assignment
   ```

2. Launch Isaac Sim

3. Enable the extension:
   - Window → Extensions
   - Search for "Franka Assignment"
   - Enable the extension

## Usage

1. Click "Load Scenario" to load the Franka robot and generate task space samples
2. Click "Run Scenario A" to start automatic target cycling
3. The robot will:
   - Move to each target pose
   - Wait briefly at each target
   - Reset to default configuration
   - Move to the next target

## Configuration

You can adjust parameters in `Lula_Kinematics_python/obj_a.py`:

- `seed=42`: Random seed for reproducible sampling
- `N=200`: Number of poses to sample
- `_reached_threshold = 0.02`: Position tolerance (meters)
- `_quat_threshold = 0.05`: Orientation tolerance
- `_wait_frames = 30`: Frames to wait at each target

Task space bounds are defined in `frames_viz.py`:
- X: -0.5 to 0.5 meters
- Y: -0.5 to 0.5 meters  
- Z: -0.2 to 0.6 meters

## Files

- `Lula_Kinematics_python/obj_a.py`: Main logic for target cycling and IK control
- `Lula_Kinematics_python/frames_viz.py`: Visualization and task space sampling
- `Lula_Kinematics_python/ui_builder.py`: User interface
- `config/extension.toml`: Extension configuration

## License

See LICENSE.txt
