# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import os

import carb
import numpy as np
from isaacsim.core.prims import SingleArticulation as Articulation
from isaacsim.core.prims import SingleXFormPrim as XFormPrim
from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats, rot_matrices_to_quats
from isaacsim.core.utils.stage import add_reference_to_stage
import lula
from isaacsim.robot_motion.motion_generation import (
    ArticulationKinematicsSolver,
    LulaKinematicsSolver,
    interface_config_loader,
    LulaTaskSpaceTrajectoryGenerator
)
from .frames_viz import FramesViz

class FrankaTrajectoryStudy:
    def __init__(self):
        self._kinematics_solver = None
        self._aks = None
        self._taskspace_trajectory_generator = None
        self._articulation = None
        self._target = None
        self._default_joint_positions = None  # Save the default pose used for IK calculations
        self._waypoints = None
        self._current_waypoint_idx = 0
    def load_example_assets(self):
        # Add the Franka and target to the stage

        robot_prim_path = "/panda"
        path_to_robot_usd = get_assets_root_path() + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"

        add_reference_to_stage(path_to_robot_usd, robot_prim_path)
        self._articulation = Articulation(robot_prim_path)

        add_reference_to_stage(get_assets_root_path() + "/Isaac/Props/UIElements/frame_prim.usd", "/World/target")
        self._target = XFormPrim("/World/target", scale=[0.04, 0.04, 0.04])
        self._target.set_default_state(np.array([0.3, 0, 0.5]), euler_angles_to_quats([0, np.pi, 0]))

        # Return assets that were added to the stage so that they can be registered with the core.World
        return self._articulation, self._target

    def setup_B(self):
        # Load a URDF and Lula Robot Description File for this robot:
        mg_extension_path = get_extension_path_from_name("isaacsim.robot_motion.motion_generation")
        kinematics_config_dir = os.path.join(mg_extension_path, "motion_policy_configs")

        robot_desc_yaml = kinematics_config_dir + "/franka/rmpflow/robot_descriptor.yaml"
        robot_urdf      = kinematics_config_dir + "/franka/lula_franka_gen.urdf"

        self._lula_robot_desc = lula.load_robot(robot_desc_yaml, robot_urdf)  # (yaml, urdf) :contentReference[oaicite:0]{index=0}
        self._lula_kin = self._lula_robot_desc.kinematics()                                    # :contentReference[oaicite:1]{index=1}
        self._lula_n   = self._lula_robot_desc.num_c_space_coords()
        print(f"Lula Robot Description loaded with {self._lula_n} DOF.")
        self._kinematics_solver = LulaKinematicsSolver(
            robot_description_path=robot_desc_yaml,
            urdf_path=robot_urdf,
        )

        # Kinematics for supported robots can be loaded with a simpler equivalent
        # print("Supported Robots with a Lula Kinematics Config:", interface_config_loader.get_supported_robots_with_lula_kinematics())
        # kinematics_config = interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")
        # self._kinematics_solver = LulaKinematicsSolver(**kinematics_config)

        #print("Valid frame names at which to compute kinematics:", self._kinematics_solver.get_all_frame_names())
        self._taskspace_trajectory_generator = LulaTaskSpaceTrajectoryGenerator(
            robot_description_path=robot_desc_yaml,
            urdf_path=robot_urdf,

        )
        self._ee_frame = "right_gripper"
        self._aks = ArticulationKinematicsSolver(
            self._articulation, self._kinematics_solver, self._ee_frame
        )
        self._waypoints = self.make_circle_waypoints(center=[0.3,0,0.2], r=0.3, n=400)
        self.frames_viz = FramesViz()
        self.frames_viz.draw_waypoints(self._waypoints)

    def update_B(self, step: float):
        target_p, target_q = self._target.get_world_pose()
        base_p, base_q = self._articulation.get_world_pose()
        self._kinematics_solver.set_robot_base_pose(base_p, base_q)
        
        target_quat = euler_angles_to_quats([0, np.pi, 0]) 
        
        # Get current waypoint target
        target_pos = self._waypoints[self._current_waypoint_idx]
        
        # Check distance from end-effector to target waypoint
        ee_p, _ = self._aks.compute_end_effector_pose()
        dist = np.linalg.norm(np.array(target_pos) - np.array(ee_p))

        if dist < 0.05:
            self._current_waypoint_idx = (self._current_waypoint_idx + 1) % len(self._waypoints)
            target_pos = self._waypoints[self._current_waypoint_idx]
        
        # Update target visual to show where robot is going
        self._target.set_world_pose(target_pos, target_quat)

        action, success = self._aks.compute_inverse_kinematics(target_pos, target_quat)
        if success:
            # Get all current joint positions (9 DOF including gripper)
            current_joints = self._articulation.get_joint_positions()
            # IK returns only 7 DOF (arm joints)
            target_joints = action.joint_positions
            
            # Interpolate only the arm joints (first 7) for smooth motion
            alpha = 0.1  # Smoothing factor (lower = smoother but slower)
            smooth_arm_joints = current_joints[:7] + alpha * (target_joints - current_joints[:7])
            
            # Combine smoothed arm joints with current gripper positions
            smooth_joints = np.concatenate([smooth_arm_joints, current_joints[7:]])
            self._articulation.set_joint_positions(smooth_joints)
        else:
            carb.log_warn(f"IK failed to find a solution for target position {target_pos}")
        
        q = np.asarray(self._articulation.get_joint_positions()[:7], dtype=np.float64).reshape((-1, 1))
        J = self._lula_kin.jacobian(q, self._ee_frame)
        cond = np.linalg.cond(J)
        #if cond > 100:
            
        print(f"High condition number for Jacobian at waypoint {self._current_waypoint_idx}: {cond}") 
    def reset(self):

        pass
    def make_circle_waypoints(self, center=[0.4,0,0.4], r=0.4, n=400):
        pts = []
        for k in range(n):
            th = 2*np.pi*k/(n-1)
            pts.append([center[0],
                        center[1] + r*np.cos(th),
                        center[2] + r*np.sin(th)])
        return np.array(pts)
    
