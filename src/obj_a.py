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
from isaacsim.robot_motion.motion_generation import (
    ArticulationKinematicsSolver,
    LulaKinematicsSolver,
    interface_config_loader,
)
from .frames_viz import FramesViz

class FrankaKinematicsStudy:
    def __init__(self):
        self._kinematics_solver = None
        self._articulation_kinematics_solver = None

        self._articulation = None
        self._target = None
        self._default_joint_positions = None  # Save the default pose used for IK calculations
        
        # Target cycling variables
        self._samples_failed = []
        self._samples_success = []
        self._all_targets = []
        self._current_target_idx = 0
        self._reached_threshold = 0.02  # 2cm position threshold
        self._quat_threshold = 0.05  # quaternion difference threshold
        self._wait_frames = 30  # Wait 30 frames before moving to next target
        self._wait_counter = 0

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

    def setup_A(self):
        # Load a URDF and Lula Robot Description File for this robot:
        mg_extension_path = get_extension_path_from_name("isaacsim.robot_motion.motion_generation")
        kinematics_config_dir = os.path.join(mg_extension_path, "motion_policy_configs")

        self._kinematics_solver = LulaKinematicsSolver(
            robot_description_path=kinematics_config_dir + "/franka/rmpflow/robot_descriptor.yaml",
            urdf_path=kinematics_config_dir + "/franka/lula_franka_gen.urdf",
        )

        # Kinematics for supported robots can be loaded with a simpler equivalent
        # print("Supported Robots with a Lula Kinematics Config:", interface_config_loader.get_supported_robots_with_lula_kinematics())
        # kinematics_config = interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")
        # self._kinematics_solver = LulaKinematicsSolver(**kinematics_config)

        #print("Valid frame names at which to compute kinematics:", self._kinematics_solver.get_all_frame_names())
    
        end_effector_name = "right_gripper"
        self._aks = ArticulationKinematicsSolver(
            self._articulation, self._kinematics_solver, end_effector_name
        )
        
        # Save the default joint positions - this is the pose used for all IK calculations
        self._default_joint_positions = self._articulation.get_joint_positions()
        print(f"Saved default joint positions: {self._default_joint_positions}")
        
        #use obj_a frames_viz
        
        self._frames_viz = FramesViz()
        self._samples_failed, self._samples_success = self._frames_viz.obj_a_viz(
            end_effector_name, self._kinematics_solver, N=200, L=0.04, clear=True, seed=42
        )
        
        # Combine failed (red) and success (green) targets, prioritizing failed ones
        self._all_targets = self._samples_failed + self._samples_success
        
        if len(self._all_targets) > 0:
            # Set initial target
            target_pos, target_quat = self._all_targets[0]
            self._target.set_world_pose(target_pos, target_quat)
            print(f"Starting target cycling through {len(self._all_targets)} poses ({len(self._samples_failed)} failed, {len(self._samples_success)} successful)")
            

    def update_A(self, step: float):
        if len(self._all_targets) == 0:
            return
            
        # Inverse Kinematics to move end-effector to target
        target_position, target_orient = self._target.get_world_pose()
        #set base pose for kinematics solver
        base_p, base_q = self._articulation.get_world_pose()
        self._kinematics_solver.set_robot_base_pose(base_p, base_q)
        
        action, success = self._aks.compute_inverse_kinematics(target_position, target_orient)
        if success:
            self._articulation.apply_action(action)
            
            # Check if end-effector reached target
            ee_pos, ee_rot = self._aks.compute_end_effector_pose()
            pos_diff = np.linalg.norm(ee_pos - target_position)
            quat_diff = np.linalg.norm(rot_matrices_to_quats(ee_rot.reshape(1,3,3))[0] - target_orient)
            
            if pos_diff < self._reached_threshold and quat_diff < self._quat_threshold:
                self._wait_counter += 1
                
                if self._wait_counter >= self._wait_frames:
                    # Move to next target
                    self._current_target_idx = (self._current_target_idx + 1) % len(self._all_targets)
                    next_pos, next_quat = self._all_targets[self._current_target_idx]
                    self._target.set_world_pose(next_pos, next_quat)
                    self._wait_counter = 0
                    
                    # Reset arm to the default pose that was used for IK calculations
                    self._articulation.set_joint_positions(self._default_joint_positions)
                    print(f"Moving to target {self._current_target_idx + 1}/{len(self._all_targets)}")
            else:
                self._wait_counter = 0
        else:
            self._current_target_idx = (self._current_target_idx + 1) % len(self._all_targets)
            next_pos, next_quat = self._all_targets[self._current_target_idx]
            self._target.set_world_pose(next_pos, next_quat)
            carb.log_warn("IK did not converge to a solution.  No action is being taken")
        

    def reset(self):
        # Reset to the default joint positions
        if self._default_joint_positions is not None:
            self._articulation.set_joint_positions(self._default_joint_positions)
        self._wait_counter = 0
