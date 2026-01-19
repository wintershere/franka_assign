# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
# ...

import os, sys, subprocess
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from isaacsim.core.prims import SingleArticulation as Articulation
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.prims import SingleXFormPrim as XFormPrim
from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats, rot_matrices_to_quats
from isaacsim.core.utils.stage import add_reference_to_stage
import lula
from .frames_viz import FramesViz


class FrankaTrajectoryStudy:
    def __init__(self):
        self._kinematics_solver = None
        self._aks = None
        self._taskspace_trajectory_generator = None
        self._articulation = None
        self._target = None

        # For viz only
        self._waypoints_pos = None
        self._waypoints_quat = None

        # Logging
        self._log_len = 1200  # ~20s at 60 Hz
        self._log_t = deque(maxlen=self._log_len)
        self._log_q = [deque(maxlen=self._log_len) for _ in range(7)]
        self._t = 0.0
        self._prev_q_wrapped = None
        self._prev_q_unwrapped = None

        # Trajectory params (set in setup_B)
        self._circle_center = None
        self._circle_r = None
        self._circle_T = None
        self._theta0 = None  # initial angle for circle trajectory
        self._q_nom = None  # nominal posture for nullspace

        # Manipulability gradient cache (optional)
        self._manip_grad_cache = np.zeros(7, dtype=np.float64)
        self._manip_cache_counter = 0

        self._run_id = 0

    def load_example_assets(self):
        robot_prim_path = "/panda"
        path_to_robot_usd = get_assets_root_path() + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
        add_reference_to_stage(path_to_robot_usd, robot_prim_path)
        self._articulation = Articulation(robot_prim_path)

        add_reference_to_stage(get_assets_root_path() + "/Isaac/Props/UIElements/frame_prim.usd", "/World/target")
        self._target = XFormPrim("/World/target", scale=[0.04, 0.04, 0.04])
        self._target.set_default_state(np.array([0.3, 0, 0.5]), euler_angles_to_quats([0, np.pi, 0]))

        return self._articulation, self._target

    def setup_B(self):
        mg_extension_path = get_extension_path_from_name("isaacsim.robot_motion.motion_generation")
        kinematics_config_dir = os.path.join(mg_extension_path, "motion_policy_configs")

        robot_desc_yaml = kinematics_config_dir + "/franka/rmpflow/robot_descriptor.yaml"
        robot_urdf = kinematics_config_dir + "/franka/lula_franka_gen.urdf"

        self._lula_robot_desc = lula.load_robot(robot_desc_yaml, robot_urdf)
        self._lula_kin = self._lula_robot_desc.kinematics()
        self._lula_n = self._lula_robot_desc.num_c_space_coords()
        self._ee_frame = "right_gripper"
        
        # ---- Continuous trajectory definition (circle in task space) ----
        self._circle_center = np.array([0.6, 0.0, 0.2], dtype=np.float64)
        self._circle_r = 0.2
        self._circle_T = 12.0  # seconds per loop (tune)

        
        ee = self._lula_kin.pose(
            np.asarray(self._articulation.get_joint_positions()[:7], dtype=np.float64).reshape((-1, 1)),
            self._ee_frame
        )
        ee_p = np.asarray(ee.translation, dtype=np.float64)
        delta = ee_p - self._circle_center
        self._theta0 = np.arctan2(delta[2], delta[1])  # atan2(z, y) for circle in y-z plane

        self._build_ts_path()
        # Waypoints only for visualization (NOT used as discrete setpoints)
        waypoints_data = self.make_circle_waypoints(center=self._circle_center, r=self._circle_r, n=100)
        self._waypoints_pos, self._waypoints_quat = waypoints_data
        self.frames_viz = FramesViz()
        self.frames_viz.draw_waypoints(self._waypoints_pos)

        
        
        self._t = 0.0
        self._path_ready = False
        # Sanity check Jacobian shape
        q = np.asarray(self._articulation.get_joint_positions()[:7], dtype=np.float64)
        J = np.asarray(self._lula_kin.jacobian(q.reshape((-1, 1)), self._ee_frame), dtype=np.float64)

    def update_B(self, step: float):
        # base pose (world) and prinnt once
        base_p, base_q = self._articulation.get_world_pose()
        if self._t == 0.0:
            print(f"Base pos: {base_p}, quat (wxyz): {base_q.flatten()[:4]}")

        dt = float(step)
        self._t += dt
        if not self._path_ready:
            q_now = np.asarray(self._articulation.get_joint_positions()[:7], dtype=np.float64)
            if np.allclose(q_now, 0.0):
                return  # wait next physics tick

            self._build_ts_path_from_current()
            self._path_ready = True
            return

        self._s = min(self._s + (self._dom.span()/12.0)*dt, self._dom.upper)
        self._s1 = min(self._s +(self._dom.span()/12.0)*dt, self._dom.upper)
        p_des = np.asarray(self._ts_path.eval(self._s).translation, dtype=np.float64).reshape(3, 1)
        p_des1 = np.asarray(self._ts_path.eval(self._s1).translation, dtype=np.float64).reshape(3, 1)
        p_des_dot = (p_des1 - p_des)/dt

        q_des = self._q_des_const  # keep constant orientation

        self._target.set_world_pose(p_des.flatten(), q_des)

        # current EE pose (world)
        joint_positions = self._articulation.get_joint_positions()
        if joint_positions is None:
            return
        q = np.asarray(joint_positions[:7], dtype=np.float64)

        ee_p = np.asarray(self._lula_kin.position(q.reshape(-1,1), self._ee_frame), dtype=np.float64).reshape(3,1)
        ee_o = rot_matrices_to_quats(self._lula_kin.orientation(q.reshape(-1,1), self._ee_frame).matrix()).reshape(4,1)

        # joints
        
        # Jacobian (Lula): 6x7, expressed in BASE frame coords  :contentReference[oaicite:3]{index=3}
        J = np.asarray(self._lula_kin.jacobian(q.reshape((-1, 1)), self._ee_frame), dtype=np.float64)
        #compute J_pseudo_inv
        J_pinv = np.linalg.pinv(J)
        #dls
        lambda_dls = 0.01
        J_pinv_dls = J.T @ np.linalg.inv(J @ J.T + (lambda_dls**2) * np.eye(6))
        #null space projection matrix
        N = np.eye(7) - J_pinv @ J
        Kp = 1.0
        Ko = 1.0
        Eo = self._orientation_error_from_quat(q_des.reshape(4,1), ee_o.reshape(4,1))
        xdot = np.vstack((
            p_des_dot + Kp*(p_des - ee_p),
            Ko*Eo
        ))
        q_dot_task = J_pinv @ xdot

        #Redundance utilization
        #k_null = 0.0
        q_dot_null = self.qdot0_manipulability_sigmin_fd(lambda q_: self._lula_kin.jacobian(q_, self._ee_frame), q.reshape((7,1)), k=0.1)
        q_dot = q_dot_task + N @ q_dot_null
        # Update joint positions
        q_new = q.reshape((7,1)) + q_dot * dt
        #stack 0 to get 9dof franka
        q_dot_new = np.vstack((q_dot, np.zeros((2,1), dtype=np.float64)))
        q_new = np.vstack((q_new, np.zeros((2,1), dtype=np.float64)))
        #articulation command
        self._articulation.apply_action(ArticulationAction(joint_positions=q_new.flatten().tolist(),joint_velocities= q_dot_new.flatten().tolist()))

        ##logging
        if len(self._log_t) == 0:
            self._log_t.clear()
            for i in range(7):
                self._log_q[i].clear()

        self._log_t.append(self._t)
        q7 = q.reshape(7,)  # or q_new[:7].reshape(7,)
        for i in range(7):
            self._log_q[i].append(float(q7[i]))


    def reset(self):
        # dump plots for the run that just finished
        self.plot_joint_trajectories(out_dir="extsUser/kinematics_analysis/assets/plots", tag=f"run_{self._run_id:03d}")
        self._run_id += 1
        self.clear_joint_log()

        # reset state
        self._t = 0.0
        self._path_ready = False

##-------------------##
##------Helpers ------##
    @staticmethod
    def _normalize(v, eps=1e-12):
        v = np.asarray(v, dtype=np.float64)
        n = float(np.linalg.norm(v))
        if n < eps:
            return np.zeros_like(v)
        return v / n

    def make_circle_waypoints(self, center=(0.4, 0.0, 0.4), r=0.25, n=400):
        center = np.asarray(center, dtype=np.float64)

        pos_wps = np.zeros((n, 3), dtype=np.float64)
        quat_wps = np.zeros((n, 4), dtype=np.float64)

        for k in range(n):
            th = 2.0 * np.pi * k / (n - 1)

            p = np.array([center[0], center[1] + r * np.cos(th), center[2] + r * np.sin(th)], dtype=np.float64)
            pos_wps[k] = p

            t = np.array([0.0, -np.sin(th), np.cos(th)], dtype=np.float64)
            x_axis = self._normalize(t)
            z_axis = self._normalize(center - p)
            y_axis = self._normalize(np.cross(z_axis, x_axis))
            x_axis = self._normalize(np.cross(y_axis, z_axis))
            R = np.column_stack((x_axis, y_axis, z_axis))

            q = rot_matrices_to_quats(R[None, :, :])[0]  
            if k > 0 and np.dot(q, quat_wps[k - 1]) < 0.0:
                q = -q
            quat_wps[k] = q

        return (pos_wps, quat_wps)


    def _c3(self,v):
        return np.asarray(v, dtype=np.float64).reshape(3, 1)

    def _build_ts_path(self):
        q0 = np.asarray(self._articulation.get_joint_positions()[:7], dtype=np.float64).reshape(-1, 1)
        ee0 = self._lula_kin.pose(q0, self._ee_frame)  # lula.Pose3

        # Circle definition (your values)
        c = self._circle_center
        r = float(self._circle_r)

        # Start point on circle: angle = 0 -> y = cy + r, z = cz
        p_start = np.array([c[0], c[1] + r, c[2]], dtype=np.float64)

        # Build a continuous task-space path spec starting at current EE pose
        ts = lula.create_task_space_path_spec(ee0)

        # Approach segment, pure translation 
        # blend_radius helps smooth the junction into the next segment
        ts.add_translation(self._c3(p_start), blend_radius=0.02)

        # Circle: 4 quarter arcs using three-point arcs (constant orientation)
        for k in range(4):
            a0 = k * (np.pi / 2.0)
            amid = a0 + (np.pi / 4.0)
            a1 = a0 + (np.pi / 2.0)

            p_mid = np.array([c[0], c[1] + r*np.cos(amid), c[2] + r*np.sin(amid)], dtype=np.float64)
            p_end = np.array([c[0], c[1] + r*np.cos(a1),   c[2] + r*np.sin(a1)],   dtype=np.float64)

            ts.add_three_point_arc(self._c3(p_end), self._c3(p_mid), constant_orientation=True)

        # Materialize to a continuous path you can sample
        self._ts_path = ts.generate_path()
        self._dom = self._ts_path.domain()
        self._s = float(self._dom.lower)

    def _build_ts_path_from_current(self):
        # build path starting from the CURRENT EE pose (after sim init)
        self._build_ts_path()

        # keep constant orientation = current EE orientation at start (no jumps)
        q0 = np.asarray(self._articulation.get_joint_positions()[:7], dtype=np.float64).reshape(-1, 1)
        self._q_des_const = rot_matrices_to_quats(self._lula_kin.orientation(q0, self._ee_frame).matrix()).reshape(4,)
        self._q_nom = np.asarray(self._articulation.get_joint_positions()[:7], dtype=np.float64).reshape(7,1)

    def _orientation_error_from_quat(self, q_target, q_current):
        """Compute orientation error vector from two quaternions (w,x,y,z)

        Args:
            q_target (np.array): (4 x 1) quaternion describing the desired orientation"
            q_current (np.array): (4 x 1) quaternion describing the current orientation
        Returns:
            np.array: (3 x 1) orientation error vector
        """
        term1 = q_target[0] * q_current[1:] - q_current[0] * q_target[1:]
        term2 = np.cross(q_current[1:].flatten(), q_target[1:].flatten()).reshape(3, 1)
        return term2 - term1
    @staticmethod
    def qdot0_nominal(J, q, q_nom, k=1.0):
        """
        Move towadr nominal posture 
        """
        _, _, Vt = np.linalg.svd(J, full_matrices=True)
        n = Vt[-1, :].reshape(-1, 1)             
        n /= np.linalg.norm(n) + 1e-12

        # move toward nominal posture along null direction
        alpha = -k * float(n.T @ (q - q_nom))
        return alpha * n
    @staticmethod
    def qdot0_manipulability_sigmin_fd(J_func, q, eps=1e-4, k=0.2):
        """
        Maximize sigma_min(J): qdot0 = k * grad(sigma_min)
        """
        q = np.asarray(q, dtype=np.float64).reshape(-1, 1)
        n = q.shape[0]

        def sigma_min(q_):
            J = np.asarray(J_func(q_), dtype=np.float64)
            A = J @ J.T
            eig = np.linalg.eigvalsh(A)  # ascending
            return float(np.sqrt(max(eig[0], 0.0)))

        grad = np.zeros((n, 1), dtype=np.float64)
        for i in range(n):
            dq = np.zeros((n, 1), dtype=np.float64)
            dq[i, 0] = eps
            grad[i, 0] = (sigma_min(q + dq) - sigma_min(q - dq)) / (2.0 * eps)

        gnorm = float(np.linalg.norm(grad))
        if gnorm > 1e-12:
            grad /= gnorm

        return k * grad

    def plot_joint_trajectories(self, out_dir="extsUser/kinematics_analysis/assets/plots", tag=None):
        os.makedirs(out_dir, exist_ok=True)

        t = np.array(self._log_t, dtype=np.float64)
        if t.size < 2:
            print("Not enough samples to plot")
            return

        Q = np.vstack([np.array(d, dtype=np.float64) for d in self._log_q])  # (7,N)
        Q_unwrap = np.unwrap(Q, axis=1)

        if tag is None:
            tag = f"run_{self._run_id:03d}"

        # per-joint
        for i in range(7):
            plt.figure()
            plt.plot(t, Q_unwrap[i])
            plt.xlabel("time (s)")
            plt.ylabel(f"q{i+1} (rad)")
            plt.title(f"Joint {i+1} vs time")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"joint_traj_{tag}_q{i+1}.png"))
            plt.close()

        # combined
        plt.figure()
        for i in range(7):
            plt.plot(t, Q_unwrap[i], label=f"q{i+1}")
        plt.xlabel("time (s)")
        plt.ylabel("joint angle (rad)")
        plt.title("Franka joints vs time")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        combined = os.path.join(out_dir, f"joint_traj_{tag}.png")
        plt.savefig(combined)
        plt.close()

        print(f"Saved plots to {out_dir} (tag={tag})")
        
        if sys.platform.startswith("win"):
            os.startfile(combined)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", combined])
        else:
            subprocess.Popen(["xdg-open", combined])
    def clear_joint_log(self):
        self._log_t.clear()
        for d in self._log_q:
            d.clear()


    def get_joint_log(self, unwrap=True):
        nt = len(self._log_t)
        if nt < 2:
            return [], [[] for _ in range(7)]

        nq = min(len(d) for d in self._log_q)
        n = min(nt, nq)
        t = np.asarray(list(self._log_t)[-n:], dtype=np.float64)

        qs = []
        for j in range(7):
            q = np.asarray(list(self._log_q[j])[-n:], dtype=np.float64)
            qs.append(np.unwrap(q) if unwrap else q)

        return t.tolist(), [q.tolist() for q in qs]

