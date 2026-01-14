import numpy as np
from isaacsim.util.debug_draw import _debug_draw
from isaacsim.robot_motion.motion_generation import ArticulationKinematicsSolver
from isaacsim.core.utils.numpy.rotations import quats_to_rot_matrices, rot_matrices_to_quats


class FramesViz:
    def __init__(self):
        self._dd = _debug_draw.acquire_debug_draw_interface()

    def clear(self):
        self._dd.clear_lines()

    def draw_all(self, articulation, kin_solver, frame_names, L=0.04, thickness=2.0, clear=True, max_frames=None):
        """
        Assumes kin_solver.set_robot_base_pose(...) was already called by the caller.
        Draws XYZ axes for each frame using debug_draw.draw_lines().
        """
        if clear:
            self.clear()

        starts, ends, colors, sizes = [], [], [], []

        for i, name in enumerate(frame_names):
            if max_frames is not None and i >= max_frames:
                break

            aks = ArticulationKinematicsSolver(articulation, kin_solver, name)
            p, R = aks.compute_end_effector_pose()

            p = (float(p[0]), float(p[1]), float(p[2]))
            R = np.asarray(R, dtype=float)

            x = (p[0] + L*R[0, 0], p[1] + L*R[1, 0], p[2] + L*R[2, 0])
            y = (p[0] + L*R[0, 1], p[1] + L*R[1, 1], p[2] + L*R[2, 1])
            z = (p[0] + L*R[0, 2], p[1] + L*R[1, 2], p[2] + L*R[2, 2])

            starts += [p, p, p]
            ends   += [x, y, z]
            colors += [(1,0,0,1), (0,1,0,1), (0,0,1,1)]
            sizes  += [float(thickness), float(thickness), float(thickness)]

        self._dd.draw_lines(starts, ends, colors, sizes)

    def obj_a_viz(self, endeffname, kin_solver, N=100, L=0.04, thickness=2.0, clear=True, seed=None):
        # draw task space bounds(cuboid)
        if clear:
            self.clear()
        
        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
        points = np.array([
            [0.5,-0.5, 0.6],
            [0.5, 0.5, 0.6],
            [-0.5, 0.5, 0.6],
            [-0.5,-0.5, 0.6],
            [0.5,-0.5, -0.2],
            [0.5, 0.5, -0.2],
            [-0.5, 0.5, -0.2],
            [-0.5,-0.5, -0.2]])
        lines = [
            (0,1), (1,2), (2,3), (3,0),
            (4,5), (5,6), (6,7), (7,4),
            (0,4), (1,5), (2,6), (3,7)
        ]
        start , end, color, size = [], [], [], []
        for line in lines:
            start.append(tuple(points[line[0]]))
            end.append(tuple(points[line[1]]))
            color.append((1,1,0,1))
            size.append(8.0)
        self._dd.draw_lines(start, end, color, size)
        # draw random poses in task space

        num_poses = N
        samples_success = []
        samples_failed = []
        for _ in range(num_poses):
            px = np.random.uniform(-0.5, 0.5)
            py = np.random.uniform(-0.5, 0.5)
            pz = np.random.uniform(-0.2, 0.6)
            # random orientation as quaternion
            qx = np.random.uniform(-1, 1)
            qy = np.random.uniform(-1, 1)
            qz = np.random.uniform(-1, 1)
            qw = np.random.uniform(-1, 1)
            q = np.array([qx, qy, qz, qw])
            q = q / np.linalg.norm(q)  # normalize quaternion
            
            joint_positions, success = kin_solver.compute_inverse_kinematics(endeffname,
                np.array([px, py, pz]), q)   
            if success:
                #check forward kinematics
                P, R =kin_solver.compute_forward_kinematics(endeffname,joint_positions)
                Q = rot_matrices_to_quats(R.reshape(1,3,3))[0]

                # check if P,Q matches px,py,pz,qx,qy,qz,qw
                if not (np.allclose(P, np.array([px, py, pz]), atol=1e-2) and
                        np.allclose(Q, q, atol=1e-2)):
                    print("FK/IK mismatch:", (px,py,pz),(qx,qy,qz,qw), "->", P,Q)
                    #blue for mismatch
                samples_success.append((P, Q))
                # draw frame at P,R 
                p = (float(P[0]), float(P[1]), float(P[2]))
                R = np.asarray(R, dtype=float)
                x = (p[0] + L*R[0, 0], p[1] + L*R[1, 0], p[2] + L*R[2, 0])
                y = (p[0] + L*R[0, 1], p[1] + L*R[1, 1], p[2] + L*R[2, 1])
                z = (p[0] + L*R[0, 2], p[1] + L*R[1, 2], p[2] + L*R[2, 2])
                #green for success
                self._dd.draw_lines([p], [x], [(0,1,0,1)], [float(thickness)])
                self._dd.draw_lines([p], [y], [(0,1,0,1)], [float(thickness)])
                self._dd.draw_lines([p], [z], [(0,1,0,1)], [float(thickness)])
            else:
                #print("IK failed for sampled pose:", (px,py,pz),(qx,qy,qz,qw))
                samples_failed.append((np.array([px, py, pz]), q))
                #draw red coordinate at sampled pose
                p = (float(px), float(py), float(pz))
                R = quats_to_rot_matrices(np.array([[qx, qy, qz, qw]]))[0]
                x = (p[0] + L*R[0, 0], p[1] + L*R[1, 0], p[2] + L*R[2, 0])
                y = (p[0] + L*R[0, 1], p[1] + L*R[1, 1], p[2] + L*R[2, 1])
                z = (p[0] + L*R[0, 2], p[1] + L*R[1, 2], p[2] + L*R[2, 2])
                self._dd.draw_lines([p], [x], [(1,0,0,1)], [float(thickness)])
                self._dd.draw_lines([p], [y], [(1,0,0,1)], [float(thickness)])
                self._dd.draw_lines([p], [z], [(1,0,0,1)], [float(thickness)])
        print(f"Visualized {len(samples_success)} successful IK solutions out of {num_poses} sampled poses.")
        print(f"Found {len(samples_failed)} unreachable poses.")
        return samples_failed, samples_success
    #connect waypoints to form a circle
    def draw_waypoints(self, waypoints, thickness=2.0):
        starts, ends, colors, sizes = [], [], [], []
        n = len(waypoints)
        for i in range(n):
            p1 = waypoints[i]
            p2 = waypoints[(i+1)%n]
            starts.append((float(p1[0]), float(p1[1]), float(p1[2])))
            ends.append((float(p2[0]), float(p2[1]), float(p2[2])))
            colors.append((1,1,0,1))
            sizes.append(float(thickness))
        self._dd.draw_lines(starts, ends, colors, sizes)