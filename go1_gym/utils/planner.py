import sys

from os.path import abspath, dirname, join
# sys.path.insert(0, join(dirname(dirname(abspath(__file__))), 'py-bindings'))
sys.path.insert(-1, "/usr/lib/python3/dist-packages")
from ompl import util as ou
from ompl import base as ob
from ompl import geometric as og

import numpy as np
from math import sqrt
import argparse

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def quat_apply(a, b):
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, :3]
    t = np.cross(xyz, b, axis=-1) * 2
    return (b + a[:, 3:] * t + np.cross(xyz, t, axis=-1)).reshape(shape)

def quat_from_euler_xyz(roll, pitch, yaw):
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    return np.array([qx, qy, qz, qw])

def quat_apply_inverse(a, b):
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, :3]
    t = np.cross(xyz, b, axis=-1) * 2
    return (b - a[:, 3:] * t + np.cross(xyz, t, axis=-1)).reshape(shape)

def get_euler_xyz(q):
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * \
        q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = np.where(np.abs(sinp) >= 1, copysign(
        np.pi / 2.0, sinp), np.arcsin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * \
        q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll % (2*np.pi), pitch % (2*np.pi), yaw % (2*np.pi)

def wrap_to_pi(angles):
    angles %= 2*np.pi
    angles -= 2*np.pi * (angles > np.pi)
    return angles

def copysign(a, b):
    a = np.array(a).repeat(b.shape[0])
    return np.abs(a) * np.sign(b)

class ValidityChecker(ob.StateValidityChecker):
    def __init__(
        self,
        StateInfo,
        elevation_map,  # (2, n_row, n_col) two elevation map representing the top and bottom
        scaler=0.1,
        size=[0.267, 0.194, 0.114],  # [0.267, 0.194, 0.114] (a1), [0.3762, 0.0935, 0.114] (go1)
        tolerance=[0.0, 0.0, 0.0],  # 1cm tolerance
        error_file="../error_measure/Feb16_22-18-30_.csv",
    ):
        super().__init__(StateInfo)
        self.elevation_map = elevation_map
        self.height_points = [[], []]
        self.size = [(s - t)/2. for s, t in zip(size, tolerance)]  # allows 1 cm tolerance
        for i in [0, 1]:
            for x in range(elevation_map.shape[1]):
                for y in range(elevation_map.shape[2]):
                    self.height_points[i].append(np.array([
                        (x - elevation_map.shape[1] // 2) * scaler,
                        (y - elevation_map.shape[2] // 2) * scaler,
                        elevation_map[i, x, y]
                    ]))
        # (2, n_row * n_col, 3)
        self.height_points = np.stack([np.stack(self.height_points[0]), np.stack(self.height_points[1])])

        if error_file:
            self.reject_fn = create_error_reject_fn(error_file)
        else:
            self.reject_fn = None

    def isValid(self, state):
        # Validation using the current state
        xyz = np.array([state[0][0], state[0][1], state[0][2]])
        quat = np.array([state[1].x, state[1].y, state[1].z, state[1].w])
        height_points = self.height_points - xyz[None, ...]
        height_points = quat_apply_inverse(quat, height_points)

        # out of either x or y range  
        # or above z upper range (higher points cannot below the upper range)   
        higher_point_valid = np.logical_or(
            np.sum(np.abs(height_points[0, :, :2]) > np.array(self.size)[None, :2], axis=1),
            height_points[0, :, 2] > np.array(self.size)[None, 2]
        )

        # out of either x or y range 
        # below z upper range (lower points cannot above the upper range)
        lower_point_valid = np.logical_or(
            np.sum(np.abs(height_points[1, :, :2]) > np.array(self.size)[None, :2], axis=1),
            height_points[1, :, 2] < -np.array(self.size)[None, 2]
        )

        if self.reject_fn is not None:
            z = state[0][2]
            quat = np.array([[state[1].x, state[1].y, state[1].z, state[1].w]])
            roll, pitch, _ = get_euler_xyz(quat)
            roll = wrap_to_pi(roll[0])
            pitch = wrap_to_pi(pitch[0])
            low_error = self.reject_fn(z, roll, pitch)
        else:
            low_error = True

        return bool(higher_point_valid.all()) and bool(lower_point_valid.all()) and bool(low_error)

class AccuracyObjective(ob.StateCostIntegralObjective):
    def __init__(self, si, error_file="../error_measure/Feb22_23-48-15_.csv"):
        super(AccuracyObjective, self).__init__(si, True)
        self.si_ = si
        self.cost_fn = create_pose_cost_fn(error_file)

    def stateCost(self, state):
        z = state[0][2]
        quat = np.array([[state[1].x, state[1].y, state[1].z, state[1].w]])
        roll, pitch, _ = get_euler_xyz(quat)
        roll = wrap_to_pi(roll[0])
        pitch = wrap_to_pi(pitch[0])
        return ob.Cost(100 * self.cost_fn(z, roll, pitch))

# Keep these in alphabetical order and all lower case
def allocatePlanner(si, plannerType):
    if plannerType.lower() == "bfmtstar":
        return og.BFMT(si)
    elif plannerType.lower() == "bitstar":
        return og.BITstar(si)
    elif plannerType.lower() == "fmtstar":
        return og.FMT(si)
    elif plannerType.lower() == "informedrrtstar":
        return og.InformedRRTstar(si)
    elif plannerType.lower() == "prmstar":
        return og.PRMstar(si)
    elif plannerType.lower() == "rrtstar":
        return og.RRTstar(si)
    elif plannerType.lower() == "sorrtstar":
        return og.SORRTstar(si)
    elif plannerType.lower() == "rrtconnect":
        planner = og.RRTConnect(si)
        planner.setRange(0.3)
        # planner.setIntermediateStates(True)
        return planner
    else:
        ou.OMPL_ERROR("Planner-type is not implemented in allocation function.")

def plot_elevation_map(elevationMap, way_points, z_range, size, scaler):
    # way_points = np.stack([startState])

    height_points = [[], []]
    for i in [0, 1]:
        for x in range(elevationMap.shape[1]):
            for y in range(elevationMap.shape[2]):
                if z_range[0] <= elevationMap[i, x, y] < z_range[1]:
                    height_points[i].append(np.array([
                        (x - elevationMap.shape[1] // 2) * scaler,
                        (y - elevationMap.shape[2] // 2) * scaler,
                        elevationMap[i, x, y]
                    ]))
    height_points[0] = np.stack(height_points[0])
    height_points[1] = np.stack(height_points[1])
    fig = plt.figure(figsize=(72, 30))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(height_points[0][:, 0], height_points[0][:, 1], height_points[0][:, 2], color="blue", alpha=0.5)
    ax.scatter3D(height_points[1][:, 0], height_points[1][:, 1], height_points[1][:, 2], color="green", alpha=0.5)

    xx = size[0]/2.; yy = size[1]/2.; zz = size[2]/2.
    cube = np.array([
        [-xx, -yy, -zz],
        [+xx, -yy, -zz],
        [+xx, +yy, -zz],
        [-xx, +yy, -zz],
        [-xx, -yy, +zz],
        [+xx, -yy, +zz],
        [+xx, +yy, +zz],
        [-xx, +yy, +zz],
    ])

    bottom = [0,1,2,3]
    top    = [4,5,6,7]
    front  = [0,1,5,4]
    right  = [1,2,6,5]
    back   = [2,3,7,6]
    left   = [0,3,7,4]

    surfs = np.stack([
        cube[bottom], cube[top], cube[front], cube[right], cube[back], cube[left]
    ])

    for wp, a in zip(way_points, np.linspace(0.1, 0.5, len(way_points))):
        surfs_rot = quat_apply(wp[3:], surfs) + wp[:3]
        ax.add_collection3d(Poly3DCollection(surfs_rot[[1]], facecolors='r', alpha=min(1.0, a*2)))
        ax.add_collection3d(Poly3DCollection(surfs_rot[[0, 2, 3, 4, 5]], facecolors='r', alpha=a))
    ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])
    
    plt.show(block=True)

def create_pose_cost_fn(error_file):
    arr = np.loadtxt(error_file, delimiter=" ", dtype=float)
    total_cost = arr[:, -1].reshape(26, 26, 26)
    z = arr[:, 0].reshape(26, 26, 26)[:, 0, 0]
    z_start = z[0]; z_end = z[-1]; z_interval = (z_end - z_start) / len(z)

    roll = arr[:, 1].reshape(26, 26, 26)[0, :, 0]
    roll_start = roll[0]; roll_end = roll[-1]; roll_interval = (roll_end - roll_start) / len(roll)
    
    pitch = arr[:, 2].reshape(26, 26, 26)[0, 0, :]
    pitch_start = pitch[0]; pitch_end = pitch[-1]; pitch_interval = (pitch_end - pitch_start) / len(pitch)

    def fn(z, roll, pitch):
        if z_start < z < z_end and roll_start < roll < roll_end and pitch_start < pitch < pitch_end:
            return total_cost[
                int((z - z_start) / z_interval),
                int((roll - roll_start) / roll_interval),
                int((pitch - pitch_start) / pitch_interval),
            ]
        else:
            return np.max(total_cost)
    return fn

def create_error_reject_fn(error_file):
    arr = np.loadtxt(error_file, delimiter=" ", dtype=float)
    z_error_measure = arr[:, 3].reshape(26, 26, 26)
    roll_error_measure = arr[:, 4].reshape(26, 26, 26)
    pitch_error_measure = arr[:, 5].reshape(26, 26, 26)
    z = arr[:, 0].reshape(26, 26, 26)[:, 0, 0]
    z_start = z[0]; z_end = z[-1]; z_interval = (z_end - z_start) / len(z)

    roll = arr[:, 1].reshape(26, 26, 26)[0, :, 0]
    roll_start = roll[0]; roll_end = roll[-1]; roll_interval = (roll_end - roll_start) / len(roll)
    
    pitch = arr[:, 2].reshape(26, 26, 26)[0, 0, :]
    pitch_start = pitch[0]; pitch_end = pitch[-1]; pitch_interval = (pitch_end - pitch_start) / len(pitch)

    def fn(z, roll, pitch):
        if z_start < z < z_end and roll_start < roll < roll_end and pitch_start < pitch < pitch_end:
            z_error = z_error_measure[
                int((z - z_start) / z_interval),
                int((roll - roll_start) / roll_interval),
                int((pitch - pitch_start) / pitch_interval),
            ]
            roll_error = roll_error_measure[
                int((z - z_start) / z_interval),
                int((roll - roll_start) / roll_interval),
                int((pitch - pitch_start) / pitch_interval),
            ]
            pitch_error = pitch_error_measure[
                int((z - z_start) / z_interval),
                int((roll - roll_start) / roll_interval),
                int((pitch - pitch_start) / pitch_interval),
            ]
            return z_error < 0.02 and roll_error < 0.3 and pitch_error < 0.3
        else:
            return False
    return fn


class SO3StateSampler(ob.SO3StateSampler):
    def __init__(self, sp, ignore_roll_pitch):
        self.ignore_roll_pitch = ignore_roll_pitch
        super(SO3StateSampler, self).__init__(sp)

    def sampleUniform(self, state):
        if not self.ignore_roll_pitch:
            roll = np.random.uniform(-np.pi/4, np.pi/4)
            pitch = np.random.uniform(-np.pi/4, np.pi/4)
        else:
            roll = pitch = 0
        yaw = np.random.uniform(-np.pi, np.pi)
        x, y, z, w = quat_from_euler_xyz(roll, pitch, yaw)
        state.x = x
        state.y = y
        state.z = z
        state.w = w
        return True


class CustomSO3Space(ob.SO3StateSpace):
    def __init__(self, ignore_roll_pitch=False):
        self.ignore_roll_pitch = ignore_roll_pitch
        super(CustomSO3Space, self).__init__()

    def allocDefaultStateSampler(self):
        return SO3StateSampler(self, self.ignore_roll_pitch)

def plan(
    elevationMap,
    startState,
    goalState,
    plannerType="rrtstar",
    objectiveType="balanced",
    x_range=[-5.0, 5.0],
    y_range=[-5.0, 5.0],
    z_range=[0.0, 0.5],
    scaler=0.1,
    runtime=5,
    tolerance=[0.0, 0.0, 0.0],
    visual=False,
    size=[0.267, 0.194, 0.114],
    trajectory_length=1,
    error_measure="../error_measure/Feb22_23-48-15_.csv",
    ignore_roll_pitch=False,
    ignore_z=False
):
    # if visual:
    #     plot_elevation_map(elevationMap, [startState, goalState], z_range, size)
    # in 2D metric space of the robot frame
    # startState should have x, y, yaw = 0, 0, 0
    # z, roll, pitch are absolute values
    space = ob.CompoundStateSpace()
    sub_space1 = ob.RealVectorStateSpace(3)
    rvb = ob.RealVectorBounds(3)
    rvb.setLow(0, x_range[0]); rvb.setHigh(0, x_range[1])
    rvb.setLow(1, y_range[0]); rvb.setHigh(1, y_range[1])
    if ignore_z:
        z_range = [0.27-0.005, 0.27+0.005]
    rvb.setLow(2, z_range[0]); rvb.setHigh(2, z_range[1])
    sub_space1.setBounds(rvb)
    space.addSubspace(sub_space1, 1.0)
    sub_space2 = CustomSO3Space(ignore_roll_pitch)
    space.addSubspace(sub_space2, 1.0)
    si = ob.SpaceInformation(space)

    validityChecker = ValidityChecker(si, elevationMap, scaler=scaler, size=size, tolerance=tolerance, error_file=error_measure)
    si.setStateValidityChecker(validityChecker)

    # set start at the middle
    # set the rotation to the current rotation
    start = ob.State(space)
    start[0] = startState[0].item()  # x
    start[1] = startState[1].item()  # y
    start[2] = startState[2].item()  # z
    start[3] = round(startState[3].item(), 2)  # axis-x
    start[4] = round(startState[4].item(), 2)  # axis-y
    start[5] = round(startState[5].item(), 2)  # axis-z
    start[6] = round(startState[6].item(), 2)  # w

    norm = (start[3]**2 + start[4]**2 + start[5]**2 + start[6]**2)**0.5
    start[3] /= norm
    start[4] /= norm
    start[5] /= norm
    start[6] /= norm
    assert np.abs((start[3]**2 + start[4]**2 + start[5]**2 + start[6]**2)**0.5 - 1) < 1e-9, \
        np.abs((start[3]**2 + start[4]**2 + start[5]**2 + start[6]**2)**0.5 - 1)


    # set the goal state
    goal = ob.State(space)
    goal[0] = goalState[0].item()  # x
    goal[1] = goalState[1].item()  # y
    goal[2] = goalState[2].item()  # z
    goal[3] = round(goalState[3].item())  # axis-x
    goal[4] = round(goalState[4].item())  # axis-y
    goal[5] = round(goalState[5].item())  # axis-z
    goal[6] = round(goalState[6].item())  # w

    norm = (goal[3]**2 + goal[4]**2 + goal[5]**2 + goal[6]**2)**0.5
    goal[3] /= norm
    goal[4] /= norm
    goal[5] /= norm
    goal[6] /= norm
    assert np.abs((goal[3]**2 + goal[4]**2 + goal[5]**2 + goal[6]**2)**0.5 - 1) < 1e-9, \
        np.abs((goal[3]**2 + goal[4]**2 + goal[5]**2 + goal[6]**2)**0.5 - 1)

    #import pdb; pdb.set_trace()

    # Create a problem instance
    pdef = ob.ProblemDefinition(si)
    # Set the start and goal states
    pdef.setStartAndGoalStates(start, goal)

    # Create the optimization objective specified by our command-line argument.
    # This helper function is simply a switch statement.
    if objectiveType == "pathlength":
        ob_objective = ob.PathLengthOptimizationObjective(si)
    elif objectiveType == "balanced":
        lengthObj = ob.PathLengthOptimizationObjective(si)
        accuracyObj = AccuracyObjective(si)
        ob_objective = ob.MultiOptimizationObjective(si)
        ob_objective.addObjective(lengthObj, 1.0)
        ob_objective.addObjective(accuracyObj, 1.0)
    elif objectiveType == "trackingerror":
        ob_objective = AccuracyObjective(si)
    pdef.setOptimizationObjective(ob_objective)

    # Construct the optimal planner specified by our command line argument.
    # This helper function is simply a switch statement.
    optimizingPlanner = allocatePlanner(si, plannerType)

    # Set the problem instance for our planner to solve
    optimizingPlanner.setProblemDefinition(pdef)
    optimizingPlanner.setup()

    # attempt to solve the planning problem in the given runtime
    solved = optimizingPlanner.solve(runtime)

    if solved:
        sp = pdef.getSolutionPath()
        #if sp.length() < trajectory_length+1:
        #    sp.interpolate(trajectory_length+1)
        cost = sp.cost(pdef.getOptimizationObjective()).value()
        print(sp.printAsMatrix())
        print('{0} found solution of path length {1:.4f} with an optimization ' \
             'objective value of {2:.4f}'.format( \
             optimizingPlanner.getName(), \
             sp.length(), \
             cost))
        wps = []
        for i in range(sp.getStateCount()):
            wp = sp.getState(i)
            wps.append(np.array([
                wp[0][0], wp[0][1], wp[0][2], wp[1].x, wp[1].y, wp[1].z, wp[1].w
            ]))
        if visual:
            plot_elevation_map(elevationMap, wps, z_range, size, scaler)
        # wp = wps[1:1+trajectory_length]
        wp = wps[1:]
        return np.stack(wp), cost
    else:
        print("No solution found.")

def triangle_elevation_map(width, angle, scaler=0.1):
    elevationMap = np.zeros((40, 40))
    y_start = int(-1 / scaler + 20); y_end = int(1 / scaler + 20)
    
    for i in range(int((0.25 / np.tan(angle) + width) / 0.1)):
        elevationMap[20 + i, y_start:y_end] = i * scaler * np.tan(angle)
    return elevationMap

def valid_checking(
    elevation_map,
    start_state,
    goal_state,
    env_length,
    env_width,
    tunnel_ratio_y,
    horizontal_scale,

):
    target_pose, cost = plan(
        elevation_map,
        start_state,
        goal_state,
        runtime=90.0,
        visual=False,
        x_range=[-env_length*0.4, env_length*0.4],
        y_range=[-env_width*tunnel_ratio_y/2, env_width*tunnel_ratio_y/2],
        z_range=[0., 0.37],
        objectiveType="pathlength",  # "balanced",
        plannerType="rrtconnect",
        trajectory_length=10,
        scaler=horizontal_scale,
        size=[0.42, 0.26, 0.13],
        tolerance=[0.05, 0.05, 0.05],
        error_measure=None,
        ignore_roll_pitch=False,
        ignore_z=False,
    )

    return np.linalg.norm(target_pose[-1, :2] - goal_state[:2]) < 0.1

if __name__ == "__main__":
    """
    elevationMap_high = np.ones((15, 15)) * 0.5
    elevationMap_low = np.zeros((15, 15))
    elevationMap = np.stack([elevationMap_high, elevationMap_low])
    startState = np.array([0, 0, 0.25, 0, 0, 1, 0])
    goalState = np.array([1, 0, 0.25, 0, 0, 1, 0])
    plan(elevationMap, startState, goalState, "pathlength")
    """

    elevationMap_high = np.ones((40, 40)) * 4.0
    elevationMap_low = np.zeros((40, 40))
    elevationMap = np.stack([elevationMap_high, elevationMap_low])
    # startState = np.array([0, 0, 0.146779, -0.022191, 0.00295649, 0, 0.999749])
    startState = np.array([0.        , 0.        , 0.25, 0., 0., 0., 1.0])
    goalState = np.array([2.99323, 0.0185709, 0.25, 0.0, 0.0, -0.00455139, 0.99999])

    plan(
        elevationMap, startState, goalState, 
        runtime=5.0, visual=True,
        x_range=[-5.0, 5.0], y_range=[-5.0, 5.0],
        objectiveType="pathlength", plannerType="prmstar"
    )
