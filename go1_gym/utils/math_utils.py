# License: see [LICENSE, LICENSES/legged_gym/LICENSE]

from typing import Tuple

import numpy as np
import torch
from isaacgym.torch_utils import quat_apply, normalize, get_euler_xyz, quat_from_euler_xyz, quat_rotate_inverse
from torch import Tensor


# @ torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)


# @ torch.jit.script
def wrap_to_pi(angles):
    angles %= 2 * np.pi
    angles -= 2 * np.pi * (angles > np.pi)
    return angles


# @ torch.jit.script
def torch_rand_sqrt_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    r = 2 * torch.rand(*shape, device=device) - 1
    r = torch.where(r < 0., -torch.sqrt(-r), torch.sqrt(r))
    r = (r + 1.) / 2.
    return (upper - lower) * r + lower


def get_scale_shift(range):
    scale = 2. / (range[1] - range[0])
    shift = (range[1] + range[0]) / 2.
    return scale, shift

def quaternion_to_roll_pitch_yaw(quat):
    # quat (*, 4) -> (*, 3)
    roll, pitch, yaw = get_euler_xyz(quat)
    rotations = torch.stack([roll, pitch, yaw], dim=-1)
    # bring to [-pi, pi]
    rotations = wrap_to_pi(rotations)
    return rotations

def quat_without_yaw(quat):
    #quat_wo_yaw = quat.clone().view(-1, 4)
    #quat_wo_yaw[:, 2] = 0.
    #quat_wo_yaw = normalize(quat_wo_yaw)
    #return quat_wo_yaw
    rotations = quaternion_to_roll_pitch_yaw(quat)
    rotations[:, 2] = 0.0
    return quat_from_euler_xyz(rotations[:, 0], rotations[:, 1], rotations[:, 2])

def quat_apply_yaw_inverse(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_rotate_inverse(quat_yaw, vec)