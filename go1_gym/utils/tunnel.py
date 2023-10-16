# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import os

from multiprocessing.sharedctypes import Value
import numpy as np
from numpy.random import choice
from scipy import interpolate

from isaacgym import terrain_utils
from go1_gym.envs.base.legged_robot_trajectory_tracking_config import Cfg
try:
    from go1_gym.utils.planner import valid_checking
except:
    pass

# import jax.numpy as jnp
# from jax import jit
# import jax
from functools import partial

class Terrain:
    def __init__(self, cfg: Cfg.terrain, num_robots) -> None:

        self.cfg = cfg
        self.num_robots = num_robots
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.horizontal_scale = cfg.horizontal_scale
        self.vertical_scale = cfg.vertical_scale
        
        self.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))
        self.terrain_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))
        self.all_terrain_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)
        self.terrain_ratio_x = cfg.terrain_ratio_x  # terrain does not occupy full area of the env, default is set to 0.5
        self.terrain_ratio_y = cfg.terrain_ratio_y  # terrain does not occupy full area of the env, default is set to 0.5

        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels)

        # Empty space by default has height points of 4 meters that does not block the robot
        self.height_field_raw = np.ones((2, self.tot_rows , self.tot_cols), dtype=np.int16) * int(1. / cfg.vertical_scale)
        self.height_field_env = []
        self.height_samples_by_row_col = np.ones((cfg.num_rows, cfg.num_cols, 2, self.length_per_env_pixels, self.width_per_env_pixels))
        self.height_samples_by_row_col[:, :, 1, :, :] = 0  # bottom 0, top 1

        for k in range(self.num_sub_terrains):
            # Env coordinates in the world
            print("Creating subterrain %d" %k, end="\r")
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))
            difficulty = np.random.uniform(0.0, 1.0)
            valid = False
            while not valid:
                terrain_top = self.make_terrain(cfg.terrain_type, difficulty, k, top=True)
                terrain_bottom = self.make_terrain(cfg.terrain_type, difficulty, k, top=False)
                # flip the ceiling of the tunnel at the ceiling
                terrain_top.height_field_raw = self.cfg.ceiling_height / self.vertical_scale - terrain_top.height_field_raw
                terrain_top.height_field_raw = np.clip(terrain_top.height_field_raw, a_max=None, a_min=0.05 / self.vertical_scale)
                
                elevation_map_top = terrain_top.height_field_raw.T * self.vertical_scale
                elevation_map_bottom = terrain_bottom.height_field_raw.T * self.vertical_scale
                elevation_map = np.stack([elevation_map_top, elevation_map_bottom])

                if k == 0:
                    visual_elevation_map(elevation_map, cfg)

                start_state = np.array([-0.375 * self.env_length, 0, 0.27, 0, 0, 0, 1.0])
                goal_state = np.array([0.375 * self.env_length, 0, 0.27, 0, 0, 0, 1.0])
                if cfg.valid_tunnel_only:
                    try:
                        valid = valid_checking(
                            elevation_map,
                            start_state,
                            goal_state,
                            self.env_length,
                            self.env_width,
                            self.terrain_ratio_y,
                            self.horizontal_scale,
                        )
                    except:
                        valid = True
                else:
                    valid = True
                
            self.add_terrain_to_map(terrain_top, terrain_bottom, i, j)

        self.vertices, self.triangles = [], []
        for hf_top_bottom in self.height_field_env:
            v_top_bottom, t_top_bottom = [], []
            for i, hf in enumerate(hf_top_bottom):
                v, t = terrain_utils.convert_heightfield_to_trimesh(
                    hf.T, self.cfg.horizontal_scale,
                    self.cfg.vertical_scale, 0.9
                )
                v_top_bottom.append(v); t_top_bottom.append(t)
            self.vertices.append(v_top_bottom); self.triangles.append(t_top_bottom)

    def make_terrain(self, terrain_type, difficulty, k, top=True):
        terrain = terrain_utils.SubTerrain(   "terrain",
                                width=int(self.width_per_env_pixels * self.terrain_ratio_y),
                                length=int(self.length_per_env_pixels * self.terrain_ratio_x),
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
        if terrain_type == "random":
            min_height = -0.05 - 0.05 * difficulty  # hardest one has min_height=-0.1
            step = 0.005 + 0.005 * difficulty  # hardest one has step=0.01
            terrain_utils.random_uniform_terrain(
                terrain, min_height=min_height,
                max_height=0.05, step=step,
                downsampled_scale=None
            )
            # terrain.height_field_raw += int(1.0 / self.cfg.vertical_scale)
        elif terrain_type == "random_pyramid":
            if difficulty < 0.25:
                d_num = 2
            elif difficulty < 0.625:
                d_num = 1
            else:
                d_num = 0
            cfg = self.cfg.top if top else self.cfg.bottom
            random_pyramid(
                terrain,
                num_x=cfg.pyramid_num_x-d_num,
                num_y=cfg.pyramid_num_y-d_num,
                var_x=cfg.pyramid_var_x,
                var_y=cfg.pyramid_var_y,
                length_min=cfg.pyramid_length_min,
                length_max=cfg.pyramid_length_max,
                height_min=cfg.pyramid_height_min,
                height_max=cfg.pyramid_height_max,
            )
        elif terrain_type == "test_env":
            test_env(terrain, top=top)
        elif terrain_type == "test_env_2":
            test_env_2(terrain, top=top)
        else:
            raise ValueError

        return terrain

    def add_terrain_to_map(self, terrain_top, terrain_bottom, row, col):
        i = row
        j = col
        #terrain.height_field_raw[0, :] = int(0.1 / self.vertical_scale)
        #terrain.height_field_raw[-1, :] = int(0.1 / self.vertical_scale)
        # map coordinate system
        start_x = int((i + 0.5 - self.terrain_ratio_x/2.) * self.length_per_env_pixels)
        end_x = int((i + 0.5 + self.terrain_ratio_x/2.) * self.length_per_env_pixels)
        start_y = int((j + 0.5 - self.terrain_ratio_y/2.) * self.width_per_env_pixels)
        end_y = int((j + 0.5 + self.terrain_ratio_y/2.) * self.width_per_env_pixels)
        self.height_field_raw[0, start_x: end_x, start_y:end_y] = terrain_top.height_field_raw.T
        self.height_field_raw[1, start_x: end_x, start_y:end_y] = terrain_bottom.height_field_raw.T
        
        self.height_field_env.append([terrain_top.height_field_raw, terrain_bottom.height_field_raw])
        self.height_samples_by_row_col[i, j, :] = (
            self.height_field_raw[
                :, 
                i*self.length_per_env_pixels: (i+1)*self.length_per_env_pixels,
                j*self.width_per_env_pixels: (j+1)*self.width_per_env_pixels,
            ]
        )
        self.terrain_origins[i, j] = [(start_x) * self.horizontal_scale, (start_y) * self.horizontal_scale, 0]

        env_origin_x = (i + 0.5 - 0.325) * self.env_length  # an offset from the center
        terrain_origin_x = i * self.env_length
        terrain_origin_y = j * self.env_width
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length/2. - 1) / self.horizontal_scale)
        x2 = int((self.env_length/2. + 1) / self.horizontal_scale)
        y1 = int((self.env_width/2. - 1) / self.horizontal_scale)
        y2 = int((self.env_width/2. + 1) / self.horizontal_scale)
        env_origin_z = 0.0 # np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]
        self.all_terrain_origins[i, j] = [terrain_origin_x, terrain_origin_y, env_origin_z]


from matplotlib import pyplot as plot

def visual_elevation_map(elevation_map, cfg):
    hs = cfg.horizontal_scale
    vs = cfg.vertical_scale
    ax = plt.figure().add_subplot(projection='3d')

    for i, m in enumerate(elevation_map):
        color = "blue" if i == 0 else "red"
        for j, l in enumerate(m):
            # if j % 4 == 0:
            y = np.arange(0, len(l)) * hs
            x = np.ones_like(y) * j * hs
            z = l  # / vs
            ax.plot(x, y, z, color=color, alpha=0.5)
    ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])

    plt.show()

def vec_plane_from_points(p1, p2, p3, xy):
    # p1, p2, p3: (num_pyramid, 4, 3)
    # xy: (px, py, 2)
    px, py = xy.shape[:2]
    xy = xy.reshape(-1, 2)

    v1 = p3 - p1
    v2 = p3 - p2

    cp = np.cross(v1, v2)
    a, b, c = cp[..., 0], cp[..., 1], cp[..., 2]

    d = np.sum(cp * p3, axis=-1)

    assert np.all(c != 0)
    heights = np.clip((d/c)[..., None] - (a/c)[..., None]*xy[..., 0] - (b/c)[..., None]*xy[..., 1], 0, a_max=np.inf)
    heights = np.min(heights, axis=1)
    heights = np.max(heights, axis=0)
    return heights.reshape(px, py)

def plane_from_points(p1, p2, p3):
    v1 = p3 - p1
    v2 = p3 - p2

    cp = np.cross(v1, v2)
    a, b, c = cp

    d = np.dot(cp, p3)

    assert c != 0
    return lambda x, y: np.clip(d/c - a/c*x - b/c*y, 0, a_max=np.inf)

def pyramid_from_points(points):
    """
    points [np.ndarray]: in shape (4, 3, 3). 4 planes with each defined by 3 points in 3D space
    """
    return lambda x, y: np.stack([plane_from_points(*ps)(x, y) for ps in points]).min(0)

def test_env_2(terrain, top=True):
    pixel_x, pixel_y = terrain.height_field_raw.shape
    l, w = pixel_x * terrain.horizontal_scale, pixel_y * terrain.horizontal_scale

    if top:
        offset_y = 0.1
        height_max, height_min = 0.30, 0.30
        lw_low, lw_high = 0.05, 0.05
    else:
        offset_y = -0.03
        height_max, height_min = 0.2, 0.2
        lw_low, lw_high = 0.05, 0.05

    mean_x = np.zeros(1)
    mean_y = np.zeros(1)
    mean_x, mean_y = np.meshgrid(mean_x, mean_y)
    mean_y += offset_y
    mean_z = np.random.uniform(mean_x) * (height_max - height_min) + height_min

    height_field_raw = np.zeros((pixel_x, pixel_y))
    means = np.stack([mean_x.flatten(), mean_y.flatten(), mean_z.flatten()], axis=1)
    pw, pl = np.random.uniform(low=lw_low, high=lw_high, size=(2, means.shape[0]))
    wedge_points = np.stack([
        np.stack([pw+means[:, 0], pl+means[:, 1], np.zeros_like(pw)], axis=1),
        np.stack([-pw+means[:, 0], pl+means[:, 1], np.zeros_like(pw)], axis=1),
        np.stack([-pw+means[:, 0], -pl+means[:, 1], np.zeros_like(pw)], axis=1),
        np.stack([pw+means[:, 0], -pl+means[:, 1], np.zeros_like(pw)], axis=1),
        means
    ], axis=1)
    idx = [
        [0, 1, -1],
        [1, 2, -1],
        [2, 3, -1],
        [3, 0, -1]
    ]
    wedge_points = wedge_points[:, idx, :]

    # shape = (pixel_x, pixel_y, x_coord, y_coord)
    points_coord = np.stack(np.meshgrid(np.linspace(-w/2, w/2, pixel_y), np.linspace(-l/2, l/2, pixel_x)), axis=-1)
    height_field_raw = vec_plane_from_points(wedge_points[:, :, 0, :], wedge_points[:, :, 1, :], wedge_points[:, :, 2, :], points_coord)
    
    if top:
        height_field_raw[0, :] = 0.5
        height_field_raw[-1, :] = 0.5

    terrain.height_field_raw = (height_field_raw / terrain.vertical_scale).astype(int)
    
    return terrain


def test_env(terrain, top=True):
    pixel_x, pixel_y = terrain.height_field_raw.shape
    l, w = pixel_x * terrain.horizontal_scale, pixel_y * terrain.horizontal_scale

    if top:
        mean_x = np.linspace(-l/2, l/2, 8)
        height_max = 0.3
        height_min = 0.2
        offset_y = 0.5
        var_y = 0.2
        lw_low = 0.2
        lw_high = 0.3
    else:
        mean_x = np.linspace(-l/2, l/2, 8)[:-1] + l / 16.0
        height_max = 0.2
        height_min = 0.1
        offset_y = 0.7
        var_y = 0.4
        lw_low = 0.1
        lw_high = 0.2

    mean_y = np.linspace(-w/2 + offset_y, w/2 - offset_y, 2)
    mean_x, mean_y = np.meshgrid(mean_x, mean_y)
    mean_y += np.random.uniform(-var_y, var_y, mean_y.shape)
    mean_z = np.random.uniform(mean_x) * (height_max - height_min) + height_min

    means = np.stack([mean_x.flatten(), mean_y.flatten(), mean_z.flatten()], axis=1)
    pw, pl = np.random.uniform(low=0.1, high=0.3, size=(2, means.shape[0]))
    wedge_points = np.stack([
        np.stack([pw+means[:, 0], pl+means[:, 1], np.zeros_like(pw)], axis=1),
        np.stack([-pw+means[:, 0], pl+means[:, 1], np.zeros_like(pw)], axis=1),
        np.stack([-pw+means[:, 0], -pl+means[:, 1], np.zeros_like(pw)], axis=1),
        np.stack([pw+means[:, 0], -pl+means[:, 1], np.zeros_like(pw)], axis=1),
        means
    ], axis=1)
    idx = [
        [0, 1, -1],
        [1, 2, -1],
        [2, 3, -1],
        [3, 0, -1]
    ]
    wedge_points = wedge_points[:, idx, :]

    # shape = (pixel_x, pixel_y, x_coord, y_coord)
    points_coord = np.stack(np.meshgrid(np.linspace(-w/2, w/2, pixel_y), np.linspace(-l/2, l/2, pixel_x)), axis=-1)
    height_field_raw = vec_plane_from_points(wedge_points[:, :, 0, :], wedge_points[:, :, 1, :], wedge_points[:, :, 2, :], points_coord)
    if top:
        height_field_raw[0, :] = 0.5
        # height_field_raw[:, 0] = 0.5
        height_field_raw[-1, :] = 0.5
        # height_field_raw[:, -1] = 0.5
    terrain.height_field_raw = (height_field_raw / terrain.vertical_scale).astype(int)
    
    return terrain

def random_pyramid(terrain, num_x=4, num_y=4, var_x=0.1, var_y=0.1, length_min=0.3, length_max=0.6, height_min=0.5, height_max=1.0, base_height=0.42):
    pixel_x, pixel_y = terrain.height_field_raw.shape
    l, w = pixel_x * terrain.horizontal_scale, pixel_y * terrain.horizontal_scale

    mean_x = np.linspace(-l/2, l/2, num_x+2)
    mean_y = np.linspace(-w/2, w/2, num_y+2)
    mean_x, mean_y = np.meshgrid(mean_x, mean_y)
    mean_x += np.random.uniform(-var_x, var_x, mean_x.shape)
    mean_x = mean_x.clip(-l/2, l/2)
    mean_y += np.random.uniform(-var_y, var_y, mean_y.shape)
    mean_y = mean_y.clip(-w/2, w/2)
    mean_z = np.random.uniform(height_min, height_max, size=mean_x.shape)
    means = np.stack([mean_x.flatten(), mean_y.flatten(), mean_z.flatten()], axis=1)

    pw, pl = np.random.uniform(low=length_min, high=length_max, size=(2, means.shape[0]))
    wedge_points = np.stack([
        np.stack([pw+means[:, 0], pl+means[:, 1], np.zeros_like(pw)], axis=1),
        np.stack([-pw+means[:, 0], pl+means[:, 1], np.zeros_like(pw)], axis=1),
        np.stack([-pw+means[:, 0], -pl+means[:, 1], np.zeros_like(pw)], axis=1),
        np.stack([pw+means[:, 0], -pl+means[:, 1], np.zeros_like(pw)], axis=1),
        means
    ], axis=1)
    idx = [
        [0, 1, -1],
        [1, 2, -1],
        [2, 3, -1],
        [3, 0, -1]
    ]
    wedge_points = wedge_points[:, idx, :]

        # shape = (pixel_x, pixel_y, x_coord, y_coord)
    points_coord = np.stack(np.meshgrid(np.linspace(-w/2, w/2, pixel_y), np.linspace(-l/2, l/2, pixel_x)), axis=-1)
    height_field_raw = vec_plane_from_points(wedge_points[:, :, 0, :], wedge_points[:, :, 1, :], wedge_points[:, :, 2, :], points_coord)
    terrain.height_field_raw = (height_field_raw / terrain.vertical_scale).astype(int)
    
    return terrain


'''
# ----------------- JAX -----------------

def jit_plane_from_points(p1, p2, p3):
    v1 = p3 - p1
    v2 = p3 - p2

    cp = jnp.cross(v1, v2)
    a, b, c = cp

    d = jnp.dot(cp, p3)

    # assert c != 0
    return lambda x, y: jnp.clip(d/c - a/c*x - b/c*y, 0, a_max=jnp.inf)

def jit_pyramid_from_points(points):
    """
    points [np.ndarray]: in shape (4, 3, 3). 4 planes with each defined by 3 points in 3D space
    """
    return lambda x, y: jnp.stack([jit_plane_from_points(*ps)(x, y) for ps in points]).min(0)

@partial(jax.jit, static_argnames=["terrain", "num_x", "num_y", "var_x", "var_y", "length_min", "length_max", "height_min", "height_max", "base_height"])
def jit_random_pyramid(key, terrain, num_x=4, num_y=4, var_x=0.1, var_y=0.1, length_min=0.3, length_max=0.6, height_min=0.5, height_max=1.0, base_height=0.42):
    pixel_x, pixel_y = terrain.height_field_raw.shape
    l, w = pixel_x * terrain.horizontal_scale, pixel_y * terrain.horizontal_scale
    keys = jax.random.split(key, 4)

    mean_x = jnp.linspace(-l/2, l/2, num_x)
    mean_y = jnp.linspace(-w/2, w/2, num_y)
    mean_x, mean_y = jnp.meshgrid(mean_x, mean_y)
    mean_x += jax.random.uniform(keys[0], minval=-var_x, maxval=var_x, shape=mean_x.shape)
    mean_x = mean_x.clip(-l/2, l/2)
    mean_y += jax.random.uniform(keys[1], minval=-var_y, maxval=var_y, shape=mean_y.shape)
    mean_y = mean_y.clip(-w/2, w/2)
    mean_z = jax.random.uniform(keys[2], minval=height_min, maxval=height_max, shape=mean_x.shape)
    means = jnp.stack([mean_x.flatten(), mean_y.flatten(), mean_z.flatten()], axis=1)

    pw, pl = jax.random.uniform(keys[3], minval=length_min, maxval=length_max, shape=(2, len(means)))
    wedge_points = jnp.stack([
        jnp.stack([pw+means[:, 0], pl+means[:, 1], jnp.zeros_like(pw)], axis=1),
        jnp.stack([-pw+means[:, 0], pl+means[:, 1], jnp.zeros_like(pw)], axis=1),
        jnp.stack([-pw+means[:, 0], -pl+means[:, 1], jnp.zeros_like(pw)], axis=1),
        jnp.stack([pw+means[:, 0], -pl+means[:, 1], jnp.zeros_like(pw)], axis=1),
        means
    ], axis=1)
    idx = [
        [0, 1, -1],
        [1, 2, -1],
        [2, 3, -1],
        [3, 0, -1]
    ]
    wedge_points = wedge_points[:, idx, :]
    

    def f(x, y):
        return jnp.max(jnp.stack([jit_pyramid_from_points(wps)(x, y) for wps in wedge_points]))

    height_field_raw = jnp.zeros_like(terrain.height_field_raw)
    for xi, x in enumerate(jnp.linspace(-l/2, l/2, pixel_x)):
        for yi, y in enumerate(jnp.linspace(-w/2, w/2, pixel_y)):
            height_field_raw = height_field_raw.at[xi, yi].set(f(x, y))
    
    return height_field_raw
'''

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_elevation_map(
    elevationMap,
    z_range=[0, 1],
    size=[0.3762, 0.0935, 0.114],
    scaler=0.1,
    save_fig=False,
):
    """ Plot the elevation map
    Args:
        elevationMap: 3D numpy array of shape (2, n_pixel_x, n_pixel_y)
        z_range: tuple of (min_z, max_z)
        size: size of the robot (x, y, z)
        scaler: horizontal scaler
    """

    height_points = []
    for i in [0, 1]:
        for x in range(elevationMap.shape[1]):
            for y in range(elevationMap.shape[2]):
                if z_range[0] <= elevationMap[i, x, y] < z_range[1]:
                    height_points.append(np.array([
                        (x - elevationMap.shape[1] // 2) * scaler,
                        (y - elevationMap.shape[2] // 2) * scaler,
                        elevationMap[i, x, y]
                    ]))
    height_points = np.stack(height_points)
    if save_fig:
        fig = plt.figure(figsize=(72, 36))    
    else:
        fig = plt.figure(figsize=(3.6, 2.4))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(height_points[:, 0], height_points[:, 1], height_points[:, 2])

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

    wp = [0., 0., 0.34]; a = 0.3
    surfs_rot = surfs + wp
    ax.add_collection3d(Poly3DCollection(surfs_rot[[1]], facecolors='r', alpha=min(1.0, a*2)))
    ax.add_collection3d(Poly3DCollection(surfs_rot[[0, 2, 3, 4, 5]], facecolors='r', alpha=a))
    ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])
    if save_fig:
        plt.savefig("elevation_map.png", dpi=300)
        data = None
    else:
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
    # plt.show(block=True)
    plt.close()

    return data