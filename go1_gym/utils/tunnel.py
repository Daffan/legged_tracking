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
from go1_gym.utils.tunnel_fn import TerrainFunctions
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
        """ This terrain contains both top and bottom constraints
        """

        self.cfg = cfg
        self.num_robots = num_robots
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.horizontal_scale = cfg.horizontal_scale
        self.vertical_scale = cfg.vertical_scale
        self.terrain_fn = getattr(TerrainFunctions, cfg.terrain_type)
        
        self.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))
        self.terrain_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))
        self.all_terrain_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)
        self.terrain_ratio_x = cfg.terrain_ratio_x  # terrain does not occupy full area of the env, default is set to 0.5
        self.terrain_ratio_y = cfg.terrain_ratio_y  # terrain does not occupy full area of the env, default is set to 0.5
        self.start_loc = cfg.start_loc # 0.4 is 0.4 * env_length away from the center of the env

        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels)

        # Set ceiling height to be self.cfg.ceiling_height
        # and floor height to be 0.0
        self.height_field_raw = np.ones((2, self.tot_rows , self.tot_cols), dtype=np.int16) * int(1. / cfg.vertical_scale) * self.cfg.ceiling_height
        self.height_field_raw[1, :, :] = 0.5 * int(1. / cfg.vertical_scale)
        self.height_field_env = []
        # this is accessed later to query the height map
        self.height_samples_by_row_col = np.zeros((cfg.num_rows, cfg.num_cols, 2, self.length_per_env_pixels, self.width_per_env_pixels))

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
                # a_min=0.05 to make sure that the obstacles do not touch the ground
                terrain_top.height_field_raw = np.clip(terrain_top.height_field_raw, a_max=None, a_min=0.05 / self.vertical_scale)
                
                elevation_map_top = terrain_top.height_field_raw.T * self.vertical_scale
                elevation_map_bottom = terrain_bottom.height_field_raw.T * self.vertical_scale
                elevation_map = np.stack([elevation_map_top, elevation_map_bottom])

                if k == 0:  # plot the first tunnel environment
                    visual_elevation_map(elevation_map, cfg)

                # check if the tunnel is valid (only if ompl is installed)
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

        self.vertices_top, self.triangles_top = terrain_utils.convert_heightfield_to_trimesh(self.height_field_raw[0],
                                                                                         self.cfg.horizontal_scale,
                                                                                         self.cfg.vertical_scale,
                                                                                         0.9)
        
        self.vertices_bottom, self.triangles_bottom = terrain_utils.convert_heightfield_to_trimesh(self.height_field_raw[1],
                                                                                         self.cfg.horizontal_scale,
                                                                                         self.cfg.vertical_scale,
                                                                                         0.9)

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
            # some configs need to be passed to the terrain_fn
            if difficulty < 0.25:
                d_num = 2
            elif difficulty < 0.625:
                d_num = 1
            else:
                d_num = 0
            cfg = self.cfg.top if top else self.cfg.bottom
            self.terrain_fn(
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
        else:
            self.terrain_fn(terrain, self.cfg, top=top)

        return terrain

    def add_terrain_to_map(self, terrain_top, terrain_bottom, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = int(round((i + 0.5 - self.terrain_ratio_x/2.) * self.length_per_env_pixels, 4))
        end_x = int(round((i + 0.5 + self.terrain_ratio_x/2.) * self.length_per_env_pixels, 4))
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

        env_origin_x = (i + 0.5 - self.start_loc) * self.env_length  # an offset from the center
        terrain_origin_x = i * self.env_length
        terrain_origin_y = j * self.env_width
        env_origin_y = (j + 0.5) * self.env_width
        env_origin_z = 0.0 # np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]
        self.all_terrain_origins[i, j] = [terrain_origin_x, terrain_origin_y, env_origin_z]


from matplotlib import pyplot as plt

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


import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_elevation_map(
    elevationMap,
    z_range=[0, 1],
    size=[0.3762, 0.0935, 0.114],
    scaler=0.1,
    show_fig=True,
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
    if show_fig:
        plt.show(block=True)
        data = None
    else:
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