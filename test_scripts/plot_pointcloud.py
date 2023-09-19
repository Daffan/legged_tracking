import os

import numpy as np
from matplotlib import pyplot as plt

def rotate_gravity(gravity_vector, pointcloud):
    """
    args: 
        gravity_vector: 3d vector (3,)
        pointcloud: n x 3
    return:
        rotated_pointcloud: n x 3
    """
    # normalize gravity vector
    gravity_vector = gravity_vector / np.linalg.norm(gravity_vector)

    # get rotation matrix
    z_axis = np.array([0, 0, 1])
    rotation_axis = np.cross(gravity_vector, z_axis)
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    rotation_angle = np.arccos(np.dot(gravity_vector, z_axis))
    rotation_matrix = get_rotation_matrix(rotation_axis, rotation_angle)

    import ipdb; ipdb.set_trace()
    # rotate pointcloud
    rotated_pointcloud = np.matmul(rotation_matrix, pointcloud.T).T

    return rotated_pointcloud

np.random.uniform()

def get_rotation_matrix(axis, angle):
    """
    args:
        axis: 3d vector (3,)
        angle: scalar
    return:
        rotation_matrix: 3 x 3
    """
    axis = axis / np.linalg.norm(axis)
    a = np.cos(angle / 2)
    b, c, d = -axis * np.sin(angle / 2)
    rotation_matrix = np.array([
        [a*a + b*b - c*c - d*d, 2*(b*c - a*d), 2*(b*d + a*c)],
        [2*(b*c + a*d), a*a + c*c - b*b - d*d, 2*(c*d - a*b)],
        [2*(b*d - a*c), 2*(c*d + a*b), a*a + d*d - b*b - c*c]
    ])

    return rotation_matrix


if __name__ == "__main__":
    pointcloud_folder = "go1_point_clouds"
    pointcloud_files = os.listdir(pointcloud_folder)

    # load .csv files
    pointclouds = []
    for pointcloud_file in pointcloud_files:
        pointcloud = np.loadtxt(os.path.join(pointcloud_folder, pointcloud_file), delimiter=",")
        pointclouds.append(pointcloud)

    gravity_vector = np.array([0.1, -0.1, -1])
    rotate_gravity(gravity_vector, pointclouds[0])
