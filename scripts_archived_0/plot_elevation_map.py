import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_elevation_map(
    elevationMap,
    z_range=[0, 1],
    size=[0.3762, 0.0935, 0.114],
    scaler=0.1
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
    fig = plt.figure(figsize=(72, 30))
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

    wp = [0., 0., 0.34]
    surfs_rot = surfs + wp
    ax.add_collection3d(Poly3DCollection(surfs_rot[[1]], facecolors='r', alpha=min(1.0, a*2)))
    ax.add_collection3d(Poly3DCollection(surfs_rot[[0, 2, 3, 4, 5]], facecolors='r', alpha=a))
    ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])
    
    plt.show(block=True)


if __name__ == '__main__':
    top = np.ones((1, 21, 21))
    bottom = np.zeros((1, 11, 11))
    elevationMap = np.concatenate((top, bottom), axis=0)

    plot_elevation_map(elevationMap)