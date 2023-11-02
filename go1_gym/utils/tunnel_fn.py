import numpy as np

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

class TerrainFunctions:

    def single_path(terrain, top=True, flat=None):
        pixel_x, pixel_y = terrain.height_field_raw.shape
        l, w = pixel_x * terrain.horizontal_scale, pixel_y * terrain.horizontal_scale

        if flat is None:
            p = np.random.uniform()
        elif flat:
            p = 1.0
        else:
            p = 0.0

        if top:
            offset_y = np.random.uniform(-0.6, 0.6, size=1)
            offset_x = np.random.uniform(-0.3, 0.3, size=1)
            if p < 0.8:  # has 20% chance to be flat
                height_max, height_min = 0.35, 0.65
            else:
                height_max, height_min = 0.0, 0.0
            lw_low, lw_high = 0.2, 0.4
        else:
            offset_y = np.random.uniform(-0.4, 0.4, size=1)
            offset_x = np.random.uniform(-0.2, 0.2, size=1)
            if p < 0.8:
                height_max, height_min = 0.15, 0.25
            else:
                height_max, height_min = 0.0, 0.0
            height_max, height_min = 0.15, 0.25
            lw_low, lw_high = 0.1, 0.3

        mean_x = np.linspace(-w/2, w/2, 3)[1:-1]
        mean_y = np.zeros(1)
        mean_x, mean_y = np.meshgrid(mean_x, mean_y)
        mean_y += offset_y; mean_x += offset_x
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
        
        if not top:
            height_field_raw[0, :] = 0.5
            height_field_raw[-1, :] = 0.5
            height_field_raw[:, 0] = 0.5
            height_field_raw[:, -1] = 0.5

        terrain.height_field_raw = (height_field_raw / terrain.vertical_scale).astype(int)
        
        return terrain
    

    # -------------------- old terrains --------------------

    def test_env_3(terrain, top=True):
        pixel_x, pixel_y = terrain.height_field_raw.shape
        l, w = pixel_x * terrain.horizontal_scale, pixel_y * terrain.horizontal_scale

        if top:
            offset_y = 0.08
            height_max, height_min = 0.50, 0.50
            lw_low, lw_high = 0.1, 0.1
        else:
            offset_y = -0.08
            height_max, height_min = 0.2, 0.2
            lw_low, lw_high = 0.05, 0.05

        mean_x = np.linspace(-w/2, w/2, 8)[1:-1]
        mean_y = np.zeros(1)
        mean_x, mean_y = np.meshgrid(mean_x, mean_y)
        mean_y += offset_y
        mean_z = np.random.uniform(mean_x) * (height_max - height_min) + height_min
        # import ipdb; ipdb.set_trace()

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
        
        # if top:
        #     height_field_raw[0, :] = 0.5
        #     height_field_raw[-1, :] = 0.5

        terrain.height_field_raw = (height_field_raw / terrain.vertical_scale).astype(int)
        
        return terrain

    def test_env_4(terrain, top=True):
        pixel_x, pixel_y = terrain.height_field_raw.shape
        l, w = pixel_x * terrain.horizontal_scale, pixel_y * terrain.horizontal_scale

        if top:
            offset_y = 0.06
            height_max, height_min = 0.60, 0.60
            lw_low, lw_high = 0.1, 0.1
        else:
            offset_y = -0.08
            height_max, height_min = 0.2, 0.2
            lw_low, lw_high = 0.05, 0.05

        mean_x = np.linspace(-w/2, w/2, 8)[1:-1]
        mean_y = np.zeros(1)
        mean_x, mean_y = np.meshgrid(mean_x, mean_y)
        mean_y += offset_y
        mean_z = np.random.uniform(mean_x) * (height_max - height_min) + height_min
        # import ipdb; ipdb.set_trace()

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
        
        # if top:
        #     height_field_raw[0, :] = 0.5
        #     height_field_raw[-1, :] = 0.5

        terrain.height_field_raw = (height_field_raw / terrain.vertical_scale).astype(int)
        
        return terrain

    def test_env_5(terrain, top=True):
        pixel_x, pixel_y = terrain.height_field_raw.shape
        l, w = pixel_x * terrain.horizontal_scale, pixel_y * terrain.horizontal_scale

        if top:
            offset_y = np.random.uniform(-0.2, 0.2, size=6)
            offset_x = np.random.uniform(-0.2, 0.2, size=6)
            height_max, height_min = 0.50, 0.70
            lw_low, lw_high = 0.1, 0.1
        else:
            offset_y = np.random.uniform(-0.2, 0.2, size=6)
            offset_x = np.random.uniform(-0.2, 0.2, size=6)
            height_max, height_min = 0.15, 0.25
            lw_low, lw_high = 0.05, 0.05

        mean_x = np.linspace(-w/2, w/2, 8)[1:-1]
        mean_y = np.zeros(1)
        mean_x, mean_y = np.meshgrid(mean_x, mean_y)
        mean_y += offset_y; mean_x += offset_x
        mean_z = np.random.uniform(mean_x) * (height_max - height_min) + height_min
        # import ipdb; ipdb.set_trace()

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
        
        # if top:
        #     height_field_raw[0, :] = 0.5
        #     height_field_raw[-1, :] = 0.5

        terrain.height_field_raw = (height_field_raw / terrain.vertical_scale).astype(int)
        
        return terrain


    def test_env_6(terrain, top=True):
        pixel_x, pixel_y = terrain.height_field_raw.shape
        l, w = pixel_x * terrain.horizontal_scale, pixel_y * terrain.horizontal_scale

        if top:
            offset_y = np.random.uniform(-0.2, 0.2, size=1)
            offset_x = np.random.uniform(-0.1, 0.1, size=1)
            height_max, height_min = 0.35, 0.65
            lw_low, lw_high = 0.1, 0.1
        else:
            offset_y = np.random.uniform(-0.18, 0.18, size=1)
            offset_x = np.random.uniform(-0.1, 0.1, size=1)
            height_max, height_min = 0.15, 0.25
            lw_low, lw_high = 0.05, 0.05

        mean_x = np.linspace(-w/2, w/2, 3)[1:-1]
        mean_y = np.zeros(1)
        mean_x, mean_y = np.meshgrid(mean_x, mean_y)
        mean_y += offset_y; mean_x += offset_x
        mean_z = np.random.uniform(mean_x) * (height_max - height_min) + height_min
        # import ipdb; ipdb.set_trace()

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
        
        if not top:
            height_field_raw[0, :] = 0.4
            height_field_raw[-1, :] = 0.4
            height_field_raw[:, 0] = 0.4
            height_field_raw[:, -1] = 0.4

        terrain.height_field_raw = (height_field_raw / terrain.vertical_scale).astype(int)
        
        return terrain

    def test_env_7(terrain, top=True):
        pixel_x, pixel_y = terrain.height_field_raw.shape
        l, w = pixel_x * terrain.horizontal_scale, pixel_y * terrain.horizontal_scale

        p = 1.0  # np.random.uniform()

        if top:
            offset_y = np.random.uniform(-0.2, 0.2, size=1)
            offset_x = np.random.uniform(-0.1, 0.1, size=1)
            if p < 0.6:  # has 40% chance to be flat
                height_max, height_min = 0.35, 0.65
            else:
                height_max, height_min = 0.0, 0.0
            lw_low, lw_high = 0.1, 0.1
        else:
            offset_y = np.random.uniform(-0.18, 0.18, size=1)
            offset_x = np.random.uniform(-0.1, 0.1, size=1)
            if p < 0.6:
                height_max, height_min = 0.15, 0.25
            else:
                height_max, height_min = 0.0, 0.0
            height_max, height_min = 0.15, 0.25
            lw_low, lw_high = 0.05, 0.05

        mean_x = np.linspace(-w/2, w/2, 3)[1:-1]
        mean_y = np.zeros(1)
        mean_x, mean_y = np.meshgrid(mean_x, mean_y)
        mean_y += offset_y; mean_x += offset_x
        mean_z = np.random.uniform(mean_x) * (height_max - height_min) + height_min
        # import ipdb; ipdb.set_trace()

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
        
        if not top:
            height_field_raw[0, :] = 0.5
            height_field_raw[-1, :] = 0.5
            height_field_raw[:, 0] = 0.5
            height_field_raw[:, -1] = 0.5

        terrain.height_field_raw = (height_field_raw / terrain.vertical_scale).astype(int)
        
        return terrain

    def test_env_2(terrain, top=True):
        pixel_x, pixel_y = terrain.height_field_raw.shape
        l, w = pixel_x * terrain.horizontal_scale, pixel_y * terrain.horizontal_scale

        if top:
            offset_y = 0.08
            height_max, height_min = 0.50, 0.50
            lw_low, lw_high = 0.1, 0.1
        else:
            offset_y = -0.08
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
        
        # if top:
        #     height_field_raw[0, :] = 0.5
        #     height_field_raw[-1, :] = 0.5

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

    # def stump_env(terrain, top=True):


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
