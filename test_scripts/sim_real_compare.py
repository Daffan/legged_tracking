import numpy as np
from matplotlib import pyplot as plt

import os
import re

real_obs = np.array([
    [1.8090e+00,  4.2076e-01, -4.6422e+00,  1.0000e+00,  0.0000e+00,
    1.4500e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00, -2.4989e+00,
    1.0984e+00, -4.4550e+00,  2.7260e+00,  4.9409e-01, -5.3843e+00,
    -4.8403e+00,  8.6661e+00, -5.4352e+00,  4.0479e+00,  2.5424e+00,
    -2.8287e+00, -6.0114e-02, -2.6380e-01, -1.1867e-01,  1.4541e-02,
    -7.2388e-02,  1.0395e-05,  3.8685e-03,  1.7746e-03, -3.8462e-05,
    3.9270e-02, -1.1990e-01,  2.6439e-03, -1.7012e+00,  2.5762e+01,
    -3.2333e+01, -2.8213e+01, -5.9329e+00, -3.3677e+00, -1.1869e+01,
    4.7014e+01, -2.4294e+01, -1.3481e+01,  2.7900e+01, -3.7568e+01,],
    [1.8567e+00,  3.8402e-01, -4.6266e+00,  1.0000e+00,  0.0000e+00,
    1.4500e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00, -2.5280e+00,
    9.9246e-01, -4.5036e+00,  2.7399e+00,  4.6351e-01, -5.3843e+00,
    -4.8406e+00,  8.6661e+00, -5.4352e+00,  4.0634e+00,  2.4976e+00,
    -2.8289e+00, -7.0877e-02, -2.7445e-01, -1.2447e-01,  3.6897e-02,
    -7.8310e-02,  1.3026e-04,  1.1723e-03,  3.3609e-03, -2.8275e-05,
    3.8328e-02, -1.1399e-01,  1.0723e-04, -1.4041e+00,  2.5728e+01,
    -3.1821e+01, -2.8746e+01, -5.9254e+00, -3.5548e+00, -1.1192e+01,
    4.7357e+01, -2.4742e+01, -1.2745e+01,  2.8344e+01, -3.7811e+01],
    [1.9019e+00,  3.5520e-01, -4.6105e+00,  1.0000e+00,  0.0000e+00,
    1.4500e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00, -2.5610e+00,
    8.8619e-01, -4.5543e+00,  2.7651e+00,  4.3384e-01, -5.3843e+00,
    -4.8403e+00,  8.6658e+00, -5.4352e+00,  4.0833e+00,  2.4595e+00,
    -2.8287e+00, -8.3297e-02, -2.7124e-01, -1.2723e-01,  7.0990e-02,
    -7.5278e-02,  3.4759e-04,  1.7377e-03, -1.4909e-03, -2.8482e-05,
    4.6770e-02, -9.6838e-02,  2.4161e-03, -1.0185e+00,  2.5702e+01,
    -3.1441e+01, -2.9012e+01, -5.8934e+00, -3.6765e+00, -1.0620e+01,
    4.7749e+01, -2.5024e+01, -1.2053e+01,  2.8713e+01, -3.8188e+01],
    [1.9485e+00,  3.3386e-01, -4.5926e+00,  1.0000e+00,  0.0000e+00,
    1.4500e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00, -2.5922e+00,
    7.7053e-01, -4.6092e+00,  2.8017e+00,  3.9539e-01, -5.3843e+00,
    -4.8406e+00,  8.6658e+00, -5.4354e+00,  4.1045e+00,  2.4201e+00,
    -2.8289e+00, -6.4007e-02, -2.6398e-01, -1.2528e-01,  8.5178e-02,
    -9.0715e-02,  5.9361e-04, -2.1637e-03, -5.9414e-05, -2.1634e-03,
    4.9854e-02, -8.9392e-02, -1.0438e-04, -6.6487e-01,  2.5695e+01,
    -3.1076e+01, -2.9148e+01, -5.9084e+00, -3.8315e+00, -1.0163e+01,
    4.8012e+01, -2.5254e+01, -1.1376e+01,  2.9115e+01, -3.8507e+01],
    [1.9879e+00,  3.2317e-01, -4.5764e+00,  1.0000e+00,  0.0000e+00,
    1.4500e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00, -2.6152e+00,
    6.6940e-01, -4.6576e+00,  2.8365e+00,  3.5028e-01, -5.3843e+00,
    -4.8403e+00,  8.6658e+00, -5.4354e+00,  4.1188e+00,  2.3801e+00,
    -2.8289e+00, -5.8049e-02, -2.5632e-01, -1.2065e-01,  8.7248e-02,
    -1.1669e-01,  8.8984e-04,  1.3470e-03, -3.1391e-03, -2.6888e-03,
    3.3435e-02, -1.0415e-01, -5.2821e-04, -3.3972e-01,  2.5628e+01,
    -3.0820e+01, -2.9126e+01, -5.8813e+00, -3.8889e+00, -9.7734e+00,
    4.8175e+01, -2.5371e+01, -1.0855e+01,  2.9383e+01, -3.8784e+01]
])

sim_obs = np.array([
    [-0.2269,  0.0935, -0.9310,  1.8075, -0.0605,  0.3400,  0.0000,  0.0000,
    -0.0326,  0.3270, -0.0384, -0.9303, -0.2857,  0.1632, -0.9129,  0.2360,
    1.0981, -0.0482, -0.0738,  1.1810, -0.2964, -0.2125, -0.6386, -0.1990,
    -0.1837, -0.7721, -0.1907, -0.0095, -0.0579,  0.1332, -0.1118,  0.0772,
    -0.2755,  1.8628, -0.5370, -3.0170, -2.8381, -1.1756, -2.9059,  1.0085,
    3.3521,  0.2028, -1.0624,  4.3796, -1.7355,],
    [-0.2324,  0.1167, -0.9553,  1.7930, -0.0550,  0.3400,  0.0000,  0.0000,
    -0.0298,  0.2537, -0.2074, -0.9472, -0.3192, -0.1150, -0.9206,  0.2569,
    1.1097,  0.0175, -0.1054,  1.1643, -0.3728, -0.0830, -0.3125,  0.0518,
    -0.0706, -0.6767,  0.1262,  0.0697, -0.0433,  0.1450, -0.1408, -0.1531,
    -0.0795,  1.8974, -0.2839, -2.5636, -2.7040, -0.8242, -2.3139,  0.6685,
    2.9968,  0.3784, -1.0229,  4.3588, -1.2088],
    [-0.2770,  0.1139, -1.0029,  1.7778, -0.0478,  0.3400,  0.0000,  0.0000,
    -0.0250,  0.2313, -0.2527, -0.8939, -0.3505, -0.2775, -0.8464,  0.2456,
    1.1142,  0.0737, -0.1509,  1.1453, -0.3735, -0.0261,  0.0311,  0.2586,
    0.0252, -0.3126,  0.1926,  0.0474,  0.0674,  0.1734, -0.1203,  0.0726,
    0.1523,  2.2278,  0.0191, -2.0989, -2.1687, -0.4897, -1.9405,  0.1418,
    2.6532,  0.4997, -0.9898,  4.1971, -0.9598,],
    [-2.0702e-01,  1.0813e-01, -9.2403e-01,  1.7617e+00, -4.4454e-02,
    3.4000e-01,  0.0000e+00,  0.0000e+00, -2.1572e-02,  2.3927e-01,
    -2.2768e-01, -7.8954e-01, -3.5017e-01, -3.1004e-01, -7.5220e-01,
    2.1479e-01,  1.1093e+00,  1.6518e-01, -1.7880e-01,  1.1490e+00,
    -3.2217e-01,  1.0731e-02,  1.5908e-01,  2.9773e-01, -3.6542e-03,
    -8.2554e-02,  2.8364e-01, -6.1825e-02, -1.4051e-01,  1.9569e-01,
    -1.1191e-01, -4.0185e-02,  9.3882e-02,  2.1515e+00,  5.0460e-01,
    -1.6464e+00, -1.4970e+00,  1.1506e-02, -1.6898e+00, -2.2315e-01,
    2.7641e+00,  5.4012e-01, -5.5477e-01,  3.8845e+00, -8.2775e-01],
    [-0.1461,  0.0358, -0.9368,  1.7453, -0.0429,  0.3400,  0.0000,  0.0000,
    -0.0188,  0.2668, -0.1565, -0.6734, -0.3402, -0.2854, -0.6592,  0.1730,
    1.0551,  0.2437, -0.1755,  1.1232, -0.2671,  0.0576,  0.1951,  0.3170,
    0.0543,  0.1203,  0.2223, -0.1218, -0.0512,  0.2217, -0.0559, -0.1438,
    0.2061,  1.6550,  1.5694, -1.2794, -1.5726,  0.7309, -1.5104, -0.4297,
    2.6690,  0.5325, -0.0911,  3.2729, -0.5596,],
])

""" for i in range(45):
    plt.plot(list(range(5)), real_obs[:, i])
    plt.plot(list(range(5)), sim_obs[:, i])
    plt.title(f"dim={i}")
    plt.show()
    plt.close() """

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
x, y, z = np.array([[-1,0,0],[0,-1,0],[0,0,0]])
u, v, w = np.array([[2,0,0],[0,2,0],[0,0,5]])
ax.quiver(x,y,z,u,v,w,arrow_length_ratio=0.1, color="black")
# ax.set_axis_off()
ax.scatter(real_obs[:, 0], real_obs[:, 1], real_obs[:, 2], label="real")
ax.scatter(sim_obs[:, 0], sim_obs[:, 1], sim_obs[:, 2], label="sim")
ax.legend()
plt.show()


# visual command
