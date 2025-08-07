import numpy as np
import matplotlib.pyplot as plt
from sp8 import *

sp8 = Sp8(
    'ds',
    'cu',
    [47, 48]
)

# print(sp8.arr_list)

bounds = [min(sp8.arr_list[0][:, 0]), max(sp8.arr_list[0][:, 0])]

grid_points = sp8.make_grid(
    8980,
    np.ceil(bounds[0]*10)/10,
    np.floor(bounds[1]*10)/10
)
sp8.interpolate_and_average(
    grid_points=grid_points,
    # bounds=[min(sp8.arr_list[0][:, 0]), max(sp8.arr_list[0][:, 0])],
    # step=0.1,
    s=0.0001
)

# print(sp8.arr_list[0][:, 0])
# print(len(sp8.E_out[:, 0]))
# print(len(sp8.arr_avg[:, 0]))

# fig, ax = plt.subplots()
# ax.plot(sp8.arr_avg[:, 0], sp8.arr_avg[:, 1], 'kv', zorder=99)
# for a in sp8.arr_list:
#     ax.plot(a[:, 0], a[:, 1])
# plt.show()

np.savetxt('es/cu.dat', sp8.arr_avg, delimiter=' ')