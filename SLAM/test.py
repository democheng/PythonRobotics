import numpy as np
from gridmap2d import *
from bresenham_algorithm import *
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation


world_map = gridmap2d((50, 50), 0.1, (-20, 120))

A = np.array([0, 0])
B = np.array([world_map.mapsize[0], world_map.mapsize[1]])
B = world_map.world2pixel(B)
res = bresenham_line(A, B)
for item in res:
    world_map.set_prob_by_pixel(item, world_map.probrange[0])
print('a')

A = np.array([world_map.mapsize[0], 0])
A = world_map.world2pixel(A)
B = np.array([0, world_map.mapsize[1]])
B = world_map.world2pixel(B)
res = bresenham_line(A, B)
for item in res:
    world_map.set_prob_by_pixel(item, world_map.probrange[1])
print('b')

A = np.array([world_map.mapsize[0] * 0.5, 0])
A = world_map.world2pixel(A)
B = np.array([0, world_map.mapsize[1]])
B = world_map.world2pixel(B)
res = bresenham_line(A, B)
for item in res:
    world_map.set_prob_by_pixel(item, world_map.probrange[1] * 0.5)
print('b')

fig, ax = plt.subplots()
im = plt.imshow(world_map.mapdata, cmap=plt.cm.get_cmap('jet'))
fig.colorbar(im)
plt.show()