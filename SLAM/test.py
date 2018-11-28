import numpy as np
from gridmap2d import *
from bresenham_algorithm import *
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation

def bresenham_line_by_world(world_map, A, B, prob):
    a = world_map.world2pixel(A)
    b = world_map.world2pixel(B)
    res = bresenham_line(a, b)
    for item in res:
        world_map.set_prob_by_pixel(item, prob)

def construct_home(world_map):
    A = np.array([0, 0]) + 1
    B = np.array([0, 10]) + 1
    bresenham_line_by_world(world_map, A, B, world_map.probrange[1])
    A = np.array([0, 10]) + 1
    B = np.array([8, 10]) + 1
    bresenham_line_by_world(world_map, A, B, world_map.probrange[1])
    A = np.array([8, 10]) + 1
    B = np.array([8, 0]) + 1
    bresenham_line_by_world(world_map, A, B, world_map.probrange[1])
    A = np.array([8, 0]) + 1
    B = np.array([0, 0]) + 1
    bresenham_line_by_world(world_map, A, B, world_map.probrange[1])
    A = np.array([4, 0]) + 1
    B = np.array([4, 5]) + 1
    bresenham_line_by_world(world_map, A, B, world_map.probrange[1])
    A = np.array([4, 8]) + 1
    B = np.array([4, 10]) + 1
    bresenham_line_by_world(world_map, A, B, world_map.probrange[1])
    A = np.array([3, 6]) + 1
    B = np.array([3, 8]) + 1
    bresenham_line_by_world(world_map, A, B, world_map.probrange[1])
    A = np.array([0, 5]) + 1
    B = np.array([3, 5]) + 1
    bresenham_line_by_world(world_map, A, B, world_map.probrange[1])
    A = np.array([5, 5]) + 1
    B = np.array([8, 5]) + 1
    bresenham_line_by_world(world_map, A, B, world_map.probrange[1])
    A = np.array([0, 8]) + 1
    B = np.array([3, 8]) + 1
    bresenham_line_by_world(world_map, A, B, world_map.probrange[1])


world_map = gridmap2d((10, 12), 0.02, (-20, 120))
images = []
fig, ax = plt.subplots()
construct_home(world_map)

im = plt.imshow(world_map.mapdata, cmap=plt.cm.get_cmap('hot'), animated=True,
    vmin=world_map.probrange[0], vmax=world_map.probrange[1])

fig.colorbar(im)
images.append([im])

ani = ArtistAnimation(fig, images, interval=1, blit=True,
                                    repeat_delay=1000)
plt.show()