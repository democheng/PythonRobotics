import numpy as np
import math
from gridmap2d import *
from bresenham_algorithm import *
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
from matplotlib.animation import ArtistAnimation

def pose2transform(pose):
    """
    @brief convert (x, y , yaw) to transform matrix
    @param pose: (x, y, yaw)
    @return: 3*3 transform matrix
    """
    transform = np.identity(3)
    transform[0:2, 2] = pose[0:2]
    transform[0, 0] = math.cos(pose[2])
    transform[0, 1] = -math.sin(pose[2])
    transform[1, 0] = math.sin(pose[2])
    transform[1, 1] = math.cos(pose[2])
    return transform

def bresenham_line_by_world(world_map, A, B, prob):
    a = world_map.world2pixel(A)
    b = world_map.world2pixel(B)
    res = bresenham_line(a, b)
    for item in res:
        world_map.set_prob_by_pixel(item, prob)

def draw_circle_test(world_map):
    C = np.array([6, 6, 1]) / world_map.resolution
    res = bresenham_circle(C)
    for item in res:
        world_map.set_prob_by_pixel(item, world_map.probrange[1])

#    c(0,0)
#   /\  
#  /  \  
# /    \ 
# ------
#d(-0.6,-0.2) e(-0.6, 0.2)
def construct_robot_in_gridmap(pose):
    """
    @brief draw robot
    
    @param pose: (x, y, yaw) should be a numpy vector
    """
    transform = pose2transform(pose)
    c = np.array([0.0, 0.0, 1.0])
    d = np.array([-0.6, -0.2, 1.0])
    e = np.array([-0.6, 0.2, 1.0])
    c = transform @ c
    d = transform @ d
    e = transform @ e
    bresenham_line_by_world(world_map, e, d, world_map.probrange[1])
    bresenham_line_by_world(world_map, c, d, world_map.probrange[1])
    bresenham_line_by_world(world_map, c, e, world_map.probrange[1])

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


ell1 = Ellipse(xy = (0.0, 0.0), width = 4, height = 8, angle = 30.0, facecolor= 'yellow', alpha=0.3)
print(ell1.get_gid())

world_map = gridmap2d((10, 12), 0.02, (-20, 120))
images = []
fig, ax = plt.subplots()
construct_home(world_map)

pose = np.array([3, 3, math.pi / 2])
construct_robot_in_gridmap(pose)

draw_circle_test(world_map)

im = plt.imshow(world_map.mapdata, cmap=plt.cm.get_cmap('hot'), animated=True, 
    vmin=world_map.probrange[0], vmax=world_map.probrange[1])

fig.colorbar(im)
images.append([im])

ani = ArtistAnimation(fig, images, interval=1, blit=True,
                                    repeat_delay=10)
plt.show()