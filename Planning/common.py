import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation

# code reference: https://en.wikipedia.org/wiki/Maze_generation_algorithm
class robot_map(object):
    def __init__(self, width=40, height=40, complexity=0.01, density=0.1):
        self.width = width
        self.height = height
        self.complexity = complexity
        self.density = density
        # Only odd shapes
        self.shape = ((self.height // 2) * 2 + 1, (self.width // 2) * 2 + 1)
        self.bool_map = np.zeros(self.shape, dtype=bool)

    def init(self):
        self.width = 40
        self.height = 40
        self.complexity = 0.01
        self.density = 0.1
        # Only odd shapes
        self.shape = ((self.height // 2) * 2 + 1, (self.width // 2) * 2 + 1)
        self.bool_map = np.zeros(self.shape, dtype=bool)

    def generate_map(self):
        # Adjust complexity and density relative to maze size
        complexity = int(self.complexity * (5 * (self.shape[0] + self.shape[1]))) # number of components
        density = int(self.density * ((self.shape[0] // 2) * (self.shape[1] // 2))) # size of components
        # Fill borders
        self.bool_map[0, :] = self.bool_map[-1, :] = 1
        self.bool_map[:, 0] = self.bool_map[:, -1] = 1
        # Make aisles
        for i in range(density):
            # pick a random position
            x = np.random.randint(0, self.shape[1] // 2) * 2
            y = np.random.randint(0, self.shape[0] // 2) * 2 
            self.bool_map[y, x] = 1

            for j in range(complexity):
                neighbours = []
                if x > 1:
                    neighbours.append((y, x - 2))
                if x < self.shape[1] - 2:
                    neighbours.append((y, x + 2))
                if y > 1:
                    neighbours.append((y - 2, x))
                if y < self.shape[0] - 2:
                    neighbours.append((y + 2, x))
                if len(neighbours):
                    y_,x_ = neighbours[np.random.randint(0, len(neighbours) - 1)]
                    if self.bool_map[y_, x_] == 0:
                        self.bool_map[y_, x_] = 1
                        self.bool_map[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 1
                        x, y = x_, y_
        return self.bool_map

    def get_image(self):
        img = self.bool_map + np.ones(self.bool_map.shape) - np.ones(self.bool_map.shape)
        img *= 255
        return img

    # start position is totally random
    def get_start_position(self):
        while True:
            row = np.random.randint(1, self.bool_map.shape[0] - 1)
            col = np.random.randint(1, self.bool_map.shape[0] - 1)
            if self.bool_map[row, col] == False:
                return row, col

    # target position is closest to (maxx, maxy)
    def get_target_position(self):
        for row in range(self.bool_map.shape[0]):
            for col in range(self.bool_map.shape[1]):
                r = self.bool_map.shape[0] - 1 - row
                c = self.bool_map.shape[1] - 1 - col
                if self.bool_map[r, c] == False:
                    return r, c