import sys
import numpy as np

class gridmap2d(object):
    """
    @brief 2D matrix for grid map
    
    @param mapsize: (width, height) of the 2d grid map; unit is m
    @param resolution: unit is m
    @param dtype: data type
    """
    def __init__(self, mapsize = (50.0, 50.0), 
                resolution = 0.1, 
                probrange = (-20.0, 120.0), 
                dtype = np.float32):
        self.mapsize = mapsize
        self.resolution = resolution
        self.dtype = dtype
        self.probrange = probrange
        self.width = int(self.mapsize[0] / self.resolution) + 1
        self.height = int(self.mapsize[1] / self.resolution) + 1
        self.mapdata = np.zeros((self.width, self.height), dtype = self.dtype)
    
    def world2pixel(self, world_location):
        res = world_location / self.resolution
        return res.astype(int)

    def pixel2world(self, pixel_location):
        res = pixel_location * self.resolution
        return res.astype(np.float32)
    
    def get_prob_by_pixel(self, key):
        if key[0] < 0 or key[1] < 0 or key[0] >= self.width or key[1] >= self.height:
            print(key, 'is out of boundary ', self.width, self.height)
        row = max(0, key[0])
        row = min(self.width - 1, key[0])
        col = max(0, key[1])
        col = min(self.height - 1, key[1])
        return self.mapdata[row, col]
    
    def get_prob_by_world(self, key):
        return self.get_prob_by_pixel(self.world2pixel(key))
    
    def set_prob_by_pixel(self, key, value):
        if key[0] < 0 or key[1] < 0 or key[0] >= self.width or key[1] >= self.height:
            print(key, 'is out of boundary ', self.width, self.height)
        row = max(0, key[0])
        row = min(self.width - 1, key[0])
        col = max(0, key[1])
        col = min(self.height - 1, key[1])
        self.mapdata[row, col] = value
    
    def set_prob_by_world(self, key, value):
        self.set_prob_by_pixel(self.world2pixel(key), value)