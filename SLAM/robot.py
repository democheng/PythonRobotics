import numpy as np
import copy
import math
from gridmap2d import *

class robot(object):
    def __init__(self, 
                x = 0.0, 
                y = 0.0, 
                orientation = 0.0,
                forward_noise = 0.0, 
                turn_noise = 0.0, 
                sense_noise = 0.0,
                gridmap = gridmap2d()):
        self.x = x
        self.y = y
        self.orientation = orientation
        self.forward_noise = forward_noise
        self.turn_noise = turn_noise
        self.sense_noise = sense_noise
        self.gridmap = copy.deepcopy(gridmap)
    
    def sense(self):
        pass
    
    def move(self, turn, forward):
        pass
    
    def print_sense(self):
        pass
           
    def print_pose(self):
        print(self.x, ' m, ', self.y, ' m, ', self.orientation / math.pi * 180.0, ' degree')
