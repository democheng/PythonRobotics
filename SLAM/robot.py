import numpy as np
import copy
import math
from gridmap2d import *
from bresenham_algorithm import * 

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

#    c(0,0)
#      |
#     /\  
#--  /  \  --
#   /    \ 
#   ------
#      |
#d(-0.6,-0.2) e(-0.6, 0.2)
class robot(object):
    """
    @brief the robot has four lidars(one line), each one can sense a max distance at 5m
    """
    def __init__(self, 
                x = 0.0, # m
                y = 0.0, # m
                orientation = 0.0, # radian

                forward_speed = 0.0, # m/s
                forward_noise = 0.0, # standard deviation

                turn_speed = 0.0, # radian/s
                turn_noise = 0.0, # standard deviation

                sense_distance = 0.0, # m
                sense_noise = 0.0, # standard deviation
                
                map_gt = gridmap2d()):
        # estimated state (x, y, yaw)
        self.x = x
        self.y = y
        self.orientation = orientation
        # groundtruth state (x, y, yaw)
        self.x_gt = x
        self.y_gt = y
        self.orientation_gt = orientation
        # motion noise
        self.forward_noise = forward_noise
        self.turn_noise = turn_noise
        # sense noise
        self.sense_distance = sense_distance
        self.sense_noise = sense_noise
        # map_gt
        self.map_gt = copy.deepcopy(map_gt)
        # map_prob
        self.map_prob = gridmap2d(map_gt.mapsize, map_gt.resolution, map_gt.probrange)

    def sense(self):
        sensors = [np.array([0.0, self.sense_distance + np.random.randn() * self.sense_noise, 1.0]), 
                np.array([self.sense_distance + np.random.randn() * self.sense_noise, 0.0, 1.0]), 
                np.array([0.0, -self.sense_distance + np.random.randn() * self.sense_noise, 1.0]), 
                np.array([-self.sense_distance + np.random.randn() * self.sense_noise, 0.0, 1.0])]
        pose_gt = np.array([self.x_gt, self.y_gt, self.orientation_gt])
        transform_gt = pose2transform(pose_gt)

        occupied_items = []
        free_items = []
        free_items.append((self.x_gt, self.y_gt))
        for sensor in self.sensors:
            end = transform_gt @ sensor
            sense_line = bresenham_line(pose_gt, end)
            for idx in range(1, len(sense_line)):
                point = sense_line[idx]
                if map_gt.get_prob_by_world(point) == map_gt.probrange[1]:
                    occupied_items
                    break
                free_items.append(point)
            
        
    
    def move(self, turn, forward):
        pass
    
    def print_sense(self):
        pass
           
    def print_pose(self):
        print(self.x, ' m, ', self.y, ' m, ', self.orientation / math.pi * 180.0, ' degree')
