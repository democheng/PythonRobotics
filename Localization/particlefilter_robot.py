import sys
import math
import copy
import numpy as np
from scipy.stats import norm
from scipy.ndimage import gaussian_filter1d

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def gaussian_entropy(sig):
    entropy = np.log(sig * np.sqrt(2 * np.pi * np.e))
    return entropy

class particle_map():
    def __init__(self, minx, maxx, number):
        self.number = number
        self.position = np.random.uniform(minx, maxx, number)
        self.weight = np.ones(self.number) * 1.0 / self.number
        self.minx = self.position[np.argmin(self.position)]
        self.maxx = self.position[np.argmax(self.position)]
    
    def init(self):
        self.number = 2
        self.position = np.random.uniform(0.0, 1.0, self.number)
        self.weight = np.ones(self.number) * 1.0 / self.number
        self.minx = self.position[np.argmin(self.position)]
        self.maxx = self.position[np.argmax(self.position)]
    
    def print(self):
        print('[minx, max] = [', self.minx, ',', self.maxx, ']')
        print('number = ', self.number)
        print('position = ', self.position)
        print('weight = ', self.weight)
    
    def normalize_weight(self):
        self.weight /= np.sum(self.weight)
    
    def current_position(self):
        return np.sum(self.position * self.weight)

    def Low_variance_sampler(self):
        Neff = np.sum(self.weight * self.weight)
        Nth = 0.6 * self.number
        ret = False
        if Neff < Nth:
            weight_cumsum = np.zeros(self.number + 1)
            weight_cumsum[1:] = np.cumsum(self.weight)
            idx = np.digitize(np.random.uniform(0.0, 1.0, self.number), weight_cumsum) - 1
            self.position = self.position[idx]
            self.weight = np.ones(self.number) * 1.0 / self.number
            ret = True
        return ret
    
    # the move_length is an array, since every particle is moving independently
    # the weight doesn't change, since the state has been changed
    # the information is also in the changed state
    def motion_predict(self, move_length):
        self.position += move_length
        return self.current_position()
    
    def measurement_update(self, min_distance, map_data, measurement_noise):
        for i in range(self.number): # loop all the estimated positions
            cur_pos = self.position[i]
            min_dis = 1e3
            for j in range(len(map_data)):
                delta = np.abs(cur_pos - map_data[j])
                if delta < min_dis:
                    min_dis = delta
            # likelihood
            pdf = 1.0 - norm.cdf(np.abs(min_dis - min_distance), 0.0, measurement_noise)
            self.weight[i] *= pdf
        self.normalize_weight()
        return self.current_position()
    
    def particle_map_entropy(self):
        var = np.var(self.position * self.weight)
        return gaussian_entropy(var)

class pfrobot(object):
    def __init__(self, 
                start_position, 
                end_position, 
                velocity, 
                motion_noise, 
                measurement_noise, 
                sensor_range,
                map_data, 
                particle_map):
        self.start_position = start_position # m
        self.end_position = end_position # m
        self.velocity = velocity # m/s

        self.groundtruth = start_position

        self.motion_noise = motion_noise
        self.measurement_noise = measurement_noise

        self.sensor_range = sensor_range
        self.map = map_data

        self.particle_map = copy.deepcopy(particle_map)
        self.position = self.particle_map.current_position()
    
    def init(self):
        self.start_position = 0.0
        self.end_position = 0.0
        self.groundtruth = 0.0

        self.velocity = 0.0

        self.motion_noise = 0.0
        self.measurement_noise = 0.0
        
        self.sensor_range = 0.0
        self.map = np.array([0.0])
        self.position = 0.0

        self.particle_map = copy.deepcopy(particle_map(0.0, 10.0, 50))

    def motion(self):
        self.groundtruth += self.velocity * 1.0

    def motion_predict(self):
        self.motion()
        # print('motion_predict:current position', self.position)
        # print('motion_predict:groundtruth', self.groundtruth)
        move_length = self.velocity * 1.0 + np.random.randn(self.particle_map.number)
        self.position = self.particle_map.motion_predict(move_length)
    
    def measurement(self):
        # find what the robot senses
        min_distance = 1e3
        min_door = 1e3
        ret = False
        for item in self.map:
            delta = np.abs(item - self.groundtruth)
            if delta <= self.sensor_range and delta < min_distance:
                ret = True
                min_distance = delta
                min_door = item
        return ret, min_distance, min_door
    
    def measurement_update(self):
        ret, min_distance, _ = self.measurement()
        if ret:
            print('measurement success')
            min_distance += np.random.randn() * self.measurement_noise
            self.position = self.particle_map.measurement_update(min_distance, self.map, self.measurement_noise)
        return ret

def draw_robot(cur_pos, road_line):
    p0_x = cur_pos
    p0_y = road_line
    p1_x = cur_pos
    p1_y = road_line + 0.15
    p2_x = cur_pos - 0.1
    p2_y = road_line + 0.15
    p3_x = cur_pos
    p3_y = road_line + 0.25
    p4_x = cur_pos + 0.1
    p4_y = road_line + 0.15
    p5_x = p1_x
    p5_y = p1_y
    return np.array([[p0_x, p1_x, p2_x, p3_x, p4_x, p5_x], 
                    [p0_y, p1_y, p2_y, p3_y, p4_y, p5_y]])

def draw_gaussian(bin_num, mu, sig):
    sig3 = 3.0 * sig
    step = 2.0 * sig3 / bin_num
    x = np.arange(mu - sig3, mu + sig3 + step, step)
    y = norm.pdf(x, mu, sig)
    data = np.zeros((2, len(x)))
    data[0, :] = x
    data[1, :] = y
    return data

def draw_robot_groundtruths(robot_groundtruths, road_line):
    res = []
    for item in robot_groundtruths:
        res.append(draw_robot(item, road_line))
    return res

def draw_robot_groundtruths_with(robot_groundtruths):
    res = []
    for item in robot_groundtruths:
        res.append(np.array([[item, item], [0.0, 1.0]]))
    return res

def draw_robot_positions(robot_positions_with_noise, road_line):
    res = []
    for item in robot_positions_with_noise:
        res.append(draw_robot(item[0], road_line))
    return res

def draw_robot_entropy(robot_groundtruths, robot_positions_with_noise):
    res = np.zeros((2, len(robot_groundtruths)))
    for idx in range(res.shape[1]):
        res[0, idx] = robot_groundtruths[idx]
        item = copy.deepcopy(robot_positions_with_noise[idx])
        res[1, idx] = item[1].particle_map_entropy()
    return res

def draw_robot_particle_map(particle_map):
    res = np.zeros((2, 3 * particle_map.number))
    index = 0
    for i in range(particle_map.number):
        x = particle_map.position[i]
        y = 0
        res[0, index] = x
        res[1, index] = y
        index += 1

        res[0, index] = x
        res[1, index] = y + particle_map.weight[i]
        index += 1

        res[0, index] = x
        res[1, index] = y
        index += 1
    return res

def draw_robot_positions_with_noise(robot_positions_with_noise):
    res = []
    for item in robot_positions_with_noise:
        res.append(draw_robot_particle_map(item[1]))
    return res

def draw_door(cur_pos, road_line):
    w = 1.0
    h = 0.3
    left_down_x = cur_pos - w
    left_down_y = road_line
    left_up_x = left_down_x
    left_up_y = road_line + h
    right_up_x = cur_pos + w
    right_up_y = left_up_y
    right_down_x = right_up_x
    right_down_y = left_down_y
    return np.array([[left_down_x, left_up_x, right_up_x, right_down_x], 
                    [left_down_y, left_up_y, right_up_y, right_down_y]])

def draw_doors(doors, road_line):
    res = draw_door(doors[0], road_line)
    for item in doors[1:]:
        res = np.concatenate((res, draw_door(item, road_line)), axis=1)
    return res

def draw_doors_with_noise(doors, measurement_noise):
    bin_num = 50
    res = draw_gaussian(bin_num, doors[0], measurement_noise)
    for item in doors[1:]:
        res = np.concatenate((res, draw_gaussian(bin_num, item, measurement_noise)), axis=1)
    return res

def draw_road_line(start_position, end_position, road_line):
    return np.array([[start_position, end_position], 
                    [road_line, road_line]])

class update_draw(object):
    def __init__(self, ax, 
                start_position, end_position,
                groundtruth, groundtruth_with, positions, doors, road_line,
                positions_with_noise, doors_with_noise, entropy):
        self.groundtruth = groundtruth
        self.groundtruth_with = groundtruth_with
        self.positions = positions
        self.doors = doors
        self.road_line = road_line
        self.positions_with_noise = positions_with_noise
        self.doors_with_noise = doors_with_noise
        self.entropy = entropy

        ## Setting the axes properties
        ax.set_xlim([start_position, end_position])
        ax.set_xlabel('X')
        ax.set_ylim([0, self.road_line[1, 0] * 2])
        ax.set_ylabel('Y')
        ax.set_title('kalman filter')

        self.lines = [ax.plot([], [], color='b')[0],   # groundtruth
                    ax.plot([], [], color='b')[0],     # groundtruth_with
                    ax.plot([], [], color='g')[0],     # positions
                    ax.plot([], [], color='black')[0], # doors
                    ax.plot([], [], color='black')[0], # road_line
                    ax.plot([], [], color='g')[0],     # positions_with_noise
                    ax.plot([], [], color='black')[0], # doors_with_noise
                    ax.plot([], [], color='gray')[0]] # entropy
        # ax.legend((self.lines[0], self.lines[2], self.lines[3], self.lines[7]), 
        #         ('groundtruth', 'positions with uncertainty', 'doors', 'entropy'))
        
        ax.legend((self.lines[0], self.lines[2], self.lines[3]), 
                ('groundtruth', 'positions with uncertainty', 'doors'))
        
    def init(self):
        for line in self.lines:
            line.set_data([], [])
        return self.lines
    
    def __call__(self, i):
        # This way the plot can continuously run and we just keep
        # watching new realizations of the process
        self.lines[0].set_data(self.groundtruth[i])
        self.lines[1].set_data(self.groundtruth_with[i])
        self.lines[2].set_data(self.positions[i])
        self.lines[3].set_data(self.doors)
        self.lines[4].set_data(self.road_line)
        self.lines[5].set_data(self.positions_with_noise[i])
        self.lines[6].set_data(self.doors_with_noise)
        # self.lines[7].set_data(self.entropy[:, :i + 1])
        return self.lines
    

def main():
    start_position = 0.0
    end_position = 50.0
    velocity = 0.5
    motion_noise = np.sqrt(0.08)
    measurement_noise = np.sqrt(0.5)
    sensor_range = 0.1
    doors = np.array([5.0, 10.0, 15.0, 25.0, 40.0])

    road_line = 1.5

    pf_particle_map = copy.deepcopy(particle_map(0.0, 9.5, 30))

    pf_robot = pfrobot(start_position, 
                end_position, 
                velocity, 
                motion_noise, 
                measurement_noise, 
                sensor_range,
                doors,
                pf_particle_map)

    loop_num = int((end_position - start_position) / velocity)

    # dynamic
    robot_positions_with_noise = []
    robot_groundtruths = []
    # pf_robot.particle_map.print()
    robot_positions_with_noise.append(np.array([pf_robot.position, copy.deepcopy(pf_robot.particle_map) ]))
    robot_groundtruths.append(pf_robot.groundtruth)

    for loop in range(loop_num):
        # print(pf_robot.position)
        ret = pf_robot.measurement_update()
        if ret:
            # pf_robot.particle_map.print()
            robot_positions_with_noise.append(np.array([pf_robot.position, copy.deepcopy(pf_robot.particle_map) ]))
            robot_groundtruths.append(pf_robot.groundtruth)
        
        pf_robot.motion_predict()
        # pf_robot.particle_map.print()
        robot_positions_with_noise.append(np.array([pf_robot.position, copy.deepcopy(pf_robot.particle_map) ]))
        robot_groundtruths.append(pf_robot.groundtruth)

        pf_robot.particle_map.Low_variance_sampler()
    
    # convert data for drawing
    d_groundtruths = draw_robot_groundtruths(robot_groundtruths, road_line)
    d_groundtruths_with = draw_robot_groundtruths_with(robot_groundtruths)
    d_robot_positions = draw_robot_positions(robot_positions_with_noise, road_line)
    d_road_line = draw_road_line(start_position, end_position, road_line)
    d_doors = draw_doors(doors, road_line)
    d_positions_with_noise = draw_robot_positions_with_noise(robot_positions_with_noise)
    d_doors_with_noise = draw_doors_with_noise(doors, measurement_noise)
    d_entropy = draw_robot_entropy(robot_groundtruths, robot_positions_with_noise)
    # draw
    fig = plt.figure()
    ax = plt.axes()

    ud = update_draw(ax, start_position, end_position,
                d_groundtruths, d_groundtruths_with, d_robot_positions, d_doors, d_road_line,
                d_positions_with_noise, d_doors_with_noise, d_entropy)
    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = FuncAnimation(fig, ud, frames=loop_num, interval=200, blit=True)
    # anim.save('histogramfilter.gif', dpi=80, writer='imagemagick')
    plt.show()

if __name__ == '__main__':
    main()