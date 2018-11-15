import sys
import math
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.animation import FuncAnimation

def gaussian(x, mu, sig):
    return norm.pdf(x, mu, sig)

def gaussian_histogram(bin_num, mu, sig):
    sig3 = 3.0 * sig
    step = 2.0 * sig3 / bin_num
    x = np.arange(mu - sig3, mu + sig3 + step, step)
    y = gaussian(x, mu, sig)
    data = np.zeros((2, len(x)))
    data[0, :] = x
    data[1, :] = y
    return data

def gaussian_entropy(mu, var):
    sig = np.sqrt(var)
    entropy = np.log(sig * np.sqrt(2 * np.pi * np.e))
    return np.array([[mu],[entropy]])

def gaussian_update(mu0, var0, mu1, var1):
    mu = (mu0 * var1 + mu1 * var0) / (var0 + var1)
    std = 1.0 / (1.0 / var0 + 1.0 / var1)
    return mu, std

def gaussin_predict(mu0, var0, mu1, var1):
    mu = mu0 + mu1
    var = var0 + var1
    return mu, var

def draw_one_door(cur_pos):
    w = 1.0
    left_down_x = cur_pos - w
    left_down_y = 1.5
    left_up_x = left_down_x
    left_up_y = 1.5 + 0.3
    right_up_x = cur_pos + w
    right_up_y = left_up_y
    right_down_x = right_up_x
    right_down_y = left_down_y
    return np.array([[left_down_x, left_up_x, right_up_x, right_down_x], 
                    [left_down_y, left_up_y, right_up_y, right_down_y]])

def draw_robot(cur_pos):
    p0_x = cur_pos
    p0_y = 1.5
    p1_x = cur_pos
    p1_y = 1.5 + 0.15
    p2_x = cur_pos - 0.1
    p2_y = 1.5 + 0.15
    p3_x = cur_pos
    p3_y = 1.5 + 0.25
    p4_x = cur_pos + 0.1
    p4_y = 1.5 + 0.15
    p5_x = p1_x
    p5_y = p1_y
    return np.array([[p0_x, p1_x, p2_x, p3_x, p4_x, p5_x], 
                    [p0_y, p1_y, p2_y, p3_y, p4_y, p5_y]])


class update_prob(object):
    def __init__(self, ax, trajectory, doors, total_length, doors_geometry, robot_geometry, robot_pos_entropy):
        # The first is for robot, the second is for the doors
        self.lines = [ax.plot([], [], color='b')[0],   # robot_pos_prob
                    ax.plot([], [], color='g')[0],     # door_pos_prob
                    ax.plot([], [], color='black')[0], # road
                    ax.plot([], [], color='black')[0], # doors
                    ax.plot([], [], color='r')[0], 
                    ax.plot([], [], color='gray')[0]]     # robot entropy
        self.ax = ax
        self.ax.grid(True)
        self.trajectory = trajectory
        self.total_length = total_length
        self.doors = doors[0]
        for door in doors[1:]:
            self.doors = np.concatenate((self.doors, door), axis=1)
        self.doors_geometry = doors_geometry[0]
        for door in doors_geometry[1:]:
            self.doors_geometry = np.concatenate((self.doors_geometry, door), axis=1)
        self.robot_geometry = robot_geometry
        
        self.robot_pos_entropy = robot_pos_entropy[0]
        for entropy in robot_pos_entropy[1:]:
            self.robot_pos_entropy = np.concatenate((self.robot_pos_entropy, entropy), axis=1)
        self.robot_pos_entropy[1,:] -= self.robot_pos_entropy[1,np.argmax(self.robot_pos_entropy[1,:])]
        ## Setting the axes properties
        ax.set_xlim([-0, 50])
        ax.set_xlabel('X')
        ax.set_ylim([-1.5, 3.5])
        ax.set_ylabel('Y')
        ax.set_title('kalman filter')

    def init(self):
        for line in self.lines:
            line.set_data([], [])
        return self.lines

    def __call__(self, i):
        # This way the plot can continuously run and we just keep
        # watching new realizations of the process
        self.lines[0].set_data(self.trajectory[i])
        self.lines[1].set_data(self.doors)
        self.lines[2].set_data([0, self.total_length], [1.5, 1.5])
        self.lines[3].set_data(self.doors_geometry)
        self.lines[4].set_data(self.robot_geometry[i])
        self.lines[5].set_data(self.robot_pos_entropy[0, 0:i], self.robot_pos_entropy[1, 0:i])
        return self.lines

def main():
    smp_num = 100
    door_positions = np.array([5.0, 10.0, 15.0, 25.0, 40.0])
    door_stds = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    door_vars  = door_stds ** 2
    doors = []
    doors_geometry = []
    for mu, std in zip(door_positions, door_stds):
        doors.append(gaussian_histogram(smp_num, mu, std))
        doors_geometry.append(draw_one_door(mu))
    step_num = 100
    step_length = 0.5
    step_var = 0.04

    total_length = step_num * step_length

    cur_pos = 0.0
    cur_var = 0.09

    robot_trajectory = []
    robot_geometry = []
    robot_pos_entropy = []
    for step in range(step_num):
        # do measure, update
        for door_pos, door_var in zip(door_positions, door_vars):
            if np.abs(cur_pos - door_pos) < 0.01:
                cur_pos, cur_var = gaussian_update(cur_pos, cur_var, door_pos, door_var)
                robot_pos_entropy.append(gaussian_entropy(cur_pos, cur_var))
                robot_trajectory.append(gaussian_histogram(smp_num, cur_pos, cur_var))
                robot_geometry.append(draw_robot(cur_pos))
        # do move, predict
        cur_pos, cur_var = gaussin_predict(cur_pos, cur_var, step_length, step_var)
        robot_pos_entropy.append(gaussian_entropy(cur_pos, cur_var))
        robot_trajectory.append(gaussian_histogram(smp_num, cur_pos, cur_var))
        robot_geometry.append(draw_robot(cur_pos))
    # more specifical geometry
    # print(robot_trajectory)
    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    ax = plt.axes(xlim=(0, 2), ylim=(0, 20))

    up = update_prob(ax, robot_trajectory, doors, total_length, doors_geometry, robot_geometry, robot_pos_entropy)
    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = FuncAnimation(fig, up, frames=len(robot_trajectory), interval=200, blit=True)
    # anim.save('kalmanfilter.gif', dpi=80, writer='imagemagick')
    plt.show()

if __name__ == '__main__':
    main()