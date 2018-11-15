import sys
import math
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.animation import FuncAnimation

def gaussian(x, mu, sig):
    return norm.pdf(x, mu, sig)

def draw_histogram(data, bin_length):
    # half bin length
    res = np.zeros((2, 4 * data.shape[1]))
    index = 0
    for idx in range(data.shape[1]):
        res[0, index] = data[0, idx]
        res[1, index] = 0
        index += 1

        res[0, index] = data[0, idx]
        res[1, index] = data[1, idx]
        index += 1

        res[0, index] = data[0, idx] + bin_length
        res[1, index] = data[1, idx]
        index += 1

        res[0, index] = data[0, idx] + bin_length
        res[1, index] = 0
        index += 1
    return res

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

def histogram_update(hist_cur, hist_measurement):
    # print(hist_measurement.shape)
    # print(hist_cur.shape)
    # print('------')
    if hist_cur.shape[1] < hist_measurement.shape[1]:
        hist_cur[1, :] = hist_cur[1, :] * hist_measurement[1, 0:hist_cur.shape[1]]
    else:
        hist_cur[1, 0:hist_measurement.shape[1]] = hist_cur[1, 0:hist_measurement.shape[1]] * hist_measurement[1, :]
    
    hist_cur[1, :] /= np.sum(hist_cur[1, :])
    return hist_cur
    

def histogram_predict(hist_cur, hist_move, step_length):
    m = hist_cur.shape[1]
    n = hist_move.shape[1]
    k = m + n - 1
    hist_predict = np.zeros((2, k))
    hist_predict[0,:] = np.arange(0, k * step_length, step_length)
    hist_predict[1,:] = np.convolve(hist_cur[1, :], hist_move[1, :])
    return hist_predict

def histogram_map(max_length, step_length, doors):
    x = np.arange(0, max_length + step_length, step_length)
    hist_map = np.zeros((2, len(x)))
    hist_map[0, :] = x
    for index in range(len(x)):
        for idx in range(doors.shape[1]):
            if x[index] >= doors[0, idx] - 3.0 * doors[1, idx] and \
                x[index] <=  doors[0, idx] + 3.0 * doors[1, idx]:
                hist_map[1, index] = gaussian(x[index], doors[0, idx], doors[1, idx])
    return hist_map

class update_prob(object):
    def __init__(self, ax, map, robot_hist, robot_geometry):
        # The first is for robot, the second is for the doors
        self.lines = [ax.plot([], [], color='g')[0], # map prob
                    ax.plot([], [], color='b')[0], # robot prob
                    ax.plot([], [], color='r')[0]]  
        self.ax = ax
        self.ax.grid(True)
        self.map = map
        self.robot_hist = robot_hist
        self.robot_geometry = robot_geometry

        ## Setting the axes properties
        ax.set_xlim([-0, 50])
        ax.set_xlabel('X')
        ax.set_ylim([-1.5, 3.5])
        ax.set_ylabel('Y')
        ax.set_title('histogram filter')

    def init(self):
        for line in self.lines:
            line.set_data([], [])
        return self.lines

    def __call__(self, i):
        # This way the plot can continuously run and we just keep
        # watching new realizations of the process
        self.lines[0].set_data(self.map)
        self.lines[1].set_data(self.robot_hist[i])
        self.lines[2].set_data(self.robot_geometry[i])
        return self.lines  

def main():
    hist_cur = np.array([
        [0.0, 0.5, 1.0, 1.5, 2.0], # position
        [0.2, 0.2, 0.2, 0.2, 0.2]  # probability
    ])

    hist_move = np.array([
        [0.0, 0.5, 1.0], # move length
        [0.1, 0.6, 0.3]  # probability
    ])
    mean_move = np.sum(hist_move[0,:] * hist_move[1,:])

    doors = np.array([
        [5.0, 10.0, 15.0, 25.0, 40.0], 
        [0.5, 0.5, 0.5, 0.5, 0.5]
        ])

    hist_map = histogram_map(50.0, 0.5, doors)
    step_length = 0.5
    robot_trajectory =[]
    robot_hist = []
    robot_hist.append(hist_cur)
    robot_cur = 1.0
    robot_trajectory.append(robot_cur)
    for step in range(0, 100):
        for idx in range(doors.shape[1]):
            if np.abs(robot_cur - doors[0, idx]) < doors[1, idx]:
                hist_cur = histogram_update(hist_cur, hist_map)
                robot_hist.append(hist_cur)
                robot_trajectory.append(hist_cur[0, np.argmax(hist_cur[1,:])])
        hist_cur = histogram_predict(hist_cur, hist_move, step_length)
        robot_trajectory.append(hist_cur[0, np.argmax(hist_cur[1,:])])
        robot_hist.append(hist_cur)
        

    draw_robot_trajectory = []
    for item in robot_trajectory:
        draw_robot_trajectory.append(draw_robot(item))
    draw_hist_map = draw_histogram(hist_map, 0.5)
    draw_robot_hist = []
    for item in robot_hist:
        draw_robot_hist.append(draw_histogram(item, 0.5))

    fig = plt.figure()
    ax = plt.axes(xlim=(0, 2), ylim=(0, 20))

    up = update_prob(ax, draw_hist_map, draw_robot_hist, draw_robot_trajectory)
    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = FuncAnimation(fig, up, frames=len(draw_robot_hist), interval=200, blit=True)
    # anim.save('kalmanfilter.gif', dpi=80, writer='imagemagick')
    plt.show()

if __name__ == '__main__':
    main()