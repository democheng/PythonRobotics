import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation

from collections import deque

from common import * 

def dynamic_programming(robotmap, row, col, target_row, target_col):
    unique_id = np.zeros(robotmap.shape)
    id_map = {}
    count = 0
    for i in range(robotmap.shape[0]):
        for j in range(robotmap.shape[1]):
            unique_id[i, j] = count
            id_map[count] = [i, j]
            count += 1
    parent_map = {unique_id[row, col]:None}

    move = np.array([[-1, 0], [0, -1], [1, 0], [0, 1]])
    max_number = robotmap.shape[0] * robotmap.shape[1]
    state_map = np.ones((robotmap.shape[0], robotmap.shape[1])) * max_number
    changed = True
    dp_path = []

    while changed:
        changed = False
        for i in range(robotmap.shape[0]):
            for j in range(robotmap.shape[1]):
                if i == target_row and j == target_col and state_map[i, j] > 0:
                    parent_map[unique_id[target_row, target_col]] = None
                    state_map[i, j] = 0
                    changed = True
                elif robotmap.bool_map[i, j] == False:
                    for k in range(move.shape[0]):
                        ii = i + move[k, 0]
                        jj = j + move[k, 1]
                        if ii >= 0 and jj >= 0 and ii < robotmap.shape[0] and jj < robotmap.shape[1]:
                            vv = state_map[ii, jj] + 1
                            if vv < state_map[i, j]:
                                changed = True
                                state_map[i, j] = vv
                                parent_map[ unique_id[i, j] ] = unique_id[ii, jj]
                                dp_path.append([i, j])
    
    ret = False
    if state_map[row, col] == max_number:
        return ret, dp_path, parent_map
    
    ret = True
    res_path = get_result_path(unique_id, id_map, parent_map, row, col)
    return ret, dp_path, res_path


def main():
    robotmap = robot_map(20, 20, 0.099, 0.9)
    robotmap.generate_map()
    row, col = robotmap.get_start_position()
    target_row, target_col = robotmap.get_target_position()
    
    print('start position:', row, col)
    print('target position:', target_row, target_col)
    ret, dp_path, res_path = dynamic_programming(robotmap, row, col, target_row, target_col)

    dp_path = np.array(dp_path)
    res_path = np.array(res_path)
    print('length of res_path = ', len(res_path) - 1)

    if ret == False:
        print('can not find a path')
        sys.exit(0)
    else:
        print('success to find a path')

    fig = plt.figure()
    ax = plt.axes()
    ax.set_xlim([0, robotmap.shape[0]])
    ax.set_xlabel('X')
    ax.set_ylim([0, robotmap.shape[1]])
    ax.set_ylabel('Y')
    ax.set_title('maze')

    images = []
    max_number = robotmap.shape[0] * robotmap.shape[1]

    image = robotmap.get_image()
    # draw start and target
    image[row, col] = 50
    image[target_row, target_col] = 200
    im = plt.imshow(image, cmap=plt.cm.get_cmap('jet'), animated=True, interpolation='nearest')
    images.append([im])
    # draw path
    for i in range(dp_path.shape[0]):
        image[dp_path[i, 0], dp_path[i, 1]] = 100
        image[row, col] = 50
        image[target_row, target_col] = 200
        im = plt.imshow(image, cmap=plt.cm.get_cmap('jet'), animated=True, interpolation='nearest')
        images.append([im])

    image = robotmap.get_image()
    image[row, col] = 50
    image[target_row, target_col] = 200
    for i in range(1, res_path.shape[0] - 1):
        image[res_path[i, 0], res_path[i, 1]] = 150
    for i in range(10):
        im = plt.imshow(image, cmap=plt.cm.get_cmap('jet'), animated=True, interpolation='nearest')
        images.append([im])

    ani = ArtistAnimation(fig, images, interval=len(images), blit=True,
                                    repeat_delay=1)
    # ani.save('dynamic_programming.gif', dpi=80, writer='imagemagick')
    plt.show()

if __name__ == '__main__':
    main()