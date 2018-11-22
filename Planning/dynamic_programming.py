import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation

from collections import deque

from common import * 

def dynamic_programming(robotmap, row, col, target_row, target_col):
    move = np.array([[-1, 0], [0, -1], [1, 0], [0, 1]])
    state_map = np.ones((robotmap.shape[0], robotmap.shape[1])) * 1e9
    changed = True
    while changed:
        changed = False
        for i in range(robotmap.shape[0]):
            for j in range(robotmap.shape[1]):
                if i == target_row and j == target_col:
                    if state_map[i, j] > 0:
                        state_map[i, j] = 0
                        changed = True
                elif robotmap.bool_map[i ,j] == False:
                    for k in range(move.shape[0]):
                        ii = i + move[k, 0]
                        jj = j + move[k, 1]
                        if ii >= 0 and jj >= 0 and ii < robotmap.shape[0] and jj < robotmap.shape[1]:
                            vv = state_map[ii, jj] + 1
                            if vv < state_map[i, j]:
                                changed = True
                                state_map[i, j] = vv
    print(state_map)
    min_distance = state_map[row, col] - state_map[target_row, target_col] 
    print('min_distance = ', min_distance)

    is_visited = np.zeros(robotmap.shape, dtype=bool)
    unique_id = np.zeros(robotmap.shape)
    id_map = {}
    count = 0
    for i in range(robotmap.shape[0]):
        for j in range(robotmap.shape[1]):
            unique_id[i, j] = count
            id_map[count] = [i, j]
            count += 1
    parent_map = {unique_id[row, col]:None}

    ret = False
    queue = deque([[row, col]])
    dfs_path = []
    while(len(queue) > 0):
        idx = queue.popleft()
        row = idx[0]
        col = idx[1]
        if is_visited[row, col] != True:
            is_visited[row, col] = True
            dfs_path.append([row, col])
        
        if row == target_row and col == target_col:
            ret = True
            res_path = get_result_path(unique_id, id_map, parent_map, target_row, target_col)
            return ret, dfs_path, res_path
        
        # ------add neighbors-------
        # up 
        if row - 1 >= 0 and \
            is_visited[row - 1, col] != True and \
            robotmap.bool_map[row - 1, col] != True:
            queue.append([row - 1, col])
            parent_map[ unique_id[row - 1, col] ] = unique_id[row, col]
        #left
        if col - 1 >= 0 and \
            is_visited[row, col - 1] != True and \
            robotmap.bool_map[row, col - 1] != True:
            queue.append([row, col - 1])
            parent_map[ unique_id[row, col - 1] ] = unique_id[row, col]
        # down
        if row + 1 < robotmap.bool_map.shape[0] and \
            is_visited[row + 1, col] != True and \
            robotmap.bool_map[row + 1, col] != True:
            queue.append([row + 1, col])
            parent_map[ unique_id[row + 1, col] ] = unique_id[row, col]
        # right
        if col + 1 < robotmap.bool_map.shape[1] and \
            is_visited[row, col + 1] != True and \
            robotmap.bool_map[row, col + 1] != True:
            queue.append([row, col + 1])
            parent_map[ unique_id[row, col + 1] ] = unique_id[row, col]

    return ret, dfs_path, parent_map


def main():
    robotmap = robot_map(40, 40, 0.099, 0.9)
    robotmap.generate_map()
    row, col = robotmap.get_start_position()
    target_row, target_col = robotmap.get_target_position()
    
    ret, dfs_path, res_path = dynamic_programming(robotmap, row, col, target_row, target_col)
    print('start position:', row, col)
    print('target position:', target_row, target_col)

    dfs_path = np.array(dfs_path)
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
    image = robotmap.get_image()
    # draw start and target
    image[row, col] = 50
    image[target_row, target_col] = 200
    im = plt.imshow(image, cmap=plt.cm.get_cmap('jet'), animated=True, interpolation='nearest')
    images.append([im])
    # draw path
    for i in range(1, dfs_path.shape[0] - 1):
        image[dfs_path[i, 0], dfs_path[i, 1]] = 100
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