import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from common import * 

def deep_first_search(robotmap, row, col, target_row, target_col):
    is_visited = np.zeros(robotmap.shape, dtype=bool)
    ret = False
    stack = [[row, col]]
    dfs_path = []
    while(len(stack) > 0):
        idx = stack.pop()
        row = idx[0]
        col = idx[1]
        if is_visited[row, col] != True:
            is_visited[row, col] = True
            dfs_path.append([row, col])
        
        if row == target_row and col == target_col:
            ret = True
            return ret, dfs_path
        
        # ------add neighbors-------
        # up 
        if row - 1 >= 0 and \
            is_visited[row - 1, col] != True and \
            robotmap.bool_map[row - 1, col] != True:
            stack.append([row - 1, col])
        #left
        if col - 1 >= 0 and \
            is_visited[row, col - 1] != True and \
            robotmap.bool_map[row, col - 1] != True:
            stack.append([row, col - 1])
        # down
        if row + 1 < robotmap.bool_map.shape[0] and \
            is_visited[row + 1, col] != True and \
            robotmap.bool_map[row + 1, col] != True:
            stack.append([row + 1, col])
        # right
        if col + 1 < robotmap.bool_map.shape[1] and \
            is_visited[row, col + 1] != True and \
            robotmap.bool_map[row, col + 1] != True:
            stack.append([row, col + 1])

    return ret, dfs_path


def main():
    robotmap = robot_map(40, 40, 0.05, 0.5)
    robotmap.generate_map()
    row, col = robotmap.get_start_position()
    target_row, target_col = robotmap.get_target_position()
    
    ret, dfs_path = deep_first_search(robotmap, row, col, target_row, target_col)
    print('start position:', row, col)
    print('target position:', target_row, target_col)

    dfs_path = np.array(dfs_path)

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
    ani = ArtistAnimation(fig, images, interval=len(images), blit=True,
                                    repeat_delay=100)
    plt.show()

if __name__ == '__main__':
    main()