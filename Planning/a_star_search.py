import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from heapq import heappush, heappop
from common import * 

def a_star_search(robotmap, row, col, target_row, target_col):
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

    node_cost = np.zeros(robotmap.shape, dtype=float)
    node_cost[robotmap.bool_map == True] = 1e9
    node_cost[robotmap.bool_map == False] = 1
    for i in range(robotmap.shape[0]):
        for j in range(robotmap.shape[1]):
            if robotmap.bool_map[i, j]:
                continue 
            node_cost[i, j] += (abs(target_row - i) + abs(target_col - j)) ** 6
    node_cost[row, col] = 0

    ret = False
    minheap = [(node_cost[row, col], [node_cost[row, col], row, col] )]
    dfs_path = []
    
    while(len(minheap) > 0):
        curnode = heappop(minheap)[1]
        cost = curnode[0]
        row = curnode[1]
        col = curnode[2]
        # print(cost, row, col)

        if is_visited[row, col] != True:
            is_visited[row, col] = True
            dfs_path.append([row, col])
        
        if row == target_row and col == target_col:
            ret = True
            res_path = get_result_path(unique_id, id_map, parent_map, target_row, target_col)
            return ret, dfs_path, res_path
        
        # ------compare neighbors-------
        # up 
        if row - 1 >= 0 and \
            is_visited[row - 1, col] != True and \
            robotmap.bool_map[row - 1, col] != True:
            tmp = node_cost[row - 1, col] + cost
            heappush(minheap, (tmp, [tmp, row - 1, col]) )
            parent_map[ unique_id[row - 1, col] ] = unique_id[row, col]
        #left
        if col - 1 >= 0 and \
            is_visited[row, col - 1] != True and \
            robotmap.bool_map[row, col - 1] != True:
            tmp = node_cost[row, col - 1] + cost
            heappush(minheap, (tmp, [tmp, row, col - 1]) )
            parent_map[ unique_id[row, col - 1] ] = unique_id[row, col]
        # down
        if row + 1 < robotmap.bool_map.shape[0] and \
            is_visited[row + 1, col] != True and \
            robotmap.bool_map[row + 1, col] != True:
            tmp = node_cost[row + 1, col] + cost
            heappush(minheap, (tmp, [tmp, row + 1, col]) )
            parent_map[ unique_id[row + 1, col] ] = unique_id[row, col]
        # right
        if col + 1 < robotmap.bool_map.shape[1] and \
            is_visited[row, col + 1] != True and \
            robotmap.bool_map[row, col + 1] != True:
            tmp = node_cost[row, col + 1] + cost
            heappush(minheap, (tmp, [tmp, row, col + 1]) )
            parent_map[ unique_id[row, col + 1] ] = unique_id[row, col]

    return ret, dfs_path


def main():
    robotmap = robot_map(40, 40, 0.09, 0.9)
    robotmap.generate_map()
    row, col = robotmap.get_start_position()
    target_row, target_col = robotmap.get_target_position()
    
    ret, dfs_path, res_path = a_star_search(robotmap, row, col, target_row, target_col)
    print('start position:', row, col)
    print('target position:', target_row, target_col)

    dfs_path = np.array(dfs_path)
    res_path = np.array(res_path)

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
                                    repeat_delay=100)
    plt.show()

if __name__ == '__main__':
    main()