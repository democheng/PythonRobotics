import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3

def draw_result(result):
        
    ## Attaching 3D axis to the figure
    fig = plt.figure()
    plt.ion()
    ax = p3.Axes3D(fig)
    ax.axis('equal')

    ## Setting the axes properties
    # ax.set_xlim3d([0.0, 10.0])
    ax.set_xlabel('X')
    # ax.set_ylim3d([0.0, 10.0])
    ax.set_ylabel('Y')
    # ax.set_zlim3d([0.0, 10.0])
    # ax.set_zlabel('Z')
    ax.set_title('trajectory')
    # result = {'time':[], 
    #     'groundtruth':[], 
    #     'dead_reckoning':[], 
    #     'observation':[], 
    #     'state':[], 
    #     'sigma':[],
    #     'control':[]}
    groundtruth_position = np.zeros((len(result['groundtruth']), 2))
    for index in range(len(result['groundtruth'])):
        groundtruth_position[index, 0] = result['groundtruth'][index][0]
        groundtruth_position[index, 1] = result['groundtruth'][index][1]
    
    dead_reckoning_position = np.zeros((len(result['dead_reckoning']), 2))
    for index in range(len(result['dead_reckoning'])):
        dead_reckoning_position[index, 0] = result['dead_reckoning'][index][0]
        dead_reckoning_position[index, 1] = result['dead_reckoning'][index][1]

    state_position = np.zeros((len(result['state']), 2))
    for index in range(len(result['state'])):
        state_position[index, 0] = result['state'][index][0]
        state_position[index, 1] = result['state'][index][1]

    ax.plot(groundtruth_position[:, 0], groundtruth_position[:, 1], color = 'r', label='groundtruth')
    ax.plot(dead_reckoning_position[:, 0], dead_reckoning_position[:, 1], color = 'g', label='dead_reckoning')
    ax.plot(state_position[:, 0], state_position[:, 1], color = 'b', label='EKF')
    ax.legend()
    plt.show()
    input("Press Enter to continue...")