'''
% -------------------------------------------------------------------------
%
% File : ExtendedKalmanFilterLocalization.m
%
% Discription : Mobible robot localization sample code with
% Extended Kalman Filter (EKF)
%
% Environment : Python
%
% Author : democheng
%
% Copyright (c): 2018 democheng
%
% License : MIT License
% -------------------------------------------------------------------------
'''
import math
import time
import sys
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
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
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
    plt.pause(0)
'''
the pipeline of Kalman Filter

prediction:
1. state prediction:
u_p_k = F_k * u_k-1
2. error prediction:
Sigma_p_k = F_k * Sigma_k * F_k^T + Q_k

update:
1. Kalman gain:
G_k = Sigma_p_k * H_k^T * (H_k * Sigma_p_k * H_k^T + R_k)^(-1)
2. observation residual:
delta_z_k = (z_p - H_k * u_p_k)
3. update state:
u_k = u_p_k + G_k * delta_z_k
4. error update:
Sigma_k = (I - G_k * H_k) * Sigma_p_k
'''
# checked
def generate_Control(time):
    '''
    :param time: current system time
    :return: control parameters
    '''
    # print('---generate_Control---')
    T = 10.0 # [sec]
    # [V yawrate]
    V = 1.0 # [m/s]
    yawrate = 5.0 # [deg/s]

    V = V * (1.0 - math.exp(-time/T))
    yawrate = math.radians(yawrate) * (1.0 - math.exp(-time/T))
    control = np.array([V, yawrate])
    control = control.reshape(len(control), 1)
    return control
#checked
def F_transition(state, control):
    '''
    :param state: previous updated state
    :param control: control parameters    
    :return: state of prediction
    '''
    state = state.reshape(len(state), 1)
    control = control.reshape(len(control), 1)
    global time_step
    F = np.eye(4)
    F[3, 3] = 0
    B = np.array([[time_step * math.cos(state[2]), 0.0], 
                [time_step * math.sin(state[2]), 0.0], 
                [0, time_step], 
                [1.0, 0.0]])
    state_prediction = F @ state + B @ control
    return state_prediction
#checked
def jacobian_F_transition(state, control):
    state = state.reshape(len(state), 1)
    control = control.reshape(len(control), 1)
    global time_step
    sin_yaw = math.sin(state[2])
    cos_yaw = math.cos(state[2])
    jF = np.array([[1.0, 0.0, 0.0, 0.0], 
                [0.0, 1.0, 0.0, 0.0], 
                [-time_step * control[0] * sin_yaw, time_step * control[0] * cos_yaw, 1.0, 0.0], 
                [time_step * cos_yaw, time_step * sin_yaw, 0.0, 1.0]])
    return jF

#checked
def H_observation(state):
    '''
    :param time: current system time
    :return: control parameters
    '''
    H = np.eye(4)
    observation = H @ state
    return observation
#checked
def jacobian_H_observation(state):
    '''
    :param state: state of prediction
    :return: jacobian_H_observation
    '''
    jH = np.eye(4)
    return jH

def observation_noise(groundtruth, dead_reckoning, control):
    '''
    :param groundtruth: groundtruth of prediction
    :param dead_reckoning: dead_reckoning of prediction
    :param control: control parameters
    :return: jacobian_H_observation
    '''
    groundtruth = groundtruth.reshape(len(groundtruth), 1)
    dead_reckoning = dead_reckoning.reshape(len(dead_reckoning), 1)
    control = control.reshape(len(control), 1)
    global Qsigma
    global Rsigma
    # groundtruth
    groundtruth = F_transition(groundtruth, control)
    # add process noise
    # test1 = np.ones((2, 1))
    # control = control + Qsigma @ test1
    control = control + Qsigma @ np.random.randn(2,1)
    # Dead Reckoning
    dead_reckoning = F_transition(dead_reckoning, control)
    # test2 = np.ones((4, 1))
    # observation = H_observation(groundtruth + Rsigma @ test2)
    observation = H_observation(groundtruth + Rsigma @ np.random.randn(4,1))
    return observation, groundtruth, dead_reckoning, control

def ExtendedKalmanFilterLocalization():
    print('Extended Kalman Filter (EKF) sample program start!!')
    
    # unit: s
    time_end = 60.0
    global time_step
    time_step = 0.1
    system_times = np.arange(time_step, time_end + time_step, time_step)
    # save all the variables for plotting
    result = {'time':[], 
            'groundtruth':[], 
            'dead_reckoning':[], 
            'observation':[], 
            'state':[], 
            'sigma':[],
            'control':[]}
    # x y yaw v
    state = np.array([0, 0, 0, 0])
    state.reshape(4, 1)
    groundtruth = state
    dead_reckoning = state
    observation = state
    
    # Covariance Matrix for motion
    # xx  0  0      0
    # 0   yy 0      0
    # 0   0  yawyaw 0
    # 0   0  0      vv
    Q = np.diag([0.1, 0.1, math.radians(1.0), 0.05]) ** 2

    # Covariance Matrix for observation
    # xx  0  0      0
    # 0   yy 0      0
    # 0   0  yawyaw 0
    # 0   0  0      vv
    R = np.diag([1.5, 1.5, math.radians(3.0), 0.05]) ** 2

    #------- Simulation parameter --------
    # vv  0
    # 0   yawrateyawrate
    global Qsigma
    Qsigma = np.diag([0.1, math.radians(20.0)]) ** 2
    # xx  0  0      0
    # 0   yy 0      0
    # 0   0  yawyaw 0
    # 0   0  0      vv
    global Rsigma
    Rsigma = np.diag([1.5, 1.5, math.radians(3.0), 0.05]) ** 2

    # Covariance Matrix for state
    sigma = np.eye(4)
    
    time_loop_start = time.time()
    time_loop_used = time.time() - time_loop_start

    for cur_time in system_times:
        control = generate_Control(cur_time)
        observation, groundtruth, dead_reckoning, control = observation_noise(groundtruth, dead_reckoning, control)
        # ---------EKF----------
        # Predict
        # 1. state prediction:
        state_prediction = F_transition(state, control)
        # 2. error prediction:
        jF = jacobian_F_transition(state_prediction, control)
        sigma_prediction = jF @ sigma @ jF.T + Q
        # update
        # 1. Kalman gain:
        jH = jacobian_H_observation(state_prediction)
        kalman_gain = sigma_prediction @ jH.T @ np.linalg.inv(jH @ sigma_prediction @ jH.T + R)
        # 2. observation residual:
        delta_observation = observation - H_observation(state_prediction)
        # 3. update state:
        state = state_prediction.reshape(4, 1) + kalman_gain @ delta_observation
        # 4. error update:
        sigma = (np.eye(len(state)) - kalman_gain @ jH) @ sigma_prediction
        # ----------------------

        # save all the variables for plotting
        result['time'].append(cur_time)
        result['groundtruth'].append(groundtruth)
        result['dead_reckoning'].append(dead_reckoning)
        result['observation'].append(observation)
        result['state'].append(state)
        result['sigma'].append(sigma)
        result['control'].append(control)

    print('Time loop used:', time_loop_used)
    return result

def main():
    result = ExtendedKalmanFilterLocalization()
    draw_result(result)

if __name__ == '__main__':
    main()