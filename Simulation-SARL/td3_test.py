import math

import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import Environment
from td3 import Agent_TD3
import matplotlib.pyplot as plt

# ################## SETTINGS ######################
# RIS coordination
RIS_x, RIS_y, RIS_z = 250, 220, 25

# BS coordination
BS_x, BS_y, BS_z = 0, 0, 25

up_lanes = [i/2.0 for i in [400+3.5/2, 400+3.5+3.5/2, 800+3.5/2, 800+3.5+3.5/2]]
down_lanes = [i/2.0 for i in [400-3.5-3.5/2, 400-3.5/2, 800-3.5-3.5/2, 800-3.5/2]]
left_lanes = [i/2.0 for i in [400+3.5/2, 400+3.5+3.5/2, 800+3.5/2, 800+3.5+3.5/2]]
right_lanes = [i/2.0 for i in [400-3.5-3.5/2, 400-3.5/2, 800-3.5-3.5/2, 800-3.5/2]]

print('------------- lanes are -------------')
print('up_lanes :', up_lanes)
print('down_lanes :', down_lanes)
print('left_lanes :', left_lanes)
print('right_lanes :', right_lanes)
print('------------------------------------')

width = 800/2
height = 800/2

IS_TRAIN = 1

n_veh = 8
M = 36
control_bit = 3
data_buf_size = 10

###----------------环境设置，随机相移，RIS和BS之间的路程相移计算都是固定的，在初始化中直接计算------------###
env = Environment.Environ(down_lanes, up_lanes, left_lanes, right_lanes, width, height, n_veh, M, control_bit)

env.make_new_game()  # 添加车辆

n_episode = 50
n_step_per_episode = 100
theta_number = int(M / n_veh)

action_power = np.ones([2, n_veh], dtype=float) * 0.5
def get_phase_state():
    theta = env.elements_phase_shift_real / (2*math.pi)
    return np.reshape(theta, -1)
def get_state(idx):
    """ Get state from the environment """
    list = []
    position0 = env.vehicles[idx].position[0] / 300
    position1 = env.vehicles[idx].position[1] / 300

    Data_Buf = env.DataBuf[idx] / 10

    data_t = env.data_t[idx] / 10

    data_p = env.data_p[idx] / 10

    over_data = env.over_data[idx] / 10

    sinr = env.compute_sinr(idx, action_power) / 10000

    list.append(position0)
    list.append(position1)
    list.append(Data_Buf)
    list.append(data_t)
    list.append(data_p)
    list.append(over_data)
    list.append(sinr)

    return np.reshape(list, -1)

n_input = len(get_state(0)) * n_veh + len(get_phase_state())
n_output = 2 * n_veh + M

## Initializations ##
# ------- characteristics related to the network -------- #
batch_size = 64
memory_size = 1000000
gamma = 0.99
alpha = 0.0001
beta = 0.001
# ------------------------------
# actor and critic hidden layers
fc1_dims = 1024
fc2_dims = 512
fc3_dims = 256

tau = 0.005  # 参数更新权重
agent = Agent_TD3(alpha, beta, n_input, tau, gamma, n_output, memory_size, fc1_dims, fc2_dims, fc3_dims, batch_size, 2, 'OU')

#--------------------------------------------
agent.load_models()
##Let's go
if IS_TRAIN:
    record_reward_average = []
    cumulative_reward = 0
    Sum_DataBuf_length = []
    Sum_Power = []
    Sum_Power_local = []
    Sum_Power_offload = []
    user_power_episode = np.zeros((n_veh, n_episode))
    user_data_episode = np.zeros((n_veh, n_episode))
    Vehicle_positions_x0 = []
    Vehicle_positions_y0 = []
    Vehicle_positions_x1 = []
    Vehicle_positions_y1 = []
    Vehicle_positions_x2 = []
    Vehicle_positions_y2 = []
    Vehicle_positions_x3 = []
    Vehicle_positions_y3 = []
    for i_episode in range(n_episode):
        print("------------------------------step",i_episode, "---------------------------------------------")
        reward = np.zeros([n_step_per_episode], dtype=np.float16)
        #env.DataBuf = np.random.randint(0, data_buf_size - 1) / 2.0 * np.ones(n_veh)

        env.renew_positions()
        Vehicle_positions_x0.append(env.vehicles[0].position[0])
        Vehicle_positions_y0.append(env.vehicles[0].position[1])
        Vehicle_positions_x1.append(env.vehicles[1].position[0])
        Vehicle_positions_y1.append(env.vehicles[1].position[1])
        Vehicle_positions_x2.append(env.vehicles[2].position[0])
        Vehicle_positions_y2.append(env.vehicles[2].position[1])
        Vehicle_positions_x3.append(env.vehicles[3].position[0])
        Vehicle_positions_y3.append(env.vehicles[3].position[1])

        state1 = []
        state_pahse = np.reshape(get_phase_state(), -1)
        for i in range(n_veh):
            state = get_state(i)
            state1.append(state)
        state_old_all = np.concatenate(state1)
        state_old_all = np.concatenate((state_old_all, state_pahse))

        average_reward = 0
        Power = []
        Power_local = []
        Power_offload = []
        DataBuf_length = []
        user_power_step = np.zeros((n_veh, n_step_per_episode))
        user_data_step = np.zeros((n_veh, n_step_per_episode))
        for i_step in range(n_step_per_episode):
            state_new_all = []
            action_all = []
            action_all_training1 = np.zeros([2, n_veh], dtype=float) #power
            action_all_training2 = np.zeros(M)
            # receive observation
            action = agent.choose_action(np.asarray(state_old_all).flatten())
            action = np.clip(action, -0.999, 0.999)
            action_all.append(action)
            #All the agents take actions simultaneously
            for i in range(n_veh):
                action_all_training1[0, i] = ((action[i]+1)/2)
                action_all_training1[1, i] = ((action[n_veh+i]+1)/2)
            for m in range(M):
                action_all_training2[m] = math.floor(((action[m] + 1) / 2) * (2 ** control_bit)) * (
                            2 * math.pi / (2 ** control_bit))
            action_power = action_all_training1.copy()
            action_phase = action_all_training2.copy()

            train_reward, databuf, data_trans, data_local, over_power, over_data = env.step2(action_phase, action_power)

            reward[i_step] = train_reward
            Power.append(np.sum(action_power))
            Power_offload.append(np.sum(action_power[0]))
            Power_local.append(np.sum(action_power[1]))
            DataBuf_length.append(np.sum(databuf))

            for i in range(n_veh):
                user_power_step[i, i_step] = action_power[0, i] + action_power[1, i]
                user_data_step[i, i_step] = databuf[i]

            # get new state
            state2 = []
            state_pahse = np.reshape(get_phase_state(), -1)
            for i in range(n_veh):
                state_new = get_state(i)
                state2.append(state_new)
            state_new_all = np.concatenate(state2)
            state_new_all = np.concatenate((state_new_all, state_pahse))

            # old observation = new_observation
            state_old_all = state_new_all

        average_reward = np.mean(reward)
        record_reward_average.append(average_reward)
        Power_episode = np.mean(Power)
        Power_local_episode = np.mean(Power_local)
        Power_offload_episode = np.mean(Power_offload)
        DataBuf_length_episode = np.mean(DataBuf_length)
        for i in range(n_veh):
            user_power_episode[i, i_episode] = np.mean(user_power_step[i, :])
            user_data_episode[i, i_episode] = np.mean(user_data_step[i, :])

        Sum_Power.append(Power_episode)
        Sum_Power_local.append(Power_local_episode)
        Sum_Power_offload.append(Power_offload_episode)
        Sum_DataBuf_length.append(DataBuf_length_episode)
        print('Reward:', average_reward)
        print('Average Total Power:', Power_episode, end='   ')
        print('   Average local power:', Power_local_episode, '   Average offload power:',
              Power_offload_episode)

    print('Sum Average Power:', np.mean(Sum_Power), '   Sum average local power:', np.mean(Sum_Power_local), '      Sum Average offload power:',
          np.mean(Sum_Power_offload), '   Sum Average DataBuf:', np.mean(Sum_DataBuf_length))
    for i in range(n_veh):
        print('Average User Power:', i, ':', np.mean(user_power_episode[i, :]))
    for i in range(n_veh):
        print('Average User Data:', i, ':', np.mean(user_data_episode[i, :]))