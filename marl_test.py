import math
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import numpy as np
import matplotlib.pyplot as plt
import Environment
from ddpg_torch import Agent
from buffer import ReplayBuffer
from global_critic import Global_Critic


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

###----------------环境设置，随机相移，RIS和BS之间的路程相移计算都是固定的，在初始化中直接计算------------###
env = Environment.Environ(down_lanes, up_lanes, left_lanes, right_lanes, width, height, n_veh, M, control_bit)

env.make_new_game()  # 添加车辆

n_episode_test = 50
n_step_per_episode = 100

def marl_get_state(idx):
    """ Get state from the environment """
    list = []

    Data_Buf = env.DataBuf[idx] / 10

    data_t = env.data_t[idx] / 10

    data_p = env.data_p[idx] / 10

    over_data = env.over_data[idx] / 10

    rate = env.vehicle_rate[idx] / 20

    #position = env.vehicles[idx].position

    #list.append(Data_arrive)
    list.append(Data_Buf)
    list.append(data_t)
    list.append(data_p)
    list.append(over_data)
    list.append(rate)

    return list

    #return np.concatenate((np.reshape(theta, -1), np.reshape(list, -1)))

marl_n_input = len(marl_get_state(0))
marl_n_output = 2  #卸载功率和本地执行功率

##---Initializations networks parameters---##
batch_size = 64
memory_size = 1000000
gamma = 0.99
alpha = 0.0001
beta = 0.001
update_actor_interval = 2
noise = 0.2
# actor and critic hidden layers
C_fc1_dims = 1024
C_fc2_dims = 512
C_fc3_dims = 256

A_fc1_dims = 512
A_fc2_dims = 256
# ------------------------------

tau = 0.005
#--------------------------------------------


agents = []
for index_agent in range(n_veh):
    print("Initializing agent", index_agent)
    agent = Agent(alpha, beta, marl_n_input, tau, marl_n_output, gamma, C_fc1_dims, C_fc2_dims, C_fc3_dims,
                  A_fc1_dims, A_fc2_dims, batch_size, n_veh, index_agent, noise)
    agents.append(agent)
memory = ReplayBuffer(memory_size, marl_n_input, marl_n_output, n_veh)
print("Initializing Global critic ...")
global_agent = Global_Critic(beta, marl_n_input, tau, marl_n_output, gamma, C_fc1_dims, C_fc2_dims, C_fc3_dims,
                 batch_size, n_veh, update_actor_interval, noise)

global_agent.load_models()
for i in range(n_veh):
    agents[i].load_models()

##Let's go
if IS_TRAIN:
    record_reward_ = np.zeros([n_veh, n_episode_test])
    record_global_reward_average = []
    Sum_DataBuf_length = []
    Sum_Power = []
    Sum_Power_local = []
    Sum_Power_offload = []
    Vehicle_positions_x0 = []
    Vehicle_positions_y0 = []
    Vehicle_positions_x1 = []
    Vehicle_positions_y1 = []
    Vehicle_positions_x2 = []
    Vehicle_positions_y2 = []
    Vehicle_positions_x3 = []
    Vehicle_positions_y3 = []
    Vehicle_positions_x4 = []
    Vehicle_positions_y4 = []
    Vehicle_positions_x5 = []
    Vehicle_positions_y5 = []
    Vehicle_positions_x6 = []
    Vehicle_positions_y6 = []
    Vehicle_positions_x7 = []
    Vehicle_positions_y7 = []
    for i_episode in range(n_episode_test):
        done = 0
        print("-------------------------------- step:", i_episode, "-------------------------------------------")
        record_reward = np.zeros([n_veh, n_step_per_episode], dtype=np.float16)
        record_global_reward = np.zeros(n_step_per_episode)
        DataBuf_length = []
        Power = []
        Power_local = []
        Power_offload = []

        env.renew_positions()
        #env.compute_parms()
        Vehicle_positions_x0.append(env.vehicles[0].position[0])
        Vehicle_positions_y0.append(env.vehicles[0].position[1])
        Vehicle_positions_x1.append(env.vehicles[1].position[0])
        Vehicle_positions_y1.append(env.vehicles[1].position[1])
        Vehicle_positions_x2.append(env.vehicles[2].position[0])
        Vehicle_positions_y2.append(env.vehicles[2].position[1])
        Vehicle_positions_x3.append(env.vehicles[3].position[0])
        Vehicle_positions_y3.append(env.vehicles[3].position[1])
        Vehicle_positions_x4.append(env.vehicles[4].position[0])
        Vehicle_positions_y4.append(env.vehicles[4].position[1])
        Vehicle_positions_x5.append(env.vehicles[5].position[0])
        Vehicle_positions_y5.append(env.vehicles[5].position[1])
        Vehicle_positions_x6.append(env.vehicles[6].position[0])
        Vehicle_positions_y6.append(env.vehicles[6].position[1])
        Vehicle_positions_x7.append(env.vehicles[7].position[0])
        Vehicle_positions_y7.append(env.vehicles[7].position[1])


        marl_state_old_all = []

        for i in range(n_veh):
            marl_state = marl_get_state(i)
            marl_state_old_all.append(marl_state)

        average_global_reward = 0
        average_reward = np.zeros(n_veh)
        for i_step in range(n_step_per_episode):
            marl_state_new_all = []
            sarl_state_new_all = []
            marl_action_all = []
            action_all_training1 = np.zeros([marl_n_output, n_veh], dtype=float) #power
            for i in range(n_veh):
                marl_action = agents[i].choose_action(marl_state_old_all[i])
                marl_action = np.clip(marl_action, -0.999, 0.999)
                marl_action_all.append(marl_action)

                action_all_training1[0, i] = ((marl_action[0] + 1) / 2) #功率范围设置为0~1
                action_all_training1[1, i] = ((marl_action[1] + 1) / 2)

            #All the agents take actions simultaneously
            action_power = action_all_training1.copy()

            per_user_reward, global_reward, databuf, data_trans, data_local,\
            over_power, over_data = env.step1(action_power)
            Power.append(np.sum(action_power))
            Power_offload.append(np.sum(action_power[0]))
            Power_local.append(np.sum(action_power[1]))
            DataBuf_length.append(np.sum(databuf))

            record_global_reward[i_step] = global_reward
            for i in range(n_veh):
                record_reward[i, i_step] = per_user_reward[i]

            #get new state
            for i in range(n_veh):
                marl_state_new = marl_get_state(i)
                marl_state_new_all.append(marl_state_new)

            for i in range(n_veh):
                marl_state_old_all[i] = marl_state_new_all[i]

        for i in range(n_veh):
            record_reward_[i, i_episode] = np.mean(record_reward[i])
            print('user', i, record_reward_[i, i_episode], end='      ')

        #输出全局奖励
        average_global_reward = np.mean(record_global_reward)
        record_global_reward_average.append(average_global_reward)

        Power_episode = np.mean(Power)
        Power_local_episode = np.mean(Power_local)
        Power_offload_episode = np.mean(Power_offload)
        DataBuf_length_episode = np.mean(DataBuf_length)

        Sum_Power.append(Power_episode)
        Sum_Power_local.append(Power_local_episode)
        Sum_Power_offload.append(Power_offload_episode)
        Sum_DataBuf_length.append(DataBuf_length_episode)
        print('Global reward:', average_global_reward)
        print('Average Total Power:', Power_episode)
        print('Average local power:', Power_local_episode, '   Average offload power:',
              Power_offload_episode)

    print('------------BCD-MADDPG-------------')
    for i in range(n_veh):
        print('Average User Reward:', i, ':', np.mean(record_reward_[i, :]))
    print('Average Global Reward:', np.mean(record_global_reward_average), ''
        '   Sum Average Power:', np.mean(Sum_Power), '   Sum Average DataBuf:', np.mean(Sum_DataBuf_length))
    print('Sum average local power:', np.mean(Sum_Power_local), '   Sum Average offload power:', np.mean(Sum_Power_offload))

    plt.figure(1)
    # fig = plt.figure(figsize=(width/100, height/100))
    plt.plot(BS_x, BS_y, 'o', markersize=5, color='black', label='BS')
    plt.plot(RIS_x, RIS_y, 'o', markersize=5, color='brown', label='RIS')
    plt.plot(Vehicle_positions_x0, Vehicle_positions_y0)
    plt.plot(Vehicle_positions_x1, Vehicle_positions_y1)
    plt.plot(Vehicle_positions_x2, Vehicle_positions_y2)
    plt.plot(Vehicle_positions_x3, Vehicle_positions_y3)
    plt.plot(Vehicle_positions_x4, Vehicle_positions_y4)
    plt.plot(Vehicle_positions_x5, Vehicle_positions_y5)
    plt.plot(Vehicle_positions_x6, Vehicle_positions_y6)
    plt.plot(Vehicle_positions_x7, Vehicle_positions_y7)

    plt.show()