import math
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import numpy as np
import Environment
from ddpg_torch import Agent
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

n_episode = 1000
n_step_per_episode = 100

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

    sinr = env.compute_sinr(idx, action_power) / 20000

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

# ------- characteristics related to the network -------- #
batch_size = 64
memory_size = 1000000
gamma = 0.99
alpha = 0.0001
beta = 0.001
# actor and critic hidden layers
C_fc1_dims = 1024
C_fc2_dims = 512
C_fc3_dims = 256

A_fc1_dims = 512
A_fc2_dims = 256
# ------------------------------

tau = 0.005

# --------------------------------------------------------------
agent = Agent(alpha, beta, n_input, tau, n_output, gamma, memory_size, C_fc1_dims, C_fc2_dims, C_fc3_dims,
                  A_fc1_dims, A_fc2_dims, batch_size, n_veh)

##Let's go
if IS_TRAIN:
    record_reward_average = []
    cumulative_reward = 0
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
    for i_episode in range(n_episode):
        done = 0
        print("------------------------------------Step:", i_episode, "---------------------------------------")
        reward = np.zeros([n_step_per_episode], dtype=np.float16)
        #env.DataBuf = np.random.randint(0, data_buf_size - 1) / 2.0 * np.ones(n_veh)

        if i_episode % 20 == 0:
            env.renew_positions()
            #env.Random_phase()
            env.compute_parms()
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
                action_all_training2[m] = ((action[2 * n_veh + m]+1)/2) * math.pi * 2
            action_power = action_all_training1.copy()
            action_phase = action_all_training2.copy()

            train_reward, databuf, data_trans, data_local, over_power, over_data = env.step2(action_phase, action_power)

            reward[i_step] = train_reward
            Power.append(np.sum(action_power))
            Power_offload.append(np.sum(action_power[0]))
            Power_local.append(np.sum(action_power[1]))

            #get new state
            state2 = []
            state_pahse = np.reshape(get_phase_state(), -1)
            for i in range(n_veh):
                state_new = get_state(i)
                state2.append(state_new)
            state_new_all = np.concatenate(state2)
            state_new_all = np.concatenate((state_new_all, state_pahse))

            if i_step == n_step_per_episode - 1:
                done = True

            # taking the agents actions, states and reward
            agent.remember(np.asarray(state_old_all).flatten(), np.asarray(action_all).flatten(),
                            train_reward, np.asarray(state_new_all).flatten(), done)

            # agents take random samples and learn
            agent.learn()

            # old observation = new_observation
            state_old_all = state_new_all

        average_reward = np.mean(reward)
        record_reward_average.append(average_reward)
        Power_episode = np.mean(Power)
        Power_local_episode = np.mean(Power_local)
        Power_offload_episode = np.mean(Power_offload)

        Sum_Power.append(Power_episode)
        Sum_Power_local.append(Power_local_episode)
        Sum_Power_offload.append(Power_offload_episode)
        print('Reward:', average_reward)
        print('Average Total Power:', Power_episode)
        print('Average local power:', Power_local_episode, '   Average offload power:',
              Power_offload_episode)

        if (i_episode + 1) % 50 == 0 and i_episode != 0:
            agent.save_models()

    # 记录数据，绘制奖励函数曲线
    print('Sum average local power:', np.mean(Sum_Power_local), '      Sum Average offload power:',
          np.mean(Sum_Power_offload))
    np.save('Data/3-Reward_DDPG_1000.npy', record_reward_average)
    plt.figure(1)
    # fig = plt.figure(figsize=(width/100, height/100))
    plt.plot(BS_x, BS_y, 'o', markersize=5, color='black', label='BS')
    plt.plot(RIS_x, RIS_y, 'o', markersize=5, color='brown', label='RIS')
    plt.plot(Vehicle_positions_x0, Vehicle_positions_y0)
    plt.plot(Vehicle_positions_x1, Vehicle_positions_y1)
    plt.plot(Vehicle_positions_x2, Vehicle_positions_y2)
    plt.plot(Vehicle_positions_x3, Vehicle_positions_y3)

    plt.figure(2)
    x = np.linspace(0, n_episode - 1, n_episode, dtype=int)
    y1 = record_reward_average
    plt.plot(x, y1)
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    plt.show()

    print('Training Done. Saving models...')
