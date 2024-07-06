import numpy as np
import random
import math
import cmath

np.random.seed(1234)

# RIS coordination
RIS_x, RIS_y, RIS_z = 250, 220, 25

# BS coordination
BS_x, BS_y, BS_z = 0, 0, 25

ro = 10 ** -2  # 参考距离d0 = 1m处的平均路径损耗功率增益 10dBm = 0.01w

cascaded_gain = 0
lamb = 0.15  # 载波长度
d = 0.02  # RIS元素之间的距离

sigma = 10 ** (-7)
alpha1 = 2.5  # 公式中的alpha
alpha2 = 2.5
c = 3*(10**8)
class Vehicle:
    """Vehicle simulator: include all the information for a Vehicle"""

    def __init__(self, start_position, start_direction, velocity):
        self.position = start_position
        self.direction = start_direction
        self.velocity = velocity
        self.neighbors = []
        self.destinations = []

class Environ:
    def __init__(self, down_lane, up_lane, left_lane, right_lane, width, height, n_veh, M, control_bit):

        self.down_lanes = down_lane
        self.up_lanes = up_lane
        self.left_lanes = left_lane
        self.right_lanes = right_lane
        self.width = width
        self.height = height

        self.n_veh = n_veh

        self.Decorrelation_distance = 10
        self.V2I_Shadowing = np.zeros(n_veh)
        self.V2I_pathloss = np.zeros(n_veh)
        self.V2I_channels_abs = np.zeros(n_veh)
        self.delta_distance = []
        self.sig2_dB = -110
        self.sig2 = 10 ** (self.sig2_dB / 10)

        self.bsAntGain = 8  # 基站天线增益
        self.bsNoiseFigure = 5  # 基站接收器噪声系数
        self.vehAntGain = 3  # 车辆天线增益
        self.vehNoiseFigure = 9  # 车辆接收器噪声增益

        self.vehicles = []

        self.time_slow = 0.1
        self.time_fast = 0.001
        self.bandwidth = 1#MHz
        self.fc = 2

        ###--------------RIS-------------###
        self.sinr = np.zeros(n_veh)

        self.phases_R_i = np.zeros([n_veh, M], dtype=complex)  # RIS-vehicle

        self.distances_R_i = np.zeros(n_veh)
        self.angles_R_i = np.zeros(n_veh)

        self.M = M  # 元素数量
        self.control_bit = control_bit  # 元素相移控制比特
        self.possible_angles = np.linspace(0, 2 * math.pi, 2 ** self.control_bit, endpoint=False)

        self.elements_phase_shift_complex = np.zeros(self.M, dtype=complex)  # 复数形式 RIS元素的相移，这个是后面要强化学习的动作
        self.Random_phase()
        self.phase_R = np.zeros(self.M, dtype=complex)  # RIS到BS
        # self.phases_R_i = np.zeros([self.number_of_vehicles, self.M], dtype=complex) #车辆到RIS

        self.distance_B_R = math.sqrt(
            (BS_x - RIS_x) ** 2 + (BS_y - RIS_y) ** 2 + (BS_z - RIS_z) ** 2)
        self.angle_B_R = (RIS_x - BS_x) / self.distance_B_R  # 从RIS到BS的角度
        for m in range(self.M):
            self.phase_R[m] = cmath.exp(2 * (math.pi / lamb) * d * self.angle_B_R * m * 1j)

        self.elements_phase_shift_real = np.zeros(M)
        self.compute_parms()

    def Random_phase(self):
        self.elements_phase_shift_real = [random.choice(self.possible_angles) for x1 in range(self.M)] #随机产生RIS的相移
        for m in range(self.M):
            self.elements_phase_shift_complex[m] = np.exp(self.elements_phase_shift_real[m] * 1.0j)

    def get_path_loss(self, position_A):
        d1 = abs(position_A[0] - BS_x)
        d2 = abs(position_A[1] - BS_y)
        distance = math.hypot(d1, d2)
        return 128.1 + 37.6 * np.log10(math.sqrt(distance ** 2 + (BS_z - 1.5) ** 2) / 1000)

    def get_shadowing(self, delta_distance, vehicle):
        self.R = np.sqrt(0.5 * np.ones([1, 1]) + 0.5 * np.identity(1))
        return np.multiply(np.exp(-1 * (delta_distance / self.Decorrelation_distance)), self.V2I_Shadowing[vehicle]) \
               + np.sqrt(1 - np.exp(-2 * (delta_distance / self.Decorrelation_distance))) * np.random.normal(0, 8, 1)

    def get_C_RIS_path_loss(self, position_A, theta):
        """Calculate RIS pathloss between cellular users and BS"""
        theta_diag = np.diag(theta)
        ds = 0.02 # The spacing between elements
        dA1 = abs(position_A[0] - RIS_x)
        dA2 = abs(position_A[1] - RIS_y)
        dB1 = abs(BS_x - RIS_x)
        dB2 = abs(BS_y - RIS_y)
        dA = math.hypot(dA1, dA2)
        dB = math.hypot(dB1, dB2)
        d_bp = 4 * (BS_z - 1) * (1.5 - 1) * self.fc * (10 ** 9) / (3 * 10 ** 8)
        a_aoa = np.zeros(self.M, dtype=complex)
        a_aod = np.zeros(self.M, dtype=complex)
        if (position_A[0] - RIS_x) == 0:
            theta_aoa = -np.pi/2
        else:
            theta_aoa = np.arctan((position_A[1] - RIS_y) / (position_A[0] - RIS_x))
        theta_aod = np.arctan((BS_y - RIS_y)/(BS_x - RIS_x))

        def PL_Los(d):
            if d <= 3:
                return 22.7 * np.log10(3) + 41 + 20 * np.log10(self.fc / 5)
            else:
                if d < d_bp:
                    return 22.7 * np.log10(d) + 41 + 20 * np.log10(self.fc / 5)
                else:
                    return 40.0 * np.log10(d) + 9.45 - 17.3 * np.log10(BS_z) - 17.3 * np.log10(
                        1.5) + 2.7 * np.log10(self.fc / 5)

        def PL_NLos(d_a, d_b):
            n_j = max(2.8 - 0.0024 * d_b, 1.84)
            return PL_Los(d_a) + 20 - 12.5 * n_j + 10 * n_j * np.log10(d_b) + 3 * np.log10(self.fc / 5)

        PLA = 128.1 + 37.6 * np.log10(math.sqrt(dA ** 2 + (BS_z - 1.5) ** 2) / 1000)
        PLB = 23.5* np.log10(math.sqrt(dB ** 2 + (BS_z - RIS_z) ** 2) / 1000) + 57.5 + 23 * np.log10(2 / 5)

        for n in range (self.M):
            a_aod[n] = np.exp(-1.0j*(2*np.pi)*ds*(self.fc/0.3)*n*np.sin(theta_aod))
            a_aoa[n] = np.exp(-1.0j*(2*np.pi)*ds*(self.fc/0.3)*n*np.sin(theta_aoa))

        ChannelA = 1/np.power(10, PLA/10) * np.exp(-1.0j*(2*np.pi)*dA*(self.fc/0.3))*a_aoa
        ChannelB = 1/np.power(10, PLB/10) * np.exp(-1.0j*(2*np.pi)*dB*(self.fc/0.3))*a_aod.conj().T

        PL_RIS_sig = np.dot(np.dot(ChannelA,  theta_diag), ChannelB)
        PL_RIS = np.log10((1 / (PL_RIS_sig))) * 10
        return PL_RIS  # + self.shadow_std * np.random.normal()

    def get_next_phase(self, action_phase):
        self.elements_phase_shift_real = action_phase
        for m in range(self.M):
            self.elements_phase_shift_complex[m] = cmath.exp(self.elements_phase_shift_real[m] * 1j)


    def compute_parms(self):
        # Calculate vehicle to RIS distance and angles
        for vehicle in range(len(self.vehicles)):
            d_R_i = math.sqrt((self.vehicles[vehicle].position[0] - RIS_x) ** 2 + (self.vehicles[vehicle].position[1] - RIS_y) ** 2 + (1.5 - RIS_z) ** 2)
            self.distances_R_i[vehicle] = d_R_i
            self.angles_R_i[vehicle] = ((self.vehicles[vehicle].position[0] - RIS_x) / d_R_i)

        #Calculate phase shift with vehicles
        for m in range(len(self.elements_phase_shift_real)):
            for vehicle in range(len(self.vehicles)):
                self.phases_R_i[vehicle][m] = cmath.exp(-2 * (math.pi / lamb) * d * self.angles_R_i[vehicle] * m * 1j)

    # 计算一个车辆的信息传输速率, power是功率分配的动作
    #power是一个2 * n_veh的数组，第一行是卸载的功率，第二行是本地执行功率
    def compute_data_rate(self, vehicle, power):
        # 多普勒效应
        f_d = (self.fc * self.vehicles[vehicle].velocity) / c
        Doppler = 10 * alpha1 * np.log10(1 + np.power((f_d / self.fc), 2))
        power_dbm = 10 * math.log10(power* 1000)
        # 间接链路
        V2I_RIS_Pathloss = np.abs(
            self.get_C_RIS_path_loss(self.vehicles[vehicle].position, self.elements_phase_shift_complex))
        V2I_RIS_signal = 10 ** ((power_dbm - V2I_RIS_Pathloss - Doppler) / 10)
        rate_ris = np.log2(1 + np.divide(V2I_RIS_signal, self.sig2))

        return rate_ris

    def compute_sinr(self, vehicle, power):
        # 间接链路
        f_d = (self.fc * self.vehicles[vehicle].velocity) / c
        Doppler = 10 * alpha1 * np.log10(1 + np.power((f_d / self.fc), 2))
        power_dbm = 10 * math.log10(power * 1000)
        V2I_RIS_Pathloss = np.abs(
            self.get_C_RIS_path_loss(self.vehicles[vehicle].position, self.elements_phase_shift_complex))
        V2I_RIS_signal = 10 ** ((power_dbm - V2I_RIS_Pathloss - Doppler) / 10)
        sinr = np.divide(V2I_RIS_signal, self.sig2)

        return sinr

    def add_new_vehicles(self, start_position, start_direction, start_velocity):
        self.vehicles.append(Vehicle(start_position, start_direction, start_velocity))

    def add_new_vehicles_by_number(self, n):
        string = 'dulr'
        for i in range(n):
            ind = np.random.randint(0, len(self.down_lanes))

            start_position = [np.random.randint(280, 290), self.left_lanes[0]]
            start_direction = 'l'
            self.add_new_vehicles(start_position, start_direction, np.random.randint(10, 15))

            start_position = [np.random.randint(280, 290), self.left_lanes[0]]
            start_direction = 'l'
            self.add_new_vehicles(start_position, start_direction, np.random.randint(10, 15))

            start_position = [np.random.randint(280, 290), self.left_lanes[1]]
            start_direction = 'l'
            self.add_new_vehicles(start_position, start_direction, np.random.randint(10, 15))

            start_position = [np.random.randint(280, 290), self.left_lanes[1]]
            start_direction = 'l'
            self.add_new_vehicles(start_position, start_direction, np.random.randint(10, 15))

            start_position = [np.random.randint(290, 300), self.left_lanes[0]]
            start_direction = 'l'
            self.add_new_vehicles(start_position, start_direction, np.random.randint(10, 15))

            start_position = [np.random.randint(290, 300), self.left_lanes[0]]
            start_direction = 'l'
            self.add_new_vehicles(start_position, start_direction, np.random.randint(10, 15))

            start_position = [np.random.randint(290, 300), self.left_lanes[1]]
            start_direction = 'l'
            self.add_new_vehicles(start_position, start_direction, np.random.randint(10, 15))

            start_position = [np.random.randint(290, 300), self.left_lanes[1]]
            start_direction = 'l'
            self.add_new_vehicles(start_position, start_direction, np.random.randint(10, 15))

        for j in range(int(self.n_veh % 4)):  # 当车辆数不是4的倍数时，按照这个添加车辆
            ind = np.random.randint(0, len(self.down_lanes))
            str = random.choice(string)
            start_position = [self.down_lanes[ind], np.random.randint(0, self.height)]
            start_direction = str  # velocity: 10 ~ 15 m/s, random
            self.add_new_vehicles(start_position, start_direction, np.random.randint(15, 20))

        self.V2I_Shadowing = np.random.normal(0, 8, len(self.vehicles))
        self.delta_distance = np.asarray([c.velocity * self.time_slow for c in self.vehicles])

    def renew_positions(self):
        # ===============
        # This function updates the position of each vehicle
        # ===============

        i = 0
        while (i < len(self.vehicles)):
            delta_distance = self.vehicles[i].velocity * self.time_slow
            change_direction = False
            if self.vehicles[i].direction == 'u':
                # print ('len of position', len(self.position), i)
                for j in range(len(self.left_lanes)):

                    if (self.vehicles[i].position[1] <= self.left_lanes[j]) and ((self.vehicles[i].position[1] + delta_distance) >= self.left_lanes[j]):  # came to an cross
                        if (np.random.uniform(0, 1) < 0.4):
                            self.vehicles[i].position = [self.vehicles[i].position[0] - (delta_distance - (self.left_lanes[j] - self.vehicles[i].position[1])), self.left_lanes[j]]
                            self.vehicles[i].direction = 'l'
                            change_direction = True
                            break
                if change_direction == False:
                    for j in range(len(self.right_lanes)):
                        if (self.vehicles[i].position[1] <= self.right_lanes[j]) and ((self.vehicles[i].position[1] + delta_distance) >= self.right_lanes[j]):
                            if (np.random.uniform(0, 1) < 0.4):
                                self.vehicles[i].position = [self.vehicles[i].position[0] + (delta_distance + (self.right_lanes[j] - self.vehicles[i].position[1])), self.right_lanes[j]]
                                self.vehicles[i].direction = 'r'
                                change_direction = True
                                break
                if change_direction == False:
                    self.vehicles[i].position[1] += delta_distance
            if (self.vehicles[i].direction == 'd') and (change_direction == False):
                # print ('len of position', len(self.position), i)
                for j in range(len(self.left_lanes)):
                    if (self.vehicles[i].position[1] >= self.left_lanes[j]) and ((self.vehicles[i].position[1] - delta_distance) <= self.left_lanes[j]):  # came to an cross
                        if (np.random.uniform(0, 1) < 0.4):
                            self.vehicles[i].position = [self.vehicles[i].position[0] - (delta_distance - (self.vehicles[i].position[1] - self.left_lanes[j])), self.left_lanes[j]]
                            # print ('down with left', self.vehicles[i].position)
                            self.vehicles[i].direction = 'l'
                            change_direction = True
                            break
                if change_direction == False:
                    for j in range(len(self.right_lanes)):
                        if (self.vehicles[i].position[1] >= self.right_lanes[j]) and (self.vehicles[i].position[1] - delta_distance <= self.right_lanes[j]):
                            if (np.random.uniform(0, 1) < 0.4):
                                self.vehicles[i].position = [self.vehicles[i].position[0] + (delta_distance + (self.vehicles[i].position[1] - self.right_lanes[j])), self.right_lanes[j]]
                                # print ('down with right', self.vehicles[i].position)
                                self.vehicles[i].direction = 'r'
                                change_direction = True
                                break
                if change_direction == False:
                    self.vehicles[i].position[1] -= delta_distance
            if (self.vehicles[i].direction == 'r') and (change_direction == False):
                # print ('len of position', len(self.position), i)
                for j in range(len(self.up_lanes)):
                    if (self.vehicles[i].position[0] <= self.up_lanes[j]) and ((self.vehicles[i].position[0] + delta_distance) >= self.up_lanes[j]):  # came to an cross
                        if (np.random.uniform(0, 1) < 0.4):
                            self.vehicles[i].position = [self.up_lanes[j], self.vehicles[i].position[1] + (delta_distance - (self.up_lanes[j] - self.vehicles[i].position[0]))]
                            change_direction = True
                            self.vehicles[i].direction = 'u'
                            break
                if change_direction == False:
                    for j in range(len(self.down_lanes)):
                        if (self.vehicles[i].position[0] <= self.down_lanes[j]) and ((self.vehicles[i].position[0] + delta_distance) >= self.down_lanes[j]):
                            if (np.random.uniform(0, 1) < 0.4):
                                self.vehicles[i].position = [self.down_lanes[j], self.vehicles[i].position[1] - (delta_distance - (self.down_lanes[j] - self.vehicles[i].position[0]))]
                                change_direction = True
                                self.vehicles[i].direction = 'd'
                                break
                if change_direction == False:
                    self.vehicles[i].position[0] += delta_distance
            if (self.vehicles[i].direction == 'l') and (change_direction == False):
                for j in range(len(self.up_lanes)):

                    if (self.vehicles[i].position[0] >= self.up_lanes[j]) and ((self.vehicles[i].position[0] - delta_distance) <= self.up_lanes[j]):  # came to an cross
                        if (np.random.uniform(0, 1) < 0.4):
                            self.vehicles[i].position = [self.up_lanes[j], self.vehicles[i].position[1] + (delta_distance - (self.vehicles[i].position[0] - self.up_lanes[j]))]
                            change_direction = True
                            self.vehicles[i].direction = 'u'
                            break
                if change_direction == False:
                    for j in range(len(self.down_lanes)):
                        if (self.vehicles[i].position[0] >= self.down_lanes[j]) and ((self.vehicles[i].position[0] - delta_distance) <= self.down_lanes[j]):
                            if (np.random.uniform(0, 1) < 0.4):
                                self.vehicles[i].position = [self.down_lanes[j], self.vehicles[i].position[1] - (delta_distance - (self.vehicles[i].position[0] - self.down_lanes[j]))]
                                change_direction = True
                                self.vehicles[i].direction = 'd'
                                break
                    if change_direction == False:
                        self.vehicles[i].position[0] -= delta_distance

            # if it comes to an exit
            if (self.vehicles[i].position[0] < 0) or (self.vehicles[i].position[1] < 0) or (self.vehicles[i].position[0] > self.width) or (self.vehicles[i].position[1] > self.height):
                # delete
                #    print ('delete ', self.position[i])
                if (self.vehicles[i].direction == 'u'):
                    self.vehicles[i].direction = 'r'
                    self.vehicles[i].position = [self.vehicles[i].position[0], self.right_lanes[-1]]
                else:
                    if (self.vehicles[i].direction == 'd'):
                        self.vehicles[i].direction = 'l'
                        self.vehicles[i].position = [self.vehicles[i].position[0], self.left_lanes[0]]
                    else:
                        if (self.vehicles[i].direction == 'l'):
                            self.vehicles[i].direction = 'u'
                            self.vehicles[i].position = [self.up_lanes[0], self.vehicles[i].position[1]]
                        else:
                            if (self.vehicles[i].direction == 'r'):
                                self.vehicles[i].direction = 'd'
                                self.vehicles[i].position = [self.down_lanes[-1], self.vehicles[i].position[1]]

            i += 1

    def cen_step(self, action_phase):
        self.get_next_phase(action_phase)
        sum_rate = []
        for i in range(self.n_veh):
            snr = self.compute_data_rate(i, 1)
            sum_rate.append(snr)

        reward_rate = np.mean(sum_rate)

        return reward_rate

    def make_new_game(self):
        self.vehicles = []
        self.add_new_vehicles_by_number(int(self.n_veh / 8))
