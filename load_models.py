import numpy as np
import Environment
from ddpg import Agent
from sac import SAC_Trainer
from sac import ReplayBuffer
label = 'model/3-ris_sac_model'
model_path = label + '/agent'
n_veh = 8
M = 16

n_input = 3 * n_veh + M
n_output = M
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

'''##---Initializations networks parameters---##
action_range = 1.0
batch_size = 64
# --------------------------------------------------------------
replay_buffer_size = 1e6
replay_buffer = ReplayBuffer(replay_buffer_size)
hidden_dim = 256

update_itr = 1
AUTO_ENTROPY=True
DETERMINISTIC=False
frame_idx = 0
explore_steps = 0 # for random action sampling in the beginning of training
#--------------------------------------------'''

ro = 10 ** -2  # 参考距离d0 = 1m处的平均路径损耗功率增益 10dBm = 0.01w

cascaded_gain = 0
lamb = 0.15  # 载波长度
d = 0.02  # RIS元素之间的距离

sigma = 10 ** (-7)
alpha1 = 2.5  # 公式中的alpha
alpha2 = 2.5


class Model:
    def __init__(self, down_lane, up_lane, left_lane, right_lane, width, height, n_veh, M, control_bit):
        self.down_lanes = down_lane
        self.up_lanes = up_lane
        self.left_lanes = left_lane
        self.right_lanes = right_lane
        self.width = width
        self.height = height

        self.n_veh = n_veh

        self.M = M  # 元素数量
        self.control_bit = control_bit  # 元素相移控制比特


        self.ris_env = Environment.Environ(self.down_lanes, self.up_lanes, self.left_lanes, self.right_lanes, self.width, self.height, self.n_veh, self.M,
                                           self.control_bit)


    def load_saved_models(self):
        """Restore all models"""

        # --------------------------------------------

        print("Initializing DDPG Agent")
        #self.agent = SAC_Trainer(replay_buffer, n_input, n_output, hidden_dim=hidden_dim, action_range=action_range)
        self.agent = Agent(alpha, beta, n_input, tau, n_output, gamma, memory_size, C_fc1_dims, C_fc2_dims, C_fc3_dims,
                     A_fc1_dims, A_fc2_dims, batch_size, n_veh)
        print("\nRestoring the model...")
        self.agent.load_models()

    def get_ris_action(self, state_old_all):
        action = self.agent.choose_action(np.asarray(state_old_all).flatten())
        #action = self.agent.policy_net.get_action(np.asarray(state_old_all).flatten(), deterministic=DETERMINISTIC)
        action = np.clip(action, -0.999, 0.999)

        return action
