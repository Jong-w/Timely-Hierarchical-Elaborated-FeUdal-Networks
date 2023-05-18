import torch
from torch import nn
from torch.nn.functional import cosine_similarity as d_cos, normalize

from utils import init_hidden, weight_init
from preprocess import Preprocessor
from dilated_lstm import DilatedLSTM
import numpy as np


class HONET(nn.Module):
    def __init__(self,
                 num_workers,
                 input_dim,
                 hidden_dim_Hierarchies= [256, 128, 64, 32, 16],     # set of hidden_dims <- list form
                 time_horizon_Hierarchies = [20, 15, 10, 5, 1],   #time_horizon & dilation -> time_horizon  # set of time_horizon <- list form
                 n_actions=17,
                 device='cuda',
                 args=None):

        super().__init__()
        self.num_workers = num_workers
        self.time_horizon = time_horizon_Hierarchies
        c = hidden_dim_Hierarchies
        self.n_actions = n_actions
        self.device = device

        self.preprocessor = Preprocessor(input_dim, device, mlp=False)
        self.percept = Perception(input_dim, self.hidden_dim[0], mlp=False)
        self.policy_network = Policy_Network(input_dim, self.hidden_dim[0], mlp=False)
        self.Hierarchy5 = Hierarchy5(self.time_horizon[4], self.hidden_dim[4], args, device)  #self.r_m = dilation <- replace by time_horizon
        self.Hierarchy4 = Hierarchy4(self.time_horizon[3], self.hidden_dim[3], args, device)
        self.Hierarchy3 = Hierarchy3(self.time_horizon[2], self.hidden_dim[2], args, device)
        self.Hierarchy2 = Hierarchy2(self.time_horizon[1], self.hidden_dim[1], args, device)
        self.Hierarchy1 = Hierarchy1(self.time_horizon[0], self.hidden_dim[0], args, device)

        self.hidden_5 = init_hidden(args.num_workers, self.time_horizon[4] * self.hidden_dim[4],
                                    device=device, grad=True)
        self.hidden_4 = init_hidden(args.num_workers, self.time_horizon[3] * self.hidden_dim[3],
                                    device=device, grad=True)
        self.hidden_3 = init_hidden(args.num_workers, self.time_horizon[2] * self.hidden_dim[2],
                                    device=device, grad=True)
        self.hidden_2 = init_hidden(args.num_workers, self.time_horizon[1] * self.hidden_dim[1],
                                    device=device, grad=True)
        self.hidden_1 = init_hidden(args.num_workers, self.time_horizon[0] * self.hidden_dim[0],
                                    device=device, grad=True)

        self.args = args
        self.to(device)
        self.apply(weight_init)

    def forward(self, x, goals_5, states_5, goals_4, states_4, goals_3, states_3, goals_2, states_2,  mask, save=True):
        """A forward pass through the whole feudal network.

        Order of operations:
        1. input goes through a preprocessor to normalize and put on device
        2. normalized input goes to the perception module resulting in a state
        3. state is input for manager which produces a goal
        4. state and goal is both input for worker which produces an action
           distribution.

        Args:
            x (np.ndarray): observation from the environment
            goals_m (list):  list of goal tensors, length = 2 * r + 1
            states_m (list): list of state tensors, length = 2 * r + 1
            mask (tensor): mask describing for each worker if episode is done.
            save (bool, optional): If we are calculating next_v, we do not
                                   store rnn states_m. Defaults to True.
        """
        x = self.preprocessor(x)
        hierarchies_selected = self.policy_network(x)  #hierarchies_selectedsms is one-hot encoded form #except Hierarchy1
        z = self.percept(x)

        goal_5, hidden_5, state_5, value_5 = self.Hierarchy5(z, self.hidden_5, mask)
        if hierarchies_selected[4] == 0:
            goal_5 = np.zeros(goal_5.shape)
        goal_4, hidden_4, state_4, value_4 = self.Hierarchy4(z, goal_5, self.hidden_4, mask)
        if hierarchies_selected[3] == 0:
            goal_4 = np.zeros(goal_4.shape)
        goal_3, hidden_3, state_3, value_3 = self.Hierarchy3(z, goal_4, self.hidden_3, mask)
        if hierarchies_selected[2] == 0:
            goal_3 = np.zeros(goal_3.shape)
        goal_2, hidden_2, state_2, value_2 = self.Hierarchy2(z, goal_3, self.hidden_2, mask)
        if hierarchies_selected[1] == 0:
            goal_2 = np.zeros(goal_2.shape)

        # Ensure that we only have a list of size 2*c + 1, and we use FiLo
        if len(goals_5) > (2 * self.time_horizon[4] + 1):
            goals_5.pop(0)
            states_5.pop(0)

        if len(goals_4) > (2 * self.time_horizon[3] + 1):
            goals_4.pop(0)
            states_4.pop(0)

        if len(goals_3) > (2 * self.time_horizon[2] + 1):
            goals_3.pop(0)
            states_3.pop(0)

        if len(goals_2) > (2 * self.time_horizon[1] + 1):
            goals_2.pop(0)
            states_2.pop(0)


        goals_5.append(goal_5)
        states_5.append(state_5.detach())

        goals_4.append(goal_5)
        states_4.append(state_5.detach())

        goals_3.append(goal_5)
        states_3.append(state_5.detach())

        goals_2.append(goal_5)
        states_2.append(state_5.detach())


        # The manager is ahead at least c steps, so we feed
        # only the first c+1 states_m to worker

        action_dist, hidden_1, value_1 = self.Hierarchy1(z, goal_2, self.hidden_1, mask) #action_dist, hidden_1, state_1, value_1

        if save:
            # Optional, dont do this for the next_v
            self.hidden_5 = hidden_5
            self.hidden_4 = hidden_4
            self.hidden_3 = hidden_3
            self.hidden_2 = hidden_2
            self.hidden_1 = hidden_1

        return action_dist, goals_5, states_5, value_5, goals_4, states_4, value_4, goals_3, states_3, value_3, goals_2, states_2, value_2, value_1

        def intrinsic_reward(self, states_2, goals_2, masks):
            return self.worker.intrinsic_reward(states_2, goals_2, masks)

        def state_goal_cosine(self, states_n, goals_n, masks, hierarchy_num):
            if hierarchy_num == 5:
                return self.Hierarchy5.state_goal_cosine(states_n, goals_n, masks)
            if hierarchy_num == 4:
                return self.Hierarchy4.state_goal_cosine(states_n, goals_n, masks)
            if hierarchy_num == 3:
                return self.Hierarchy3.state_goal_cosine(states_n, goals_n, masks)
            if hierarchy_num == 2:
                return self.Hierarchy2.state_goal_cosine(states_n, goals_n, masks)

        def repackage_hidden(self):
            def repackage_rnn(x):
                return [item.detach() for item in x]

            self.hidden_5 = repackage_rnn(self.hidden_5)
            self.hidden_4 = repackage_rnn(self.hidden_4)
            self.hidden_3 = repackage_rnn(self.hidden_3)
            self.hidden_2 = repackage_rnn(self.hidden_2)
            self.hidden_1 = repackage_rnn(self.hidden_1)

        def init_obj(self):
            template_5 = torch.zeros(self.b, self.hidden_dim[4])
            template_4 = torch.zeros(self.b, self.hidden_dim[3])
            template_3 = torch.zeros(self.b, self.hidden_dim[2])
            template_2 = torch.zeros(self.b, self.hidden_dim[1])
            goals_5 = [torch.zeros_like(template_5).to(self.device) for _ in range(2 * self.time_horizon[4] + 1)]
            states_5 = [torch.zeros_like(template_5).to(self.device) for _ in range(2 * self.time_horizon[4] + 1)]
            goals_4 = [torch.zeros_like(template_5).to(self.device) for _ in range(2 * self.time_horizon[3] + 1)]
            states_4 = [torch.zeros_like(template_5).to(self.device) for _ in range(2 * self.time_horizon[3] + 1)]
            goals_3 = [torch.zeros_like(template_5).to(self.device) for _ in range(2 * self.time_horizon[2] + 1)]
            states_3 = [torch.zeros_like(template_5).to(self.device) for _ in range(2 * self.time_horizon[2] + 1)]
            goals_2 = [torch.zeros_like(template_5).to(self.device) for _ in range(2 * self.time_horizon[1] + 1)]
            states_2 = [torch.zeros_like(template_5).to(self.device) for _ in range(2 * self.time_horizon[1] + 1)]
            masks = [torch.ones(self.hidden_dim[4], 1).to(self.device) for _ in range(2 * self.time_horizon[1] + 1)]
            return goals_5, states_5, goals_4, states_4, goals_3, states_3, goals_2, states_2, masks

