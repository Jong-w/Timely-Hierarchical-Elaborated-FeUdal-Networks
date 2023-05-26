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
                 dynamic=0,
                 args=None):

        super().__init__()
        self.num_workers = num_workers
        self.time_horizon = time_horizon_Hierarchies
        c = hidden_dim_Hierarchies
        self.n_actions = n_actions
        self.dynamic = dynamic
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

    def forward(self, x, goals_5, states_5, goals_4, states_4, goals_3, states_3, goals_2, states_2,  mask, step, save=True):
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

        goal_5, hidden_5, state_5, value_5 = self.Hierarchy5(z, self.hidden_5, hierarchies_selected[3], mask)
        goal_4, hidden_4, state_4, value_4 = self.Hierarchy4(z, goal_5, self.hidden_4, hierarchies_selected[2], mask)
        goal_3, hidden_3, state_3, value_3 = self.Hierarchy3(z, goal_4, self.hidden_3, hierarchies_selected[1], mask)
        goal_2, hidden_2, state_2, value_2 = self.Hierarchy2(z, goal_3, self.hidden_2, hierarchies_selected[0], mask)

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

class Perception(nn.Module):
    def __init__(self, input_dim, d, mlp=False):
        super().__init__()
        if mlp:
            self.percept = nn.Sequential(
                nn.Linear(input_dim[-1] * input_dim[1] * input_dim[2], 256),
                nn.ReLU(),
                # nn.Linear(256, 256),
                # nn.ReLU(),
                nn.Linear(256, d),
                nn.ReLU())
        else:
            self.percept = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=4, stride=4),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.modules.Flatten(),
                nn.Linear(32 * 14 * 14, d),
                nn.ReLU())

    def forward(self, x):
        return self.percept(x)


###########################################################################################################2차 수정 해야하는 부분.....
class hierarchy5(nn.Module):
    def __init__(self, c_m, c_s, d, r_m, args, device):
        super().__init__()
        self.c_m = c_m  # Time Horizon
        self.c_s = c_s
        self.d = d  # Hidden dimension size
        self.r = r_m  # Dilation level
        self.eps = args.eps
        self.device = device

        self.Mspace = nn.Linear(self.d, self.d)
        self.Mrnn = DilatedLSTM(self.d, self.d, self.r)
        self.critic = nn.Linear(self.d, 1)

    def forward(self, z, hidden, mask):
        state = self.Mspace(z).relu()
        hidden = (mask * hidden[0], mask * hidden[1])
        goal_hat, hidden = self.Mrnn(state, hidden)
        value_est = self.critic(goal_hat)

        # From goal_hat to goal
        goal = normalize(goal_hat)
        state = state.detach()

        if (self.eps > torch.rand(1)[0]):
            # To encourage exploration in transition policy,
            # at every step with a small probability ε
            # we emit a random goal sampled from a uni-variate Gaussian.
            goal = torch.randn_like(goal, requires_grad=False)

        return goal, hidden, state, value_est

    def goal_goal_cosine(self, goals_m, goals_s, masks):
        """For the manager, we update using the cosine of:
            cos( S_{t+c} - S_{t}, G_{t} )

        Remember that states_m, goals_m are of size c * 2 + 1, with our current
        update time step right in the middle at t = c + 1.
        States should not have a gradient active, but goals_m[t] _should_.

        Args:
            states_m ([type]): list of size 2*C + 1, each element B x D
            goals_m ([type]): list of size 2*C + 1, each element B x D

        Returns:
            [type]: cosine distance between:
                        the difference state s_{t+c} - s_{t},
                        the goal embedding at timestep t g_t(theta).
        """
        t_m = self.c_m
        t_s = self.c_s
        mask = torch.stack(masks[t_m: t_m + self.c_m - 1]).prod(dim=0)

        # goals_s_t1 = torch.cat([goals_s[t_s - self.c_s], goals_s[t_s - self.c_s]], dim=1)
        # goals_s_t2 = torch.cat([goals_s[t_s], goals_s[t_s]], dim=1)
        # cosine_dist = d_cos(goals_s_t1 - goals_s_t2, goals_m[t_m])

        cosine_dist = d_cos(goals_s[t_s + self.c_s] - goals_s[t_s], goals_m[t_m])

        cosine_dist = mask * cosine_dist.unsqueeze(-1)

        return cosine_dist

    def state_goal_cosine(self, states_m, goals_m, masks):
        """For the manager, we update using the cosine of:
            cos( S_{t+c} - S_{t}, G_{t} )

        Remember that states_m, goals_m are of size c * 2 + 1, with our current
        update time step right in the middle at t = c + 1.
        States should not have a gradient active, but goals_m[t] _should_.

        Args:
            states_m ([type]): list of size 2*C + 1, each element B x D
            goals_m ([type]): list of size 2*C + 1, each element B x D

        Returns:
            [type]: cosine distance between:
                        the difference state s_{t+c} - s_{t},
                        the goal embedding at timestep t g_t(theta).
        """
        t_m = self.c_m
        t_s = self.c_s
        mask = torch.stack(masks[t_m: t_m + self.c_m - 1]).prod(dim=0)

        # goals_s_t1 = torch.cat([goals_s[t_s - self.c_s], goals_s[t_s - self.c_s]], dim=1)
        # goals_s_t2 = torch.cat([goals_s[t_s], goals_s[t_s]], dim=1)
        # cosine_dist = d_cos(goals_s_t1 - goals_s_t2, goals_m[t_m])

        cosine_dist = d_cos(states_m[t_s + self.c_s] - states_m[t_s], goals_m[t_m])

        cosine_dist = mask * cosine_dist.unsqueeze(-1)

        return cosine_dist
