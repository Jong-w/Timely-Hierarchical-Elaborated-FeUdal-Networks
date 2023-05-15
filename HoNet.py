import torch
from torch import nn
from torch.nn.functional import cosine_similarity as d_cos, normalize

from utils import init_hidden, weight_init
from preprocess import Preprocessor
from dilated_lstm import DilatedLSTM


class HONET(nn.Module):
    def __init__(self,
                 num_workers,
                 input_dim,
                 hidden_dim_Hierarchies= [256, 128, 64, 32, 16],     # set of hidden_dims <- list form
                 time_horizon_Hierarchies = [20, 15, 10, 5, 1],   #time_horizon & dilation -> time_horizon  # set of time_horizon <- list form
                 n_actions,
                 device='cuda',
                 args=None):

        super().__init__()
        self.num_workers = num_workers
        self.time_horizon = time_horizon_Hierarchies
        self.hidden_dim = hidden_dim_Hierarchies
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

    def forward(self, x, goals_m, states_m, goals_s, states_s, mask, save=True):
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
        hierarchies_selected = [i for i, value in enumerate(hierarchies_selected) if value == 1]
        z = self.percept(x)

        goal_5, hidden_5, state_5, value_5 = self.Hierarchy5(z, self.hidden_5, mask)
        goal_4, hidden_4, state_4, value_4 = self.Hierarchy4(z, goal_5, self.hidden_4, mask)
        goal_3, hidden_3, state_3, value_3 = self.Hierarchy3(z, goal_4, self.hidden_3, mask)
        goal_2, hidden_2, state_2, value_2 = self.Hierarchy2(z, goal_3, self.hidden_2, mask)
        goal_1, hidden_1, state_1, value_1 = self.Hierarchy1(z, goal_2, self.hidden_1, mask)

        Hierarchies = [self.Hierarchy5, self.Hierarchy4, self.Hierarchy3, self.Hierarchy2, self.Hierarchy1]

        for num in hierarchies_selected:
            Hierarchies[num]

        # Ensure that we only have a list of size 2*c + 1, and we use FiLo
        if len(goals_m) > (2 * self.c_m + 1):
            goals_m.pop(0)
            states_m.pop(0)

        if len(goals_s) > (2 * self.c_s + 1):
            goals_s.pop(0)
            states_s.pop(0)

        goals_m.append(goal_m)
        states_m.append(state_m.detach())  # states_m never have gradients active

        goals_s.append(goal_s)
        states_s.append(state_s.detach())

        # The manager is ahead at least c steps, so we feed
        # only the first c+1 states_m to worker
        action_dist, hidden_w, value_w = self.worker(
            z, goals_s[:self.c_s + 1], self.hidden_w, mask)

        if save:
            # Optional, dont do this for the next_v
            self.hidden_m = hidden_m
            self.hidden_w = hidden_w

        return action_dist, goals_m, states_m, value_m, goals_s, states_s, value_s, value_w