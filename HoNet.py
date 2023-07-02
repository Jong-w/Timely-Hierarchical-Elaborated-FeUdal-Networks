import torch
from torch import nn
from torch.nn.functional import cosine_similarity as d_cos, normalize
import torch.nn.functional as F

from utils import init_hidden, weight_init
from preprocess import Preprocessor
from dilated_lstm import DilatedLSTM
import numpy as np


class HONET(nn.Module):
    def __init__(self,
                 num_workers,
                 input_dim,
                 hidden_dim_Hierarchies = [256, 256, 256, 256, 256],     # set of hidden_dims <- list form
                 time_horizon_Hierarchies = [1, 5, 10, 15, 20, 25],   #time_horizon & dilation -> time_horizon  # set of time_horizon <- list form
                 n_actions=17,
                 device='cuda',
                 dynamic=0,
                 args=None):

        super().__init__()
        self.num_workers = num_workers
        self.time_horizon = time_horizon_Hierarchies
        self.hidden_dim = hidden_dim_Hierarchies
        self.n_actions = n_actions
        self.dynamic = dynamic
        self.device = device
        self.eps = args.eps

        template_0 = torch.zeros(self.num_workers, self.hidden_dim[4])
        self.goal_0 = [torch.zeros_like(template_0).to(self.device) for _ in range(2 * self.time_horizon[4] + 1)]

        self.preprocessor = Preprocessor(input_dim, device, mlp=False)
        self.percept = Perception(self.hidden_dim[0],  self.time_horizon[0], mlp=False)
        self.policy_network = Policy_Network()
        self.Hierarchy5 = Hierarchy5(self.time_horizon[4], self.hidden_dim[4], args, device)
        self.Hierarchy4 = Hierarchy4(self.time_horizon[3], self.hidden_dim[3], args, device)
        self.Hierarchy3 = Hierarchy3(self.time_horizon[2], self.hidden_dim[2], args, device)
        self.Hierarchy2 = Hierarchy2(self.time_horizon[1], self.hidden_dim[1], args, device)
        self.Hierarchy1 = Hierarchy1(self.num_workers, self.time_horizon[0], self.hidden_dim[0],  self.hidden_dim[1],  self.hidden_dim[4], self.n_actions, device)

        self.hidden_5 = init_hidden(args.num_workers, self.time_horizon[4] * self.hidden_dim[4],
                                    device=device, grad=True)
        self.hidden_4 = init_hidden(args.num_workers, self.time_horizon[3] * self.hidden_dim[3],
                                    device=device, grad=True)
        self.hidden_3 = init_hidden(args.num_workers, self.time_horizon[2] * self.hidden_dim[2],
                                    device=device, grad=True)
        self.hidden_2 = init_hidden(args.num_workers, self.time_horizon[1] * self.hidden_dim[1],
                                    device=device, grad=True)
        self.hidden_1 = init_hidden(args.num_workers, self.n_actions * self.hidden_dim[0],
                                    device=device, grad=True)
        self.hidden_percept = init_hidden(args.num_workers, self.time_horizon[0] * self.hidden_dim[0],
                                    device=device, grad=True)

        self.args = args
        self.to(device)
        self.apply(weight_init)

    def forward(self, x, goals_5, states_total, goals_4, goals_3, goals_2,  mask, step, save=True):
        """A forward pass through the whole feudal network.

        Order of operations:
        1. input goes through a preprocessor to normalize and put on device
        2. normalized input goes to the perception module resulting in a state
        3. state is input for manager which produces a goal
        4. state and goal is both input for worker which produces an action
           distribution.

        Args:
            x (np.ndarray): observation from the environment
            goals (list):  list of goal tensors, length = 2 * r + 1
            states (list): list of state tensors, length = 2 * r + 1
            mask (tensor): mask describing for each worker if episode is done.
            save (bool, optional): If we are calculating next_v, we do not
                                   store rnn states. Defaults to True.
        """
        train_eps = self.eps
        x = self.preprocessor(x)
        hierarchies_selected = self.policy_network(x)
        if train_eps > torch.rand(1)[0]:
            hierarchies_selected[0] = 0
        if train_eps > torch.rand(1)[0]:
            hierarchies_selected[1] = 0
        if train_eps > torch.rand(1)[0]:
            hierarchies_selected[2] = 0
        train_eps * 0.99

        z, hidden_percept = self.percept(x, self.hidden_percept, mask)

        goal_5, hidden_5, value_5 = self.Hierarchy5(z, self.goal_0, self.hidden_5, hierarchies_selected[:, 2], mask)
        goal_4, hidden_4, value_4 = self.Hierarchy4(z, goal_5, self.hidden_4, hierarchies_selected[:, 1], mask)
        goal_3, hidden_3, value_3 = self.Hierarchy3(z, goal_4, self.hidden_3, hierarchies_selected[:, 0], mask)
        goal_2, hidden_2, value_2 = self.Hierarchy2(z, goal_3, self.hidden_2,  mask)

        # Ensure that we only have a list of size 2*c + 1, and we use FiLo
        if len(goals_5) > (2 * self.time_horizon[4] + 1):
            goals_5.pop(0)
            states_total.pop(0)

        if len(goals_4) > (2 * self.time_horizon[3] + 1):
            goals_4.pop(0)

        if len(goals_3) > (2 * self.time_horizon[2] + 1):
            goals_3.pop(0)

        if len(goals_2) > (2 * self.time_horizon[1] + 1):
            goals_2.pop(0)

        goals_5.append(goal_5)
        goals_4.append(goal_4)
        goals_3.append(goal_3)
        goals_2.append(goal_2)
        states_total.append(z.detach())

        action_dist, hidden_1, value_1 = self.Hierarchy1(z, goals_2[:self.time_horizon[1] + 1], self.hidden_1, mask)

        if save:
            # Optional, don't do this for the next_v
            self.hidden_percept = hidden_percept
            self.hidden_5 = hidden_5
            self.hidden_4 = hidden_4
            self.hidden_3 = hidden_3
            self.hidden_2 = hidden_2
            self.hidden_1 = hidden_1

        return action_dist, goals_5, states_total, value_5, goals_4, value_4, goals_3, value_3, goals_2, value_2, value_1, hierarchies_selected

    def intrinsic_reward(self, states_2, goals_2, masks):
        return self.Hierarchy1.intrinsic_reward(states_2, goals_2, masks)

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

        self.hidden_percept = repackage_rnn(self.hidden_percept)
        self.hidden_5 = repackage_rnn(self.hidden_5)
        self.hidden_4 = repackage_rnn(self.hidden_4)
        self.hidden_3 = repackage_rnn(self.hidden_3)
        self.hidden_2 = repackage_rnn(self.hidden_2)
        self.hidden_1 = repackage_rnn(self.hidden_1)

    def init_obj(self):
        template_5 = torch.zeros(self.num_workers, self.hidden_dim[4])
        template_4 = torch.zeros(self.num_workers, self.hidden_dim[3])
        template_3 = torch.zeros(self.num_workers, self.hidden_dim[2])
        template_2 = torch.zeros(self.num_workers, self.hidden_dim[1])
        goals_5 = [torch.zeros_like(template_5).to(self.device) for _ in range(2 * self.time_horizon[4] + 1)]
        states_total = [torch.zeros_like(template_5).to(self.device) for _ in range(2 * self.time_horizon[4] + 1)]
        goals_4 = [torch.zeros_like(template_4).to(self.device) for _ in range(2 * self.time_horizon[3] + 1)]
        goals_3 = [torch.zeros_like(template_3).to(self.device) for _ in range(2 * self.time_horizon[2] + 1)]
        goals_2 = [torch.zeros_like(template_2).to(self.device) for _ in range(2 * self.time_horizon[1] + 1)]
        masks = [torch.ones(self.num_workers, 1).to(self.device) for _ in range(2 * self.time_horizon[4] + 1)]
        return goals_5, states_total, goals_4, goals_3, goals_2, masks

class Perception(nn.Module):
    def __init__(self, d, time_horizon,  mlp=False):
        super().__init__()
        self.percept = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=5),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.modules.Flatten(),
            nn.Linear(32 * 20 * 15, d),
            nn.ReLU())
        self.Mrnn = DilatedLSTM(d, d, time_horizon)

    def forward(self, x, hidden, mask):
        x1 = self.percept(x)
        hidden = (mask * hidden[0], mask * hidden[1])
        x2 = self.Mrnn(x1, hidden)
        return x2[0], hidden

class Policy_Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.policy_network = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=5, stride=5),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.modules.Flatten(),
                nn.Linear(32 * 20 * 15, 3),
                nn.ReLU())
    def forward(self, x):
        policy_network_result = self.policy_network(x)
        policy_network_result -= policy_network_result.min(1, keepdim=True)[0]
        policy_network_result /= policy_network_result.max(1, keepdim=True)[0]
        policy_network_result = policy_network_result.round()
        return policy_network_result

class Hierarchy5(nn.Module):
    def __init__(self, time_horizon, hidden_dim, args, device):
        super().__init__()
        self.time_horizon = time_horizon  # Time Horizon
        self.hidden_dim = hidden_dim  # Hidden dimension size
        self.eps = args.eps
        self.device = device
        self.Mrnn = DilatedLSTM(self.hidden_dim, self.hidden_dim, self.time_horizon)
        self.critic = nn.Linear(self.hidden_dim, 1)

    def forward(self, z, goal, hidden, hierarchies_selected, mask):

        hidden = (mask * hidden[0], mask * hidden[1])
        goal_hat, hidden = self.Mrnn(z, hidden)
        value_est = self.critic(goal_hat)

        # From goal_hat to goal
        goal = normalize(goal_hat)

        if (self.eps > torch.rand(1)[0]):
            goal = torch.randn_like(goal, requires_grad=False)

        hierarchies_selected = hierarchies_selected.reshape(64, 1)
        goal = hierarchies_selected.expand(64,256) * goal + 1e-7

        return goal, hidden, value_est

    def state_goal_cosine(self, states, goals, masks):

        t = self.time_horizon
        mask = torch.stack(masks[t: t + self.time_horizon - 1]).prod(dim=0)

        cosine_dist = d_cos(states[t + t] - states[t], goals[t])

        cosine_dist = mask * cosine_dist.unsqueeze(-1)

        return cosine_dist

class Hierarchy4(nn.Module):
    def __init__(self, time_horizon, hidden_dim, args, device):
        super().__init__()
        self.time_horizon = time_horizon  # Time Horizon
        self.hidden_dim = hidden_dim  # Hidden dimension size
        self.eps = args.eps
        self.device = device
        self.Mrnn = DilatedLSTM(self.hidden_dim, self.hidden_dim, self.time_horizon)
        self.critic = nn.Linear(self.hidden_dim, 1)

    def forward(self, z, goal, hidden, hierarchies_selected, mask):

        hidden = (mask * hidden[0], mask * hidden[1])
        goal_hat, hidden = self.Mrnn(z, hidden)
        value_est = self.critic(goal_hat)

        # From goal_hat to goal
        goal_hat = goal_hat + goal
        goal = normalize(goal_hat)

        if (self.eps > torch.rand(1)[0]):
            goal = torch.randn_like(goal, requires_grad=False)

        hierarchies_selected = hierarchies_selected.reshape(64, 1)
        goal = hierarchies_selected.expand(64,256) * goal + 1e-7

        return goal, hidden, value_est

    def state_goal_cosine(self, states, goals, masks):

        t = self.time_horizon
        mask = torch.stack(masks[t: t + self.time_horizon - 1]).prod(dim=0)

        cosine_dist = d_cos(states[t + t] - states[t], goals[t])

        cosine_dist = mask * cosine_dist.unsqueeze(-1)

        return cosine_dist

class Hierarchy3(nn.Module):
    def __init__(self, time_horizon, hidden_dim, args, device):
        super().__init__()
        self.time_horizon = time_horizon  # Time Horizon
        self.hidden_dim = hidden_dim  # Hidden dimension size
        self.eps = args.eps
        self.device = device
        self.Mrnn = DilatedLSTM(self.hidden_dim, self.hidden_dim, self.time_horizon)
        self.critic = nn.Linear(self.hidden_dim, 1)

    def forward(self, z, goal, hidden, hierarchies_selected, mask):

        hidden = (mask * hidden[0], mask * hidden[1])
        goal_hat, hidden = self.Mrnn(z, hidden)
        value_est = self.critic(goal_hat)

        # From goal_hat to goal
        goal_hat = goal_hat + goal
        goal = normalize(goal_hat)

        if (self.eps > torch.rand(1)[0]):
            goal = torch.randn_like(goal, requires_grad=False)

        hierarchies_selected = hierarchies_selected.reshape(64, 1)
        goal = hierarchies_selected.expand(64,256) * goal + 1e-7

        return goal, hidden, value_est

    def state_goal_cosine(self, states, goals, masks):

        t = self.time_horizon
        mask = torch.stack(masks[t: t + self.time_horizon - 1]).prod(dim=0)

        cosine_dist = d_cos(states[t + t] - states[t], goals[t])

        cosine_dist = mask * cosine_dist.unsqueeze(-1)

        return cosine_dist

class Hierarchy2(nn.Module):
    def __init__(self, time_horizon, hidden_dim, args, device):
        super().__init__()
        self.time_horizon = time_horizon  # Time Horizon
        self.hidden_dim = hidden_dim  # Hidden dimension size
        self.eps = args.eps
        self.device = device
        self.Mrnn = DilatedLSTM(self.hidden_dim, self.hidden_dim, self.time_horizon)
        self.critic = nn.Linear(self.hidden_dim, 1)

    def forward(self, z, goal, hidden, mask):

        hidden = (mask * hidden[0], mask * hidden[1])
        goal_hat, hidden = self.Mrnn(z, hidden)
        value_est = self.critic(goal_hat)

        # From goal_hat to goal
        goal_hat = goal_hat + goal
        goal = normalize(goal_hat)

        if (self.eps > torch.rand(1)[0]):
            goal = torch.randn_like(goal, requires_grad=False)

        return goal, hidden, value_est

    def state_goal_cosine(self, states, goals, masks):

        t = self.time_horizon
        mask = torch.stack(masks[t: t + self.time_horizon - 1]).prod(dim=0)

        cosine_dist = d_cos(states[t + t] - states[t], goals[t])

        cosine_dist = mask * cosine_dist.unsqueeze(-1)

        return cosine_dist

class Hierarchy1(nn.Module):
    def __init__(self, num_workers, time_horizon, hiddne_dim_5, hidden_dim_2, hidden_dim_1, num_actions, device):
        super().__init__()
        self.num_workers = num_workers
        self.time_horizon = time_horizon
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.num_actions = num_actions
        self.device = device

        self.Wrnn = nn.LSTMCell(hiddne_dim_5, hidden_dim_1 * self.num_actions)
        self.phi = nn.Linear(self.hidden_dim_2 , hidden_dim_1, bias=False)

        self.critic = nn.Sequential(
            nn.Linear(self.hidden_dim_1 * self.num_actions, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )

    def forward(self, z, goal, hidden, mask):

        hidden = (mask * hidden[0], mask * hidden[1])
        u, cx = self.Wrnn(z, hidden)
        hidden = (u, cx)

        # Detaching is vital, no end to end training
        goal = torch.stack(goal).detach().sum(dim=0)
        w = goal
        #w = self.phi(goal)
        value_est = self.critic(u)

        u = u.reshape(u.shape[0], self.hidden_dim_1, self.num_actions)
        a = torch.einsum("bk, bka -> ba", w, u).softmax(dim=-1)

        return a, hidden, value_est

    def intrinsic_reward(self, states_s, goals_s, masks):

        t = self.time_horizon
        r_i = torch.zeros(self.num_workers, 1).to(self.device)
        mask = torch.ones(self.num_workers, 1).to(self.device)

        for i in range(1, self.time_horizon + 1):
            r_i_t = d_cos(states_s[t] - states_s[t - i], goals_s[t - i]).unsqueeze(-1)
            r_i += (mask * r_i_t)

            mask = mask * masks[t - i]

        r_i = r_i.detach()
        return r_i / self.time_horizon

def mp_loss(storage, next_v_5, next_v_4, next_v_3, next_v_2, next_v_1, args):

    # Discount rewards, both of size B x T
    ret_5 = next_v_5
    ret_4 = next_v_4
    ret_3 = next_v_3
    ret_2 = next_v_2
    ret_1 = next_v_1

    storage.placeholder()  # Fill ret_m, ret_w with empty vals
    for i in reversed(range(args.num_steps)):
        ret_5 = storage.r[i] + args.gamma_5 * ret_5 * storage.m[i]
        ret_4 = storage.r[i] + args.gamma_4 * ret_4 * storage.m[i]
        ret_3 = storage.r[i] + args.gamma_3 * ret_3 * storage.m[i]
        ret_2 = storage.r[i] + args.gamma_2 * ret_2 * storage.m[i]
        ret_1 = storage.r[i] + args.gamma_1 * ret_1 * storage.m[i]
        storage.ret_5[i] = ret_5
        storage.ret_4[i] = ret_4
        storage.ret_3[i] = ret_3
        storage.ret_2[i] = ret_2
        storage.ret_1[i] = ret_1

    # Optionally, normalize the returns
    storage.normalize(['ret_5', 'ret_4', 'ret_3', 'ret_2', 'ret_1'])

    rewards_intrinsic, value_5, value_4, value_3, value_2, value_1, ret_5, ret_4, ret_3, ret_2, ret_1, logps, entropy, \
        state_goal_5_cosines, state_goal_4_cosines, state_goal_3_cosines, state_goal_2_cosines, hierarchy_selected = storage.stack(
        ['r_i', 'v_5', 'v_4', 'v_3', 'v_2', 'v_1', 'ret_5', 'ret_4', 'ret_3', 'ret_2', 'ret_1',
         'logp', 'entropy', 'state_goal_5_cos', 'state_goal_4_cos', 'state_goal_3_cos', 'state_goal_2_cos', 'hierarchy_selected'])

    advantage_5 = ret_5 - value_5
    loss_5 = (state_goal_5_cosines * advantage_5.detach()).mean()
    value_5_loss = 0.5 * advantage_5.pow(2).mean()

    advantage_4 = ret_4 - value_4
    loss_4 = (state_goal_4_cosines * advantage_4.detach()).mean()
    value_4_loss = 0.5 * advantage_4.pow(2).mean()

    advantage_3 = ret_3 - value_3
    loss_3 = (state_goal_3_cosines * advantage_3.detach()).mean()
    value_3_loss = 0.5 * advantage_3.pow(2).mean()

    advantage_2 = ret_2 - value_2
    loss_2 = (state_goal_2_cosines * advantage_2.detach()).mean()
    value_2_loss = 0.5 * advantage_2.pow(2).mean()

    # Calculate advantages, size B x T
    advantage_1 = ret_1 + args.alpha * rewards_intrinsic - value_1
    loss_1 = (logps * advantage_1.detach()).mean()
    value_1_loss = 0.5 * advantage_1.pow(2).mean()

    entropy = entropy.mean()

    loss = (- loss_5 - loss_4 - loss_3 - loss_2 - loss_1 + value_5_loss + value_4_loss + value_3_loss + value_2_loss + value_1_loss) - args.entropy_coef * entropy

    return loss, {'loss/total_mp_loss': loss.item(),
                  'loss/Hierarchy_5': loss_5.item(),
                  'loss/Hierarchy_4': loss_4.item(),
                  'loss/Hierarchy_3': loss_3.item(),
                  'loss/Hierarchy_2': loss_2.item(),
                  'loss/Hierarchy_1': loss_1.item(),

                  'loss/value_Hierarchy_5': value_5_loss.item(),
                  'loss/value_Hierarchy_4': value_4_loss.item(),
                  'loss/value_Hierarchy_3': value_3_loss.item(),
                  'loss/value_Hierarchy_2': value_2_loss.item(),
                  'loss/value_Hierarchy_1': value_1_loss.item(),

                  'value_Hierarchy_1/entropy': entropy.item(),
                  'value_Hierarchy_1/advantage': advantage_1.mean().item(),
                  'value_Hierarchy_1/intrinsic_reward': rewards_intrinsic.mean().item(),

                  'value_Hierarchy_2/cosines': state_goal_2_cosines.mean().item(),
                  'value_Hierarchy_2/advantage': advantage_2.mean().item(),

                  'value_Hierarchy_3/cosines': state_goal_3_cosines.mean().item(),
                  'value_Hierarchy_3/advantage': advantage_3.mean().item(),

                  'Hierarchy_4/cosines': state_goal_4_cosines.mean().item(),
                  'Hierarchy_4/advantage': advantage_4.mean().item(),

                  'Hierarchy_5/cosines': state_goal_5_cosines.mean().item(),
                  'Hierarchy_5/advantage': advantage_5.mean().item()}