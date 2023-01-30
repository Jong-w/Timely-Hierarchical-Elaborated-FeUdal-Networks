import torch
from torch import nn
from torch.nn.functional import cosine_similarity as d_cos, normalize

from utils import init_hidden, weight_init
from preprocess import Preprocessor
from dilated_lstm import DilatedLSTM


class FeudalNetwork(nn.Module):
    def __init__(self,
                 num_workers,
                 input_dim,
                 hidden_dim_manager,
                 hidden_dim_supervisor,
                 hidden_dim_worker,
                 n_actions,
                 time_horizon=10,
                 dilation=10,
                 device='cuda',
                 mlp=True,
                 args=None,
                 partial=1):
        """naming convention inside the FeudalNetwork is selected
        to match paper variable naming convention.
        """

        super().__init__()
        self.b = num_workers
        self.c = time_horizon
        self.d = hidden_dim_manager
        self.n = hidden_dim_supervisor
        self.k = hidden_dim_worker
        self.r = dilation
        self.n_actions = n_actions
        self.device = device

        self.preprocessor = Preprocessor(input_dim, device, mlp, partial)
        self.percept = Perception(input_dim, self.d, mlp)
        self.manager = Manager(self.c, self.d, self.r, args, device)
        self.supervisor = Supervisor(self.c, self.d, self.n, self.r, args, device)
        self.worker = Worker(self.b, self.c, self.d, self.n, self.k, n_actions, device)

        self.hidden_m = init_hidden(args.num_workers, self.r * self.d,
                                    device=device, grad=True)
        self.hidden_s = init_hidden(args.num_workers, self.r * self.n,
                                    device=device, grad=True)
        self.hidden_w = init_hidden(args.num_workers, self.k * n_actions,
                                    device=device, grad=True)

        self.args = args
        self.to(device)
        self.apply(weight_init)

    def forward(self, x, goals_m, states_m, goals_s, states_s,  mask, save=True):
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
        z = self.percept(x)

        goal_m, hidden_m, state_m, value_m = self.manager(
            z, self.hidden_m, mask)

        goal_s, hidden_s, state_s, value_s = self.supervisor(
            z, goal_m, self.hidden_s, mask)

        # Ensure that we only have a list of size 2*c + 1, and we use FiLo
        if len(goals_m) > (2 * self.c + 1):
            goals_m.pop(0)
            states_m.pop(0)

        if len(goals_s) > (2 * self.c + 1):
            goals_s.pop(0)
            states_s.pop(0)

        goals_m.append(goal_m)
        states_m.append(state_m.detach())  # states_m never have gradients active

        goals_s.append(goal_s)
        states_s.append(state_s.detach())

        # The manager is ahead at least c steps, so we feed
        # only the first c+1 states_m to worker
        action_dist, hidden_w, value_w = self.worker(
            z, goals_s[:self.c + 1], self.hidden_w, mask)

        if save:
            # Optional, dont do this for the next_v
            self.hidden_m = hidden_m
            self.hidden_w = hidden_w

        return action_dist, goals_m, states_m, value_m, goals_s, states_s, value_s, value_w

    def intrinsic_reward(self, states_m, goals_m, masks):
        return self.worker.intrinsic_reward(states_m, goals_m, masks)

    def goal_goal_cosine(self, goals_m, goals_s,  masks):
        return self.manager.goal_goal_cosine(goals_m, goals_s,  masks)

    def state_goal_cosine(self, states_m, goals_m, masks):
        return self.supervisor.state_goal_cosine(states_m, goals_m, masks)

    def repackage_hidden(self):
        def repackage_rnn(x):
            return [item.detach() for item in x]

        self.hidden_w = repackage_rnn(self.hidden_w)
        self.hidden_m = repackage_rnn(self.hidden_m)

    def init_obj(self):
        template_m = torch.zeros(self.b, self.d)
        template_s = torch.zeros(self.b, self.n)
        goals_m = [torch.zeros_like(template_m).to(self.device) for _ in range(2*self.c+1)]
        states_m = [torch.zeros_like(template_m).to(self.device) for _ in range(2*self.c+1)]
        goals_s = [torch.zeros_like(template_s).to(self.device) for _ in range(2 * self.c + 1)]
        states_s = [torch.zeros_like(template_s).to(self.device) for _ in range(2 * self.c + 1)]
        masks = [torch.ones(self.b, 1).to(self.device) for _ in range(2*self.c+1)]
        return goals_m, states_m, goals_s, states_s,  masks

class Perception(nn.Module):
    def __init__(self, input_dim, d, mlp=False, partial=1):
        super().__init__()

        if partial:
            input_dim = (56, 56, 3)

        if mlp:
            self.percept = nn.Sequential(
                nn.Linear(input_dim[-1] * input_dim[0] * input_dim[1], 64),
                nn.ReLU(),
                nn.Linear(64, d),
                nn.ReLU())
        else:
            self.percept = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=2, stride=1),
                nn.ReLU(),
                nn.modules.Flatten(),
                nn.Linear(32*4*4, d),
                nn.ReLU())

    def forward(self, x):
        return self.percept(x)

class Manager(nn.Module):
    def __init__(self, c, d, r, args, device):
        super().__init__()
        self.c = c  # Time Horizon
        self.d = d  # Hidden dimension size
        self.r = r  # Dilation level
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

    def goal_goal_cosine(self, goals_m, goals_s,  masks):
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
        t = self.c
        mask = torch.stack(masks[t: t + self.c - 1]).prod(dim=0)

        goals_m_ = torch.cat([goals_m[t], goals_m[t]], dim=1)

        cosine_dist = d_cos(goals_s[t] - goals_s[t - 1], goals_m_)
        cosine_dist = mask * cosine_dist.unsqueeze(-1)

        return cosine_dist

class Supervisor(nn.Module):
    def __init__(self, c, d, n, r, args, device, ):
        super().__init__()
        self.c = c  # Time Horizon
        self.d = d  # Hidden dimension size
        self.r = r  # Dilation level
        self.eps = args.eps
        self.device = device
        self.n = n

        self.Sspace = nn.Linear(self.d, self.n)
        self.phi = nn.Linear(self.d, self.n, bias=False)
        self.Srnn = DilatedLSTM(self.n, self.n, self.r)
        self.critic = nn.Linear(self.n, 1)

    def forward(self, z, goal_m, hidden, mask):
        state = self.Sspace(z).relu()
        manager_goal = self.phi(goal_m)
        hidden = (mask * hidden[0], mask * hidden[1])
        goal_hat, hidden = self.Srnn(manager_goal + state, hidden)
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

    def state_goal_cosine(self, states_s, goals_s, masks):
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
        t = self.c
        mask = torch.stack(masks[t: t + self.c - 1]).prod(dim=0)
        cosine_dist = d_cos(states_s[t + self.c] - states_s[t], goals_s[t])
        cosine_dist = mask * cosine_dist.unsqueeze(-1)
        return cosine_dist

class Worker(nn.Module):
    def __init__(self, b, c, d, n, k, num_actions, device):
        super().__init__()
        self.b = b
        self.c = c
        self.k = k
        self.n = n
        self.num_actions = num_actions
        self.device = device

        self.Wrnn = nn.LSTMCell(d, k * self.num_actions)
        self.phi = nn.Linear(self.n, k, bias=False)

        self.critic = nn.Sequential(
            nn.Linear(k * num_actions, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )

    def forward(self, z, goals_s, hidden, mask):
        """[summary]

        Args:
            z ([type]): latent representation from perception, size B x D
            goals_m ([type]): list of length C, with each tensor of size B x D
            hidden ([type]): recurrent cells, size B x (K*num_actions)

        Returns:
            [type]: returns the policy: a normalized distribution over
            actions, hidden cells (u, cx) and the value estimation
            specific to the worker.
        """
        hidden = (mask * hidden[0], mask * hidden[1])
        u, cx = self.Wrnn(z, hidden)
        hidden = (u, cx)

        # Detaching is vital, no end to end training
        goals_s = torch.stack(goals_s).detach().sum(dim=0)
        w = self.phi(goals_s)
        value_est = self.critic(u)

        u = u.reshape(u.shape[0], self.k, self.num_actions)
        a = torch.einsum("bk, bka -> ba", w, u).softmax(dim=-1)

        return a, hidden, value_est

    def intrinsic_reward(self, states_s, goals_s, masks):
        """To calculate the intrinsic reward for the Worker (Eq. 8),
        we look at the horizon C, and for each horizon step i, we
        take current state s_t minus horizon state s_{t-i} and
        calculate how similar it is to the goal at timestep s_{t-i}.

        Args:
            states_m: states_m from the Manager, a list of size C,
                    with each tensor of size B x D
            goals_m: goal vectors from the Manager, a list of size C,
                    with each tensor of size  B x D

        remember: our lists are of length c*2 + 1, with c + 1 being the exact
        middle, and we use this as our reference point for time step t.

        We hence calculate going backwards through the list of states_m, goals_m,
        state[t] == state[c+1]
        state[t] - state[t - 1], goal[t - 1]
        state[t] - state[t - 2], goal[t - 2]
        ....
        state[t] - state[t - c], goal[t - c]

        Returns:
            Intrinsic reward for the Worker
        """
        t = self.c
        r_i = torch.zeros(self.b, 1).to(self.device)
        mask = torch.ones(self.b, 1).to(self.device)

        for i in range(1, self.c + 1):
            r_i_t = d_cos(states_s[t] - states_s[t - i], goals_s[t - i]).unsqueeze(-1)
            r_i += (mask * r_i_t)

            mask = mask * masks[t - i]

        r_i = r_i.detach()
        return r_i / self.c

def feudal_loss(storage, next_v_m, next_v_s, next_v_w, args):
    """Calculate the loss for Worker and Manager,

    with timesteps T, batch size B and hidden dim D. Each of the objects
    below is a list of size T

    Args:
        rewards (B x T):
        R_intrinsic (B x T): Intrinsic rewards for the worker
        V_m (B x T): Value estimation of the worker's critic
        V_w (B x T): Value estimation of the manager's critic
        logps (B x T): log probabilities.
        entropies (B x T): action-distribution entropy per timestep.
        state_goal_cosines (B x T): Cosine distance between state and goal
                                    for the Manager.
        args: argparse arguments, needed for hyper parameters.
    """
    # Discount rewards, both of size B x T
    ret_m = next_v_m
    ret_s = next_v_s
    ret_w = next_v_w

    storage.placeholder()  # Fill ret_m, ret_w with empty vals
    for i in reversed(range(args.num_steps)):
        ret_m = storage.r[i] + args.gamma_m * ret_m * storage.m[i]
        ret_s = storage.r[i] + args.gamma_s * ret_s * storage.m[i]
        ret_w = storage.r[i] + args.gamma_w * ret_w * storage.m[i]
        storage.ret_m[i] = ret_m
        storage.ret_s[i] = ret_s
        storage.ret_w[i] = ret_w

    # Optionally, normalize the returns
    storage.normalize(['ret_w', 'ret_s', 'ret_m'])

    rewards_intrinsic, value_m, value_s, value_w, ret_w, ret_s, ret_m, logps, entropy, \
        state_goal_cosines, goal_goal_cosines = storage.stack(
            ['r_i', 'v_m', 'v_s', 'v_w', 'ret_w', 'ret_s', 'ret_m',
             'logp', 'entropy', 's_goal_cos', 'g_goal_cos'])

    # Calculate advantages, size B x T
    advantage_w = ret_w + args.alpha * rewards_intrinsic - value_w
    advantage_s = ret_s - value_s
    advantage_m = ret_m - value_m

    loss_worker = (logps * advantage_w.detach()).mean()
    loss_supervisor = (goal_goal_cosines * advantage_s.detach()).mean()
    loss_manager = (state_goal_cosines * advantage_m.detach()).mean()

    # Update the critics into the right direction
    value_w_loss = 0.5 * advantage_w.pow(2).mean()
    value_m_loss = 0.5 * advantage_m.pow(2).mean()

    entropy = entropy.mean()

    loss = - loss_worker - ((loss_manager + loss_supervisor)/2) + value_w_loss + value_m_loss - args.entropy_coef * entropy

    return loss, {'loss/total_fun_loss': loss.item(),
                  'loss/worker': loss_worker.item(),
                  'loss/manager': loss_manager.item(),
                  'loss/value_worker': value_w_loss.item(),
                  'loss/value_manager': value_m_loss.item(),
                  'worker/entropy': entropy.item(),
                  'worker/advantage': advantage_w.mean().item(),
                  'worker/intrinsic_reward': rewards_intrinsic.mean().item(),
                  'supervisor/cosines': goal_goal_cosines.mean().item(),
                  'supervisor/advantage': advantage_s.mean().item(),
                  'manager/cosines': state_goal_cosines.mean().item(),
                  'manager/advantage': advantage_m.mean().item()}