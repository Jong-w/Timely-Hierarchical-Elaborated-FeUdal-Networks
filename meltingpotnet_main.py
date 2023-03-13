import argparse
import torch

import numpy as np
from utils import make_envs, take_action, init_obj
from meltingpotnet import MPnets, mp_loss
from storage import Storage
from logger import Logger

import wandb

parser = argparse.ArgumentParser(description='Feudal Nets')
# GENERIC RL/MODEL PARAMETERS
parser.add_argument('--lr', type=float, default=1e-8,
                    help='learning rate')
parser.add_argument('--env-name', type=str, default='MiniGrid-FourRooms-v0',   #'MiniGrid-FourRooms-v0' 'MiniGrid-DoorKey-5x5-v0' 'MiniGrid-Empty-16x16-v0'
                    help='gym environment name')
parser.add_argument('--num-workers', type=int, default=16,
                    help='number of parallel environments to run')
# parser.add_argument('--num-steps', type=int, default=400,
#                     help='number of steps the agent takes before updating')
parser.add_argument('--num-steps', type=int, default=100,
                    help='number of steps the agent takes before updating')
parser.add_argument('--max-steps', type=int, default=int(1e6),
                    help='maximum number of training steps in total')
parser.add_argument('--cuda', type=bool, default=True,
                    help='Add cuda')
# parser.add_argument('--grad-clip', type=float, default=5.,
#                     help='Gradient clipping (recommended).')
parser.add_argument('--grad-clip', type=float, default=1.,
                    help='Gradient clipping (recommended).')
# parser.add_argument('--entropy-coef', type=float, default=0.1,
#                     help='Entropy coefficient to encourage exploration.')
parser.add_argument('--entropy-coef', type=float, default=1e-5,
                    help='Entropy coefficient to encourage exploration.')
parser.add_argument('--mlp', type=int, default=1,
                    help='toggle to feedforward ML architecture')
parser.add_argument('--whole', type=int, default=1,
                    help='use whole information of the env')
parser.add_argument('--reward-reg', type=int, default=1,
                    help='reward regulaizer')
parser.add_argument('--env-max-step', type=int, default=100,
                    help='max step for environment typically same as reward-reg')

parser.add_argument('--grid-size', type=int, default=200,
                    help='setting grid size')

# SPECIFIC FEUDALNET PARAMETERS
parser.add_argument('--time-horizon_manager', type=int, default=10,
                    help='Manager horizon (c_m)')
parser.add_argument('--time-horizon_supervisor', type=int, default=5,
                    help='Manager horizon (c_s)')
parser.add_argument('--hidden-dim-manager', type=int, default=68,
                    help='Hidden dim (d)')
parser.add_argument('--hidden-dim-supervisor', type=int, default=136,
                    help='Hidden dim (n)')
parser.add_argument('--hidden-dim-worker', type=int, default=32,
                    help='Hidden dim for worker (k)')
# parser.add_argument('--gamma-w', type=float, default=0.9,
#                     help="discount factor worker")
# parser.add_argument('--gamma-m', type=float, default=0.99,
#                     help="discount factor manager")
parser.add_argument('--gamma-w', type=float, default=0.9,
                    help="discount factor worker")
parser.add_argument('--gamma-s', type=float, default=0.95,
                    help="discount factor supervisor")
parser.add_argument('--gamma-m', type=float, default=0.99,
                    help="discount factor manager")
parser.add_argument('--alpha', type=float, default=0.5,
                    help='Intrinsic reward coefficient in [0, 1]')
parser.add_argument('--eps', type=float, default=float(1),
                    help='Random Gausian goal for exploration')
parser.add_argument('--dilation_manager', type=int, default=10,
                    help='Dilation parameter for manager LSTM.')
parser.add_argument('--dilation_supervisor', type=int, default=5,
                    help='Dilation parameter for manager LSTM.')


# EXPERIMENT RELATED PARAMS
parser.add_argument('--run-name', type=str, default='baseline',
                    help='run name for the logger.')
parser.add_argument('--seed', type=int, default=0,
                    help='reproducibility seed.')

args = parser.parse_args()


def experiment(args):

    save_steps = list(torch.arange(0, int(args.max_steps),int(args.max_steps) // 10).numpy())

    # logger = Logger(args.run_name, args)
    logger = Logger(args.env_name, "Feudal_Nets", args)
    cuda_is_available = torch.cuda.is_available() and args.cuda
    device = torch.device("cuda" if cuda_is_available else "cpu")
    args.device = device

    torch.manual_seed(args.seed)
    if cuda_is_available:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    envs = make_envs(args.env_name, args.num_workers, args.whole, args.reward_reg, args.env_max_step, args.grid_size)
    MPnet = MPnets(
        num_workers=args.num_workers,
        input_dim=envs.observation_space.shape,
        hidden_dim_manager=args.hidden_dim_manager,
        hidden_dim_supervisor=args.hidden_dim_supervisor,
        hidden_dim_worker=args.hidden_dim_worker,
        n_actions=3,
        time_horizon_manager=args.time_horizon_manager,
        time_horizon_supervisor=args.time_horizon_supervisor,
        dilation_manager=args.dilation_manager,
        dilation_supervisor=args.dilation_supervisor,
        device=device,
        mlp=args.mlp,
        args=args,
        whole=args.whole)

    #optimizer = torch.optim.RMSprop(MPnet.parameters(), lr=args.lr, alpha=0.99, eps=1e-5)
    optimizer = torch.optim.NAdam(MPnet.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-05, weight_decay=0, momentum_decay=0.004, foreach=None)

    goals_m, states_m, goals_s, states_s, masks  = MPnet.init_obj()

    x = envs.reset()
    step = 0
    eps = args.eps
    while step < args.max_steps:

        # Detaching LSTMs and goals
        MPnet.repackage_hidden()
        goals_m = [g.detach() for g in goals_m]
        goals_s = [g.detach() for g in goals_s]
        storage = Storage(size=args.num_steps,
                          keys=['r', 'r_i', 'v_w', 'v_s', 'v_m', 'logp', 'entropy',
                                's_goal_cos', 'g_goal_cos','mask', 'ret_w', 'ret_s', 'ret_m', 'adv_m', 'adv_w'])

        if eps > 0.01:
            eps = eps * 0.999
        else:
            eps = eps

        for _ in range(args.num_steps):
            action_dist, goals_m, states_m, value_m, goals_s, states_s, value_s, value_w \
                = MPnet(x, goals_m, states_m, goals_s, states_s, masks[-1], eps=eps)

            # Take a step, log the info, get the next state
            action, logp, entropy = take_action(action_dist)
            x, reward, done, info = envs.step(action)
            logger.log_episode(info, step)

            mask = torch.FloatTensor(1 - done).unsqueeze(-1).to(args.device)
            masks.pop(0)
            masks.append(mask)

            storage.add({
                'r': torch.FloatTensor(reward).unsqueeze(-1).to(device),
                'r_i': MPnet.intrinsic_reward(states_s, goals_s, masks),
                'v_w': value_w,
                'v_s': value_s,
                'v_m': value_m,
                'logp': logp.unsqueeze(-1),
                'entropy': entropy.unsqueeze(-1),
                's_goal_cos': MPnet.state_goal_cosine(states_s, goals_s, masks),
                #'g_goal_cos': MPnet.goal_goal_cosine(goals_m, goals_s,  masks),
                'g_goal_cos': MPnet.goal_goal_cosine(goals_m, goals_s, masks),
                'm': mask
            })
            reward_list = []
            length_list = []
            for _i in range(len(done)):
                if done[_i]:
                    reward_list.append(info[_i]['returns/episodic_reward'])
                    length_list.append(info[_i]['returns/episodic_length'])
            wandb.log(
                {"training/episode/reward": np.mean(reward_list),
                 "training/episode/length": np.mean(length_list)})

            step += args.num_workers

        with torch.no_grad():
            _, _, _, next_v_m, _, _, next_v_s, next_v_w = MPnet(
                x, goals_m, states_m, goals_s, states_s, mask, eps,  save=False)
            next_v_m = next_v_m.detach()
            next_v_s = next_v_s.detach()
            next_v_w = next_v_w.detach()

        optimizer.zero_grad()
        loss, loss_dict = mp_loss(storage, next_v_m, next_v_s, next_v_w, args, eps)
        wandb.log(loss_dict)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(MPnet.parameters(), args.grad_clip)
        optimizer.step()
        logger.log_scalars(loss_dict, step)

        if len(save_steps) > 0 and step > save_steps[0]:
            torch.save({
                'model': MPnet.state_dict(),
                'args': args,
                'processor_mean': MPnet.preprocessor.rms.mean,
                'optim': optimizer.state_dict()},
                f'models/{args.env_name}_{args.run_name}_MPnetv2_steps={step}.pt')
            save_steps.pop(0)


    envs.close()
    torch.save({
        'model': MPnet.state_dict(),
        'args': args,
        'processor_mean': MPnet.preprocessor.rms.mean,
        'optim': optimizer.state_dict()},
        f'models/{args.env_name}_{args.run_name}_steps={step}.pt')

def main(args):
    run_name = args.run_name
    lr_starter = 1e-6

    for seed in range(3):
        args.lr *= 10
        wandb.init(project="mpnets",
        config=args.__dict__
        )
        args.seed = seed
        args.run_name = f"{run_name}_seed={seed}"
        wandb.run.name = f"fun_seed={seed}"
        experiment(args)
        wandb.finish()


if __name__ == '__main__':
    main(args)