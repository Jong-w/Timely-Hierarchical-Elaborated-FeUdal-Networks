"""
This file is filled with miscelaneous classes and functions.
"""
import gym
from gym.wrappers import AtariPreprocessing, TransformReward
from gym_minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper, FullyObsWrapper, RGBImgObsWrapper, RGBImgObsWrapper
###############################################
import gym
import gym_minigrid
from gym_minigrid.wrappers import *
from gym_minigrid.window import Window
###############################################
import torch
import numpy as np
import cv2

from torch.distributions import Categorical


class ReturnWrapper(gym.Wrapper):
    #######################################################################
    # Copyright (C) 2020 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
    # Permission given to modify the code as long as you keep this        #
    # declaration at the top                                              #
    #######################################################################
    def __init__(self, env):
        super().__init__(env)
        self.total_rewards = 0
        self.steps = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # TODO: reward가 왜 0또는 1이 아니지?
        reward = np.ceil(reward)

        self.total_rewards += reward
        self.steps += 1
        if done:
            info['returns/episodic_reward'] = self.total_rewards
            info['returns/episodic_length'] = self.steps
            self.total_rewards = 0
            self.steps = 0
        else:
            info['returns/episodic_reward'] = None
            info['returns/episodic_length'] = None
        return obs, reward, done, info

class ReturnWrapper_wargs(ReturnWrapper):
    #######################################################################
    # Copyright (C) 2020 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
    # Permission given to modify the code as long as you keep this        #
    # declaration at the top                                              #
    #######################################################################
    def __init__(self, env, reward_reg=5000):
        super().__init__(env)
        self.total_rewards = 0
        self.steps = 0
        self.multiplier = reward_reg

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward = np.ceil(reward)*self.multiplier/(self.steps+1)

        self.total_rewards += reward
        self.steps += 1
        if done:
            info['returns/episodic_reward'] = self.total_rewards
            info['returns/episodic_length'] = self.steps
            self.total_rewards = 0
            self.steps = 0
        else:
            info['returns/episodic_reward'] = None
            info['returns/episodic_length'] = None
        return obs, reward, done, info


class FlattenWrapper(gym.core.ObservationWrapper):
    """
    Fully observable gridworld using a compact grid encoding
    """

    def __init__(self, env):
        super().__init__(env)
        imgSpace = env.observation_space.spaces['image']
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=imgSpace.shape,
            dtype='uint8'
        )
    #
    # def step(self, action):
    #     obs, reward, done, info = self.env.step(action)
    #
    #     env = self.unwrapped
    #     tup = (tuple(env.agent_pos), env.agent_dir, action)
    #
    #     # Get the count for this (s,a) pair
    #     pre_count = 0
    #     if tup in self.counts:
    #         pre_count = self.counts[tup]
    #
    #     # Update the count for this (s,a) pair
    #     new_count = pre_count + 1
    #     self.counts[tup] = new_count
    #
    #     bonus = 1 / math.sqrt(new_count)
    #     reward += bonus
    #
    #     return obs, reward, done, info
    def observation(self, obs):
        env = self.unwrapped
        full_grid = env.grid.encode()
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array([
            OBJECT_TO_IDX['agent'],
            COLOR_TO_IDX['red'],
            env.agent_dir
        ])

        return full_grid

def flatten_fullview_wrapperWrapper(env, reward_reg=5000, env_max_step=5000):
    env.max_steps = reward_reg
    env = FullyObsWrapper(env)
    env = FlattenWrapper(env)
    env = ReturnWrapper_wargs(env, reward_reg=reward_reg)
    return env

def flatten_fullview_wrapper(env):
    env = FullyObsWrapper(env)
    env = FlattenWrapper(env)
    env = ReturnWrapper(env)
    return env

def basic_birdview_wrapper(env):
    """Use this as a wrapper only for cartpole etc."""
    # env = RGBImgPartialObsWrapper(env)
    env = RGBImgObsWrapper(env)
    env = ImgObsWrapper(env)
    env = ReturnWrapper(env)
    # env = TransformReward(env, lambda r: np.clip(r, -1, 1))
    return env

def basic_wrapper(env):
    """Use this as a wrapper only for cartpole etc."""
    env = ImgObsWrapper(env)
    env = ReturnWrapper(env)
    env = TransformReward(env, lambda r: np.clip(r, -1, 1))
    return env


def atari_wrapper(env):
    # This is substantially the same CNN as in (Mnih et al., 2016; 2015),
    # the only difference is that in the pre-processing stage
    # we retain all colour channels.
    env = AtariPreprocessing(env, grayscale_obs=False, scale_obs=True)
    env = ReturnWrapper(env)
    env = TransformReward(env, lambda r: np.sign(r))
    return env


def make_envs(env_name, num_envs, seed=0, partial=1, reward_reg=5000, env_max_step=5000):
    env_ = gym.make(env_name)

    wrapper_fn = lambda env: flatten_fullview_wrapperWrapper(env, reward_reg=reward_reg, env_max_step=env_max_step)
    # wrapper_fn = flatten_fullview_wrapper

    envs = gym.vector.make(env_name, num_envs, wrappers=wrapper_fn)
    envs.seed(seed)
    return envs


def take_action(a):
    dist = Categorical(a)
    action = dist.sample()
    logp = dist.log_prob(action)
    entropy = dist.entropy()
    return action.cpu().detach().numpy(), logp, entropy


def init_hidden(n_workers, h_dim, device, grad=False):
    return (torch.zeros(n_workers, h_dim, requires_grad=grad).to(device),
            torch.zeros(n_workers, h_dim, requires_grad=grad).to(device))


def init_obj(n_workers, h_dim, c, device):
    goals = [torch.zeros(n_workers, h_dim, requires_grad=True).to(device)
             for _ in range(c)]
    states = [torch.zeros(n_workers, h_dim).to(device) for _ in range(c)]
    return goals, states


def weight_init(layer):
    if type(layer) == torch.nn.modules.conv.Conv2d or \
            type(layer) == torch.nn.Linear:
        torch.nn.init.orthogonal_(layer.weight.data)
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias.data, 0)


if __name__=="__main__":
    wrapper_fn = basic_birdview_wrapper
    env = gym.vector.make('MiniGrid-FourRooms-v0', wrappers=basic_birdview_wrapper)
    x = env.reset()
    cv2.imwrite('one_fig.png', x.reshape((120, 120, 3)))