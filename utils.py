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

class ImgObsWrapper(gym.core.ObservationWrapper):
    """
    Use the image as the only observation output, no language/mission.
    """
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space.spaces['image']

    def observation(self, obs):
        return obs['image']

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


def make_envs(env_name, num_envs, seed=0, partial=1):
    env_ = gym.make(env_name)
    is_atari = hasattr(gym.envs, 'atari') and isinstance(
            env_.unwrapped, gym.envs.atari.atari_env.AtariEnv)

    if is_atari:
        wrapper_fn = atari_wrapper
    elif partial:
        wrapper_fn = basic_birdview_wrapper
    else:
        wrapper_fn = basic_wrapper

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