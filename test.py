import gym
import gym_minigrid
from gym_minigrid.wrappers import *
from utils import flatten_fullview_wrapperWrapper

wrapper_fn = lambda env: flatten_fullview_wrapperWrapper(env)
# wrapper_fn = flatten_fullview_wrapper

envs = gym.vector.make('MiniGrid-FourRooms-v0', 2, wrappers=wrapper_fn)

state,_ = envs.reset()

for _ in range(1000):
    state, reward, done, _ = envs.step([envs.action_space.sample(),envs.action_space.sample()])

    if np.any(done):
        state = envs.reset()





env = flatten_fullview_wrapperWrapper(gym.make('MiniGrid-FourRooms-v0',render_mode='human'))
# env = gym.make('MiniGrid-FourRooms-v0')

state = env.reset()

for _ in range(1000):
    state, reward, done, _ = env.step(env.action_space.sample())
    env.render()
    if done:
        state = env.reset()