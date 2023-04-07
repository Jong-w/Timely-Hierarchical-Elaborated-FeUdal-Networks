###############################################
import gym
import gym_minigrid
from gym_minigrid.wrappers import *
from utils import *
import numpy as np

wrapper_fn = lambda env: flatten_fullview_wrapperWrapper(env, reward_reg=5000, env_max_step=1000)
env_name = 'MiniGrid-FourRooms-v0'
# env = gym.make(env_name, wrappers=wrapper_fn)
env = wrapper_fn(gym.make(env_name,render_mode='human'))

env.reset(seed=73060)
env.render()
env.close()


################################

distlen=[]
seed = 70000
while True:
	obs, info = env.reset(seed=seed)
	x0,y0 = np.where(obs[:,:,0]==8)
	x1,y1 = np.where(obs[:,:,0]==10)

	# squeeze x0,y0,x1,y1
	x0,y0,x1,y1 = x0[0],y0[0],x1[0],y1[0]

	# get distance from x0,y0 to x1,y1
	dist = np.sqrt((x0-x1)**2 + (y0-y1)**2)

	# cal maxdistance
	maxdistance = np.sqrt((1 - 17) ** 2 + (1 - 17) ** 2)

	# print dist and seed with description
	print(f"dist: {dist}, seed: {seed}")

	# print how dist is different from maxdistance
	print(f"dist - maxdistance: {dist - maxdistance}")

	# print location x0,y0 and x1,y1
	print(f"x0: {x0}, y0: {y0}\nx1: {x1}, y1: {y1}")

	# line that separates each seed
	print("=====================================")

	# append dist and seed to distlen
	distlen.append([dist,seed])

	seed += 1
	env.close()
	if dist == np.sqrt((1-17)**2 + (1-17)**2):
		break
