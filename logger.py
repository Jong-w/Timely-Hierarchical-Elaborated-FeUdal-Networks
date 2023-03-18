import logging
import os
import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, run_name, net_name, args):
        dt = datetime.now()
        # didn't figure out how to easily remove seconds from datetime
        self.log_name = 'logs/' + run_name + "["+ net_name+"]"
        self.start_time = time.time()
        self.n_eps = 0

        if not os.path.exists('logs'):
            os.makedirs('logs')
            os.makedirs('models')

        self.writer = SummaryWriter(self.log_name)

        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f'{self.log_name}.log'),
                ],
            datefmt='%Y/%m/%d %I:%M:%S %p')
        logging.info(args)


    def log_scalars(self, scalar_dict, step):
        for key, val in scalar_dict.items():
            self.writer.add_scalar(key, val, step)

    def log_episode(self, info, step):
        for episode_dict in info:
            # check if episode_dict has key 'returns/episodic_reward' or not
            # if it has, then log it to tensorboard
            # if it doesn't, then don't log it to tensorboard
            # you can check if a key exists in a dictionary by using the following syntax:
            if not 'final_info' in episode_dict:
                if episode_dict['returns/episodic_reward'] is not None:
                    self.n_eps += 1
                    self.log_scalars(episode_dict, step)
                    reward = episode_dict['returns/episodic_reward']
                    length = episode_dict['returns/episodic_length']
                    time_expired = (time.time() - self.start_time) / 60 / 60

                    logging.info(f"> ep = {self.n_eps} | total steps = {step}"
                                 f" | reward = {reward} | length = {length}"
                                 f" | hours = {time_expired:.3f}")
            else:
                if episode_dict['final_info'] is not None:
                    if episode_dict['final_info']['returns/episodic_reward'] is not None:
                        self.n_eps += 1
                        self.log_scalars(episode_dict['final_info'], step)
                        reward = episode_dict['final_info']['returns/episodic_reward']
                        length = episode_dict['final_info']['returns/episodic_length']
                        time_expired = (time.time() - self.start_time) / 60 / 60

                        logging.info(f"> ep = {self.n_eps} | total steps = {step}"
                                     f" | reward = {reward} | length = {length}"
                                     f" | hours = {time_expired:.3f}")