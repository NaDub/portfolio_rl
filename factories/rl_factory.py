import numpy as np
import gym

from rl.normalized_env import NormalizedEnv
from rl.evaluator import Evaluator
from rl.ddpg import DDPG
from rl.util import *
from experiments.eval import evaluate_rl
from training.train_loop import train_rl

def rl_factory(config, mode='train'):

    config.output = get_output_folder(config.output, config.env)
    if config.resume == 'default':
        config.resume = 'output/{}-run0'.format(config.env)

    env = NormalizedEnv(gym.make(config.env))

    if config.seed > 0:
        np.random.seed(config.seed)
        env.seed(config.seed)

    nb_states = env.observation_space.shape[0]
    nb_actions = env.action_space.shape[0]


    agent = DDPG(nb_states, nb_actions, config)
    evaluate = Evaluator(config.validate_episodes, 
        config.validate_steps, config.output, max_episode_length=config.max_episode_length)

    if mode == 'train':
        train_rl(config.train_iter, agent, env, evaluate, 
            config.validate_steps, config.output, max_episode_length=config.max_episode_length, debug=config.debug)

    elif mode == 'test':
        evaluate_rl(config.validate_episodes, agent, env, evaluate, config.resume,
            visualize=True, debug=config.debug)

    else:
        raise RuntimeError('undefined mode {}'.format(mode))