from rl.ddpg import DDPG
from rl.evaluator import Evaluator
from rl.normalized_env import NormalizedEnv
from rl.util import get_output_folder
from experiments.eval import evaluate_rl
from training.train_loop import train_rl

import gym
import numpy as np
import os

class RLUseCase:
    def __init__(self, config):
        self.config = config
        self.env = None
        self.agent = None
        self.evaluator = None

    def _setup(self):
        gym.undo_logger_setup()
        
        # Output path
        self.config.output = get_output_folder(self.config.output, self.config.env)
        if self.config.resume == 'default':
            self.config.resume = os.path.join(self.config.output, 'run0')

        # Environnement
        self.env = NormalizedEnv(gym.make(self.config.env))

        if self.config.seed > 0:
            np.random.seed(self.config.seed)
            self.env.seed(self.config.seed)

        # Dimensions
        nb_states = self.env.observation_space.shape[0]
        nb_actions = self.env.action_space.shape[0]

        # Agent and Evaluator
        self.agent = DDPG(nb_states, nb_actions, self.config)
        self.evaluator = Evaluator(
            self.config.validate_episodes,
            self.config.validate_steps,
            self.config.output,
            max_episode_length=self.config.max_episode_length
        )

    def train(self):
        self._setup()
        train_rl(
            self.config.train_iter,
            self.agent,
            self.env,
            self.evaluator,
            self.config.validate_steps,
            self.config.output,
            max_episode_length=self.config.max_episode_length,
            debug=self.config.debug
        )

    def test(self):
        self._setup()
        evaluate_rl(
            self.config.validate_episodes,
            self.agent,
            self.env,
            self.evaluator,
            self.config.resume,
            visualize=True,
            debug=self.config.debug
        )