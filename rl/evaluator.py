import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat

from rl.util import *

class Evaluator(object):

    def __init__(self, num_episodes, interval, save_path='', max_episode_length=None):
        self.num_episodes = num_episodes
        self.max_episode_length = max_episode_length
        self.interval = interval
        self.save_path = save_path
        self.results = np.array([]).reshape(num_episodes,0)

    def __call__(self, env, policy, debug=False, visualize=False, save=True):

        self.is_training = False
        obs = None
        result = []

        for episode in range(self.num_episodes):

            # reset at the start of episode
            obs = env.reset()
            episode_steps = 0
            episode_reward = 0.
                
            assert obs is not None

            # start episode
            terminated = False
            while not terminated:
                # basic operation, action ,reward, blablabla ...
                action = policy(obs)

                obs, reward, terminated, truncated, info = env.step(action) #obs, reward, terminated, truncated, info
                if self.max_episode_length and episode_steps >= self.max_episode_length -1:
                    terminated = True
                
                if visualize:
                    env.render(mode='human')

                # update
                episode_reward += reward
                episode_steps += 1

            if debug: prYellow('[Evaluate] #Episode{}: episode_reward:{}'.format(episode,episode_reward))
            result.append(episode_reward)

        result = np.array(result).reshape(-1,1)
        self.results = np.hstack([self.results, result])

        if save:
            self.save_results('{}/validate_reward'.format(self.save_path))
        return np.mean(result)

    def save_results(self, fn):

        y = np.mean(self.results, axis=0)
        error=np.std(self.results, axis=0)
                    
        x = range(0,self.results.shape[1]*self.interval,self.interval)
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        plt.xlabel('Timestep')
        plt.ylabel('Average Reward')
        ax.errorbar(x, y, yerr=error, fmt='-o')
        plt.savefig(fn+'.png')
        savemat(fn+'.mat', {'reward':self.results})