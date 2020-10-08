import numpy as np
from baselines import deepq
import os
from utils.env_utils import *
import tensorflow as tf

class FullPilotPolicy(object):
    def __init__(self, data_dir, policy_path=None):
        self.policy = None
        self.policy_path = policy_path
        self.data_dir = data_dir
        if policy_path is not None:
            self.policy = deepq.deepq.load_act(policy_path)

    def learn(self, env, max_timesteps):
        self.policy = deepq.deepq.learn(
            env,
            network='mlp',
            total_timesteps=max_timesteps,
            lr=1e-3,
            target_network_update_freq=500,
            gamma=0.99
        )
        self.policy_path = self.data_dir + '/full_pilot_policy.pkl'
        self.policy.save_act(path=self.policy_path)

    def step(self, observation):
        with tf.variable_scope('deepq', reuse=None):
            return self.policy._act(observation)[0]

