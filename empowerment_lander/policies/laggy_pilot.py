import numpy as np
from baselines import deepq
import tensorflow as tf

class LaggyPilotPolicy(object):
    def __init__(self, full_policy_path=None, full_policy=None):
        self.last_laggy_pilot_act = None
        self.full_policy = full_policy
        if full_policy_path is not None and self.full_policy is None:
            self.full_policy = deepq.deepq.load_act(full_policy_path)
        elif full_policy is None and self.full_policy is None:
            raise NotImplementedError

    def step(self, obs, lag_prob=0.85):
        with tf.variable_scope("deepq", reuse=None):
            if self.last_laggy_pilot_act is None or np.random.random() >= lag_prob:
                action = self.full_policy._act(obs)[0]
                self.last_laggy_pilot_act = action
            return self.last_laggy_pilot_act