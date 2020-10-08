import numpy as np
from baselines import deepq

class SensorPilotPolicy(object):
    def __init__(self, full_policy_path=None, full_policy=None):
        self.full_policy = full_policy
        if full_policy_path is not None and self.full_policy is None:
            self.full_policy = deepq.deepq.load_act(full_policy_path)
        elif full_policy is None and self.full_policy is None:
            raise NotImplementedError

    def step(self, obs, thresh=0.1):
        d = obs[0, 8] - obs[0, 0]  # horizontal dist to helipad
        if d < -thresh:
            return 0
        elif d > thresh:
            return 2
        else:
            return 1
