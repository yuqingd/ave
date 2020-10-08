import numpy as np
from baselines import deepq

class NoopPilotPolicy(object):
    def __init__(self, full_policy_path=None, full_policy=None):
        self.full_policy = full_policy
        if full_policy_path is not None and self.full_policy is None:
            try:
                self.full_policy = deepq.deepq.load_act(full_policy_path)
            except:
                print("Policy not found")
        elif full_policy is None and self.full_policy is None:
            raise NotImplementedError

    def step(self, obs):
        return 1