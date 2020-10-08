import numpy as np
from copy import copy

human_agent_action = None
human_agent_active = False

def disc_to_cont(action):
    # Discrete action space:
    #0 - right only
    #1 - no op
    #2 - left only
    #3 - up + right
    #4 - up only
    #5 - up and left
    throttle_mag = 0.75
    if type(action) == np.ndarray and len(action) == 2:
        return action
    # main engine
    if action < 3:
        m = -throttle_mag
    elif action < 6:
        m = throttle_mag
    else:
        raise ValueError
    # steering
    if action % 3 == 0:
        s = -throttle_mag
    elif action % 3 == 1:
        s = 0
    else:
        s = throttle_mag
    return np.array([m, s])

def mask_helipad(obs, replace=0):
  obs = copy(obs)
  if len(obs.shape) == 1:
    obs[8] = replace
  else:
    obs[:, 8] = replace
  return obs

def traj_mask_helipad(traj):
  return [mask_helipad(obs) for obs in traj]

def onehot_encode(i, n=6):
    x = np.zeros(n)
    x[i] = 1
    return x

def onehot_decode(x):
    l = np.nonzero(x)[0]
    assert len(l) == 1
    return l[0]

def init_human_action():
    return [0,1] #no action

