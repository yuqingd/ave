from __future__ import division
import pickle
import random
import os
import math
import types
import uuid
import time
from copy import copy
from collections import defaultdict, Counter

import numpy as np
import gym
from gym import spaces, wrappers
from gym.envs.registration import register
from envs import LunarLanderEmpowerment, LunarLander
import cloudpickle
from policies import FullPilotPolicy, LaggyPilotPolicy, NoopPilotPolicy, NoisyPilotPolicy, SensorPilotPolicy, CoPilotPolicy

import tensorflow as tf

from baselines import logger
from baselines.common.schedules import LinearSchedule
from baselines import deepq
from baselines.common import models
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from baselines.deepq.deepq import ActWrapper

from experiment_utils import config
from experiment_utils.utils import query_yes_no

from baselines.common.tf_util import make_session

from matplotlib import pyplot as plt
import argparse
from utils.env_utils import *
import utils.env_utils as utils
from datetime import datetime
from pyglet.window import key as pygkey

EXP_NAME = "CopilotTraining"

def str_of_config(pilot_tol, pilot_type):
  return "{'pilot_type': '%s', 'pilot_tol': %s}" % (pilot_type, pilot_tol)


LEFT = pygkey.LEFT
RIGHT = pygkey.RIGHT
UP = pygkey.UP
DOWN = pygkey.DOWN
utils.human_agent_active = False

def key_press(key, mod):
    a = int(key)
    if a == LEFT:
        utils.human_agent_action[1] = 0
        utils.human_agent_active = True
    elif a == RIGHT:
        utils.human_agent_action[1] = 2
        utils.human_agent_active = True
    elif a == UP:
        utils.human_agent_action[0] = 1
        utils.human_agent_active = True
    elif a == DOWN:
        utils.human_agent_action[0] = 0
        utils.human_agent_active = True


def key_release(key, mod):
    a = int(key)
    if a == LEFT or a == RIGHT:
        utils.human_agent_action[1] = 1
        utils.human_agent_active = False
    elif a == UP or a == DOWN:
        utils.human_agent_action[0] = 0
        utils.human_agent_active = False

def encode_human_action(action):
    return action[0] * 3 + action[1]

def human_pilot_policy(obs):
    return encode_human_action(utils.human_agent_action)

def run_test(base_dir, empowerment, scope):
    utils.human_agent_action = init_human_action()
    utils.human_agent_active = False

    max_ep_len = 500
    n_training_episodes = 50

    co_env = LunarLanderEmpowerment(empowerment=empowerment, ac_continuous=False, pilot_policy=human_pilot_policy, pilot_is_human=True, log_file=base_dir + '/' + scope)
    co_env.render()
    co_env.unwrapped.viewer.window.on_key_press = key_press
    co_env.unwrapped.viewer.window.on_key_release = key_release
    try:
        copilot_policy = CoPilotPolicy(base_dir, policy_path='policies/pretrained_policies/{}_policy.pkl'.format(scope))
    except:
        copilot_policy = CoPilotPolicy(base_dir)
        print("No pretrained policies found")
    copilot_policy.learn(co_env, max_timesteps=max_ep_len * n_training_episodes, pilot=human_pilot_policy, pilot_is_human=True, pilot_tol=0.8, copilot_scope=scope)
    co_env.close()

    rew = copilot_policy.reward_data

    mean_rewards = np.mean(rew['rewards'])
    outcome = [r if r % 100 == 0 else 0 for r in rew['outcomes']]
    outcome_distrns = Counter(outcome)



    f = open(base_dir + "/result_" + scope + ".txt", "w")
    f.write("Empowerment: {}\n".format(empowerment))
    f.write('Mean reward: ' + str(mean_rewards))
    f.write('Outcome distribution: ' + str(outcome_distrns))
    f.close()


    return

def run_experiment(empowerment):
    base_dir = os.getcwd() + '/data/human_co'
    logger.configure(dir=base_dir, format_strs=['stdout', 'log', 'csv', 'tensorboard'])
    time.sleep(10)
    start = time.time()

    if empowerment:
        run_test(base_dir, empowerment=0.001, scope='emp')
    else:
        run_test(base_dir, empowerment=0.0, scope='noemp')

    print(time.time() - start, "Total time taken")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Empowerment Lander Testbed')

    parser.add_argument('--empowerment', action='store_true', default=False)
    args = parser.parse_args()

    run_experiment(args.empowerment)