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
from datetime import datetime
from pyglet.window import key as pygkey

EXP_NAME = "CopilotTraining"

def str_of_config(pilot_tol, pilot_type):
  return "{'pilot_type': '%s', 'pilot_tol': %s}" % (pilot_type, pilot_tol)


def run_ep(policy, env, max_ep_len, render=False, pilot_is_human=False):
    if pilot_is_human:
        global human_agent_action
        global human_agent_active
        human_agent_action = init_human_action()
        human_agent_active = False

    obs = env.reset()
    done = False
    totalr = 0.
    trajectory = None
    actions = []
    for step_idx in range(max_ep_len + 1):
        if done:
            trajectory = info['trajectory']
            break
        if pilot_is_human:
            action = policy(obs[None, :])
        else:
            action = policy.step(obs[None, :])
        obs, r, done, info = env.step(action)
        actions.append(action)
        if render:
            env.render()
        totalr += r
    outcome = r if r % 100 == 0 else 0
    time.sleep(2)
    return totalr, outcome, trajectory, actions


def run_ep_copilot(policy, env, max_ep_len, pilot, pilot_tol, render=False, pilot_is_human=False):
    if pilot_is_human:
        global human_agent_action
        global human_agent_active
        human_agent_action = init_human_action()
        human_agent_active = False
    obs = env.reset()
    done = False
    totalr = 0.
    trajectory = None
    actions = []
    pilot_actions = np.zeros((env.num_concat * env.act_dim))
    for step_idx in range(max_ep_len + 1):
        if done:
            trajectory = info['trajectory']
            break
        action, pilot_actions = policy.step(obs[None, :], pilot, pilot_tol, pilot_actions, pilot_is_human=pilot_is_human)
        obs, r, done, info = env.step(action)
        actions.append(action)
        if render:
            env.render()
        totalr += r
    outcome = r if r % 100 == 0 else 0
    return totalr, outcome, trajectory, actions

LEFT = pygkey.LEFT
RIGHT = pygkey.RIGHT
UP = pygkey.UP
DOWN = pygkey.DOWN
def key_press(key, mod):
    global human_agent_action
    global human_agent_active
    a = int(key)
    if a == LEFT:
        human_agent_action[1] = 0
        human_agent_active = True
    elif a == RIGHT:
        human_agent_action[1] = 2
        human_agent_active = True
    elif a == UP:
        human_agent_action[0] = 1
        human_agent_active = True
    elif a == DOWN:
        human_agent_action[0] = 0
        human_agent_active = True


def key_release(key, mod):
    global human_agent_action
    global human_agent_active
    a = int(key)
    if a == LEFT or a == RIGHT:
        human_agent_action[1] = 1
        human_agent_active = False
    elif a == UP or a == DOWN:
        human_agent_action[0] = 0
        human_agent_active = False


def encode_human_action(action):
    return action[0] * 3 + action[1]

def human_pilot_policy(obs):
    global human_agent_action
    return encode_human_action(human_agent_action)

def run_test(base_dir, copilot, empowerment, scope):

    max_ep_len = 500
    n_training_episodes = 6

    if copilot:

        co_env = LunarLanderEmpowerment(empowerment=empowerment, ac_continuous=False, pilot_policy=human_pilot_policy, pilot_is_human=True)
        co_env.render()
        co_env.unwrapped.viewer.window.on_key_press = key_press
        co_env.unwrapped.viewer.window.on_key_release = key_release
        copilot_policy = CoPilotPolicy(base_dir)
        copilot_policy.learn(co_env, max_timesteps=max_ep_len * n_training_episodes, pilot=human_pilot_policy, pilot_is_human=True, pilot_tol=0.2, copilot_scope=scope)
        co_env.close()

        rew = copilot_policy.reward_data

        mean_rewards = np.mean(rew['rewards'])
        outcome = [r if r % 100 == 0 else 0 for r in rew['outcomes']]
        outcome_distrns = Counter(outcome)



    else:

        env = LunarLanderEmpowerment(empowerment=0.0, ac_continuous=False, pilot_is_human=True)
        env.render()
        env.unwrapped.viewer.window.on_key_press = key_press
        env.unwrapped.viewer.window.on_key_release = key_release
        outcomes = []
        rews = []
        for _ in range(n_training_episodes * 4):
            rew, outcome, _, __ = run_ep(human_pilot_policy, env, render=True, max_ep_len=max_ep_len, pilot_is_human=True)
            rews.append(rew)
            outcomes.append(outcome)

        mean_rewards = np.mean(rews)
        outcome_distrns = Counter(outcomes)
        env.close()

    f = open(base_dir + "/result_" + scope + ".txt", "w")
    f.write("Empowerment: {}\n".format(empowerment))
    f.write('Mean reward: ' + str(mean_rewards))
    f.write('Outcome distribution: ' + str(outcome_distrns))
    f.close()


    return

def run_experiment(empowerment, human_name, trial_type):
    base_dir = os.getcwd() + '/data/human_' + human_name
    logger.configure(dir=base_dir, format_strs=['stdout', 'log', 'csv', 'tensorboard'])

    max_ep_len = 500
    n_training_episodes = 7
    start = time.time()

    env = LunarLanderEmpowerment(empowerment=0.0, ac_continuous=False)
    env.render()
    env.unwrapped.viewer.window.on_key_press = key_press
    env.unwrapped.viewer.window.on_key_release = key_release

    time.sleep(10)
    intro_rollouts = []
    for _ in range(20):
        intro_rollouts.append(run_ep(human_pilot_policy, env, render=True, max_ep_len=max_ep_len, pilot_is_human=True))

    env.close()

    if trial_type:
        run_test(base_dir, copilot=True, empowerment=0.001, scope='emp1')
        run_test(base_dir, copilot=True, empowerment=0.0, scope='noemp1')
        run_test(base_dir, copilot=False, empowerment=0.0, scope='nocopilot1')
        run_test(base_dir, copilot=True, empowerment=0.001, scope='emp2')
        run_test(base_dir, copilot=True, empowerment=0.0, scope='noemp2')
        run_test(base_dir, copilot=False, empowerment=0.0, scope='nocopilot2')
    else:
        run_test(base_dir,  copilot=True,empowerment=0.0, scope='noemp1')
        run_test(base_dir, copilot=True, empowerment=0.001, scope='emp1')
        run_test(base_dir, copilot=False, empowerment=0.0, scope='nocopilot1')
        run_test(base_dir, copilot=True, empowerment=0.0, scope='noemp2')
        run_test(base_dir,  copilot=True, empowerment=0.001, scope='emp2')
        run_test(base_dir, copilot=False, empowerment=0.0, scope='nocopilot2')

    print(time.time() - start, "Total time taken")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Empowerment Lander Testbed')

    parser.add_argument('--exp_title', type=str, default='', help='Title for experiment')
    parser.add_argument('--name', type=str, help='Person')
    parser.add_argument('--trial_type', default=1, type=bool)

    parser.add_argument('--empowerment', type=float, default=0.001,
                        help='Empowerment coefficient')
    args = parser.parse_args()

    run_experiment(empowerment=args.empowerment, human_name=args.name, trial_type=args.trial_type)