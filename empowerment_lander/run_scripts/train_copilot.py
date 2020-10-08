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

import doodad as dd
import doodad.mount as mount
import doodad.easy_sweep.launcher as launcher
import multiprocessing
from doodad.easy_sweep.hyper_sweep import run_sweep_doodad

from experiment_utils import config
from experiment_utils.utils import query_yes_no

from baselines.common.tf_util import make_session

from matplotlib import pyplot as plt
import argparse
from utils.env_utils import *
from datetime import datetime

EXP_NAME = "CopilotTraining"

def str_of_config(pilot_tol, pilot_type):
  return "{'pilot_type': '%s', 'pilot_tol': %s}" % (pilot_type, pilot_tol)


def run_ep(policy, env, max_ep_len, render=False, pilot_is_human=False):
    obs = env.reset()
    done = False
    totalr = 0.
    trajectory = None
    actions = []
    for step_idx in range(max_ep_len + 1):
        if done:
            trajectory = info['trajectory']
            break
        action = policy.step(obs[None, :])
        obs, r, done, info = env.step(action)
        actions.append(action)
        if render:
            env.render()
        totalr += r
    outcome = r if r % 100 == 0 else 0
    return totalr, outcome, trajectory, actions


def run_ep_copilot(policy, env, max_ep_len, pilot, pilot_tol, render=False, pilot_is_human=False):
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
        action, pilot_actions = policy.step(obs[None, :], pilot, pilot_tol, pilot_actions)
        obs, r, done, info = env.step(action)
        actions.append(action)
        if render:
            env.render()
        totalr += r
    outcome = r if r % 100 == 0 else 0
    return totalr, outcome, trajectory, actions


def run_experiment(empowerment, exp_title, seed):
    now = datetime.now()

    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    base_dir = os.path.join(config.DOCKER_MOUNT_DIR, EXP_NAME)
#    base_dir = os.getcwd() + '/data/' + exp_title + now.strftime("%m-%d-%Y-%H-%M-%S") + "_" + empowerment #
    logger.configure(dir=base_dir, format_strs=['stdout', 'log', 'csv', 'tensorboard'])

    f = open(base_dir + "/config.txt", "w")
    f.write("Empowerment: {}\n".format(empowerment))
    f.write("Num concat: 20\n")
    f.write("Seed: {}\n".format(seed))
    f.write("No scale by height\n")
    f.write("Keep main engine from pilot")
    f.close()

    max_ep_len = 1000
    n_training_episodes = 500

    env = LunarLanderEmpowerment(empowerment=0.0, ac_continuous=False)

    max_timesteps = max_ep_len * n_training_episodes
    full_pilot_policy = FullPilotPolicy(base_dir)
    full_pilot_policy.learn(env, max_timesteps)
    laggy_pilot_policy = LaggyPilotPolicy(base_dir, full_policy=full_pilot_policy.policy)
    noisy_pilot_policy = NoisyPilotPolicy(base_dir, full_policy=full_pilot_policy.policy)
    noop_pilot_policy = NoopPilotPolicy(base_dir, full_policy=full_pilot_policy.policy)
    sensor_pilot_policy = SensorPilotPolicy(base_dir, full_policy=full_pilot_policy.policy)
    sim_pilots = [full_pilot_policy, laggy_pilot_policy, noisy_pilot_policy, noop_pilot_policy, sensor_pilot_policy]

    pilot_names = ['full', 'laggy', 'noisy', 'noop', 'sensor']
    n_eval_eps = 100

    pilot_evals = [
        list(zip(*[run_ep(sim_policy, env, render=False, max_ep_len=max_ep_len) for _ in
                   range(n_eval_eps)])) for
        sim_policy in sim_pilots]

    mean_rewards = [np.mean(pilot_eval[0]) for pilot_eval in pilot_evals]
    outcome_distrns = [Counter(pilot_eval[1]) for pilot_eval in pilot_evals]

    f = open(base_dir + "/base.txt", "w")
    f.write('\n'.join([str(x) for x in zip(pilot_names, mean_rewards, outcome_distrns)]))
    f.close()

    pilot_tol_of_id = {
        'noop': 0,
        'laggy': 0.7,
        'noisy': 0.3,
        'sensor': 0.1
    }

    copilot_of_training_pilot = {}

    for training_pilot_id, training_pilot_tol in pilot_tol_of_id.items():
        training_pilot_policy = eval('%s_pilot_policy' % training_pilot_id)
        config_kwargs = {
            'pilot_policy': training_pilot_policy,
            'pilot_tol': training_pilot_tol,
            'reuse': True,
            'copilot_scope': 'co_deepq_' + training_pilot_id
        }
        print(training_pilot_id)
        co_env = LunarLanderEmpowerment(empowerment=empowerment, ac_continuous=False, **config_kwargs)
        copilot_policy = CoPilotPolicy(base_dir)
        copilot_policy.learn(co_env, max_timesteps=max_ep_len * n_training_episodes, **config_kwargs)
        copilot_of_training_pilot[training_pilot_id] = copilot_policy

    cross_evals={}

    for training_pilot_id, training_pilot_tol in pilot_tol_of_id.items():
        # load pretrained copilot
        training_pilot_policy = eval('%s_pilot_policy' % training_pilot_id)
        config_kwargs = {
            'pilot_policy': training_pilot_policy,
            'pilot_tol': training_pilot_tol,
            'reuse': True
        }

        # evaluate copilot with different pilots
        for eval_pilot_id, eval_pilot_tol in pilot_tol_of_id.items():
            eval_pilot_policy = eval('%s_pilot_policy' % eval_pilot_id)
            copilot_policy = copilot_of_training_pilot[training_pilot_id]

            co_env_eval = LunarLanderEmpowerment(empowerment=0, ac_continuous=False, pilot_policy=eval_pilot_policy)
            cross_evals[(training_pilot_id, eval_pilot_id)] = list(zip(*[run_ep_copilot(copilot_policy, co_env_eval, pilot=eval_pilot_policy, pilot_tol=eval_pilot_tol, render=False, max_ep_len=max_ep_len)[:2] for _ in
                                                               range(n_eval_eps)]))

    f = open(base_dir + "/cross_eval.txt", "w")

    for key, value in cross_evals.items():
        mean_rewards = np.mean(value[0])
        outcome_distrns = Counter(value[1])

        f.write('\n Training pilot: {}, eval pilot: {}'.format(key[0], key[1]))
        f.write(' Mean reward: ' + str(mean_rewards))
        f.write(' Outcome distribution: '+ str(outcome_distrns))

    f.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Empowerment Lander Testbed')

    parser.add_argument('--exp_title', type=str, default='', help='Title for experiment')
    parser.add_argument('--mode', type=str, default='local',
                        help='Mode for running the experiments - local: runs on local machine, '
                             'ec2: runs on AWS ec2 cluster (requires a proper configuration file)')
    parser.add_argument('--empowerment', type=float, default=100.0,
                        help='Empowerment coefficient')
    parser.add_argument('--seed', type=int, default=1, help='Seed')
    args = parser.parse_args()

    local_mount = mount.MountLocal(local_dir=config.BASE_DIR, pythonpath=True)
    docker_mount_point = os.path.join(config.DOCKER_MOUNT_DIR, EXP_NAME)

    sweeper = launcher.DoodadSweeper([local_mount], docker_img=config.DOCKER_IMAGE,
                                     docker_output_dir=docker_mount_point,
                                     local_output_dir=os.path.join(config.DATA_DIR, 'local', EXP_NAME))
    sweeper.mount_out_s3 = mount.MountS3(s3_path='', mount_point=docker_mount_point, output=True)

    if args.mode == 'ec2':
        if query_yes_no("Continue?"):
            sweeper.run_sweep_ec2(run_experiment, {'empowerment':[0.001], 'exp_title': [''], 'seed':[1]}, bucket_name=config.S3_BUCKET_NAME,
                                  instance_type='c4.2xlarge',
                                  region='us-west-1', s3_log_name=EXP_NAME, add_date_to_logname=True)
    elif args.mode == 'local_docker':
            mode_docker = dd.mode.LocalDocker(
                image=sweeper.image,
            )
            run_sweep_doodad(run_experiment, {'empowerment':[100.0]}, run_mode=mode_docker,
                             mounts=sweeper.mounts)
    else:
        run_experiment(empowerment=args.empowerment, exp_title=args.exp_title + '_' + str(args.seed))