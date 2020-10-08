from __future__ import division
import os
from envs import LunarLanderEmpowerment, LunarLander
from utils import logger
from policies import FullPilotPolicy
from datetime import datetime
import argparse
import multiprocessing

import doodad as dd
import doodad.mount as mount
import doodad.easy_sweep.launcher as launcher
import multiprocessing
from doodad.easy_sweep.hyper_sweep import run_sweep_doodad

from experiment_utils import config
from experiment_utils.utils import query_yes_no

EXP_NAME = "FullPilotTraining"



def run_experiment(**extra):

    base_dir = os.path.join(config.DOCKER_MOUNT_DIR, EXP_NAME)
    logger.configure(dir=base_dir, format_strs=['stdout', 'log', 'csv', 'tensorboard'])

    env = LunarLanderEmpowerment(empowerment=0.0, ac_continuous=False)

    max_ep_len = 1000
    n_training_episodes = 500
    max_timesteps = max_ep_len * n_training_episodes
    full_pilot_policy = FullPilotPolicy(base_dir)
    full_pilot_policy.learn(env, max_timesteps)
    print("Done")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Empowerment Lander Testbed')

    parser.add_argument('--exp_title', type=str, default='', help='Title for experiment')
    parser.add_argument('--mode', type=str, default='local',
                        help='Mode for running the experiments - local: runs on local machine, '
                             'ec2: runs on AWS ec2 cluster (requires a proper configuration file)')
    parser.add_argument('--num_cpu', '-c', type=int, default=multiprocessing.cpu_count(),
                        help='Number of threads to use for running experiments')
    args = parser.parse_args()

    print(config.BASE_DIR)
    local_mount = mount.MountLocal(local_dir=config.BASE_DIR, pythonpath=True)
    docker_mount_point = os.path.join(config.DOCKER_MOUNT_DIR, EXP_NAME)

    sweeper = launcher.DoodadSweeper([local_mount], docker_img=config.DOCKER_IMAGE,
                                     docker_output_dir=docker_mount_point,
                                     local_output_dir=os.path.join(config.DATA_DIR, 'local', EXP_NAME))
    sweeper.mount_out_s3 = mount.MountS3(s3_path='', mount_point=docker_mount_point, output=True)

    if args.mode == 'ec2':
        if query_yes_no("Continue?"):
            sweeper.run_sweep_ec2(run_experiment, {'alg':[0]}, bucket_name=config.S3_BUCKET_NAME,
                                  instance_type='c4.xlarge',
                                  region='us-west-1', s3_log_name=EXP_NAME, add_date_to_logname=True)
    elif args.mode == 'local_docker':
            mode_docker = dd.mode.LocalDocker(
                image=sweeper.image,
            )
            run_sweep_doodad(run_experiment, {'alg':[0]}, run_mode=mode_docker,
                             mounts=sweeper.mounts)

    else:
        run_experiment()




