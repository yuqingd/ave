from __future__ import division
import os
from collections import defaultdict, Counter

from envs import LunarLanderEmpowerment, LunarLander

from policies import FullPilotPolicy, LaggyPilotPolicy, NoopPilotPolicy, NoisyPilotPolicy, SensorPilotPolicy

from utils.env_utils import *

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

if __name__ == '__main__':
    data_dir = os.path.join('data', 'lunarlander-sim')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    max_ep_len = 1000
    env = LunarLanderEmpowerment(empowerment=0.0, ac_continuous=False)

    full_pilot_policy = FullPilotPolicy(data_dir, policy_path= os.path.join(data_dir, '01-13-2020 09-46-18/full_pilot_reward.pkl'))
    laggy_pilot_policy = LaggyPilotPolicy(data_dir, full_policy=full_pilot_policy.policy)
    noisy_pilot_policy = NoisyPilotPolicy(data_dir, full_policy=full_pilot_policy.policy)
    noop_pilot_policy = NoopPilotPolicy(data_dir, full_policy=full_pilot_policy.policy)
    sensor_pilot_policy = SensorPilotPolicy(data_dir, full_policy=full_pilot_policy.policy)

    pilot_names = ['full', 'laggy', 'noisy', 'noop', 'sensor']
    n_eval_eps = 100

    pilot_evals = [
        list(zip(*[run_ep(eval('%s_pilot_policy' % pilot_name), env, render=False, max_ep_len=max_ep_len) for _ in range(n_eval_eps)])) for
        pilot_name in pilot_names]

    mean_rewards = [np.mean(pilot_eval[0]) for pilot_eval in pilot_evals]
    outcome_distrns = [Counter(pilot_eval[1]) for pilot_eval in pilot_evals]

    print('\n'.join([str(x) for x in zip(pilot_names, mean_rewards, outcome_distrns)]))
