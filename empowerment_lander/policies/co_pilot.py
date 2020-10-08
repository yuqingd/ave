from policies.co_build_graph import *
import tensorflow as tf
import baselines.common.tf_util as U
import tempfile

import os
import baselines.common.tf_util as U
from baselines.common.tf_util import load_variables, save_variables
from baselines import logger
from baselines.common.schedules import LinearSchedule
from baselines.common import set_global_seeds

import time
import csv

from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from baselines.deepq.utils import ObservationInput
from baselines.deepq.deepq import ActWrapper

from baselines.common.tf_util import get_session
from baselines.deepq.models import build_q_func

from policies.co_build_graph import *
from utils.env_utils import *
import utils.env_utils as utils

import uuid
import cloudpickle
import zipfile

class CoActWrapper(object):
    def __init__(self, act, act_params):
        self._act = act
        self._act_params = act_params
        self.initial_state = None

    @staticmethod
    def load_act(path, scope):
        with open(path, "rb") as f:
            model_data, act_params = cloudpickle.load(f)
        act = co_build_act(**act_params, scope=scope)
        sess = tf.Session()
        sess.__enter__()
        with tempfile.TemporaryDirectory() as td:
            arc_path = os.path.join(td, "packed.zip")
            with open(arc_path, "wb") as f:
                f.write(model_data)

            zipfile.ZipFile(arc_path, 'r', zipfile.ZIP_DEFLATED).extractall(td)
            load_variables(os.path.join(td, "model"))

        return ActWrapper(act, act_params)

    def __call__(self, *args, **kwargs):
        return self._act(*args, **kwargs)

    def step(self, observation, **kwargs):
        # DQN doesn't use RNNs so we ignore states and masks
        kwargs.pop('S', None)
        kwargs.pop('M', None)
        return self._act([observation], **kwargs), None, None, None

    def save_act(self, path=None):
        """Save model to a pickle located at `path`"""
        if path is None:
            path = os.path.join(logger.get_dir(), "model.pkl")

        with tempfile.TemporaryDirectory() as td:
            save_variables(os.path.join(td, "model"))
            arc_name = os.path.join(td, "packed.zip")
            with zipfile.ZipFile(arc_name, 'w') as zipf:
                for root, dirs, files in os.walk(td):
                    for fname in files:
                        file_path = os.path.join(root, fname)
                        if file_path != arc_name:
                            zipf.write(file_path, os.path.relpath(file_path, td))
            with open(arc_name, "rb") as f:
                model_data = f.read()
        with open(path, "wb") as f:
            cloudpickle.dump((model_data, self._act_params), f)

    def save(self, path):
        save_variables(path)


def load_act(path, scope):
    """Load act function that was returned by learn function.

    Parameters
    ----------
    path: str
        path to the act function pickle

    Returns
    -------
    act: ActWrapper
        function that takes a batch of observations
        and returns actions.
    """
    return CoActWrapper.load_act(path,scope)

def learn(
        env,
        network,
        seed=None,
        lr=1e-3,
        total_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        train_freq=1,
        batch_size=32,
        print_freq=100,
        checkpoint_freq=10000,
        checkpoint_path=None,
        learning_starts=1000,
        gamma=1.0,
        target_network_update_freq=500,
        prioritized_replay=False,
        prioritized_replay_alpha=0.6,
        prioritized_replay_beta0=0.4,
        prioritized_replay_beta_iters=None,
        prioritized_replay_eps=1e-6,
        param_noise=False,
        num_cpu=5,
        callback=None,
        scope='co_deepq',
        pilot_tol=0,
        pilot_is_human=False,
        reuse=False,
        load_path=None,
        **network_kwargs):
    # Create all the functions necessary to train the model

    sess = get_session() #tf.Session(graph=tf.Graph())
    set_global_seeds(seed)

    q_func = build_q_func(network, **network_kwargs)

    observation_space = env.observation_space
    def make_obs_ph(name):
        return ObservationInput(observation_space, name=name)

    using_control_sharing = True #pilot_tol > 0

    if pilot_is_human:
        utils.human_agent_action = init_human_action()
        utils.human_agent_active = False

    act, train, update_target, debug = co_build_train(
        scope=scope,
        make_obs_ph=make_obs_ph,
        q_func=q_func,
        num_actions=env.action_space.n,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
        gamma=gamma,
        grad_norm_clipping=10,
        reuse=tf.AUTO_REUSE if reuse else False,
        using_control_sharing=using_control_sharing
    )

    act_params = {
        'make_obs_ph': make_obs_ph,
        'q_func': q_func,
        'num_actions': env.action_space.n,
    }

    act = ActWrapper(act, act_params)

    # Create the replay buffer
    if prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)
        if prioritized_replay_beta_iters is None:
            prioritized_replay_beta_iters = total_timesteps
        beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                       initial_p=prioritized_replay_beta0,
                                       final_p=1.0)
    else:
        replay_buffer = ReplayBuffer(buffer_size)
        beta_schedule = None

    # Initialize the parameters and copy them to the target network.
    U.initialize()
    update_target()

    episode_rewards = [0.0]
    episode_outcomes = []
    saved_mean_reward = None
    obs = env.reset()
    reset = True
    prev_t = 0
    rollouts = []

    if not using_control_sharing:
        exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * total_timesteps),
                                     initial_p=1.0,
                                     final_p=exploration_final_eps)

    with tempfile.TemporaryDirectory() as td:
        td = checkpoint_path or td

        model_file = os.path.join(td, "model")
        model_saved = False

        if tf.train.latest_checkpoint(td) is not None:
            load_variables(model_file)
            logger.log('Loaded model from {}'.format(model_file))
            model_saved = True
        elif load_path is not None:
            load_variables(load_path)
            logger.log('Loaded model from {}'.format(load_path))

        for t in range(total_timesteps):
            masked_obs = mask_helipad(obs)

            act_kwargs = {}
            if using_control_sharing:
                if pilot_is_human:
                    act_kwargs['pilot_action'] =  env.unwrapped.pilot_policy(obs[None, :9])
                else:
                    act_kwargs['pilot_action'] = env.unwrapped.pilot_policy.step(obs[None, :9])
                act_kwargs['pilot_tol'] = pilot_tol if not pilot_is_human or (pilot_is_human and utils.human_agent_active) else 0
            else:
                act_kwargs['update_eps'] = exploration.value(t)

            #action = act(masked_obs[None, :], **act_kwargs)[0][0]
            action = act(np.array(masked_obs)[None], **act_kwargs)[0][0]
            env_action = action
            reset = False
            new_obs, rew, done, info = env.step(env_action)

            if pilot_is_human:
                env.render()

            # Store transition in the replay buffer.
            masked_new_obs = mask_helipad(new_obs)
            replay_buffer.add(masked_obs, action, rew, masked_new_obs, float(done))
            obs = new_obs

            episode_rewards[-1] += rew
            if done:
                obs = env.reset()
                episode_rewards.append(0.0)
                reset = True

                if pilot_is_human:
                    utils.human_agent_action = init_human_action()
                    utils.human_agent_active = False
                    time.sleep(2)


            if t > learning_starts and t % train_freq == 0:
                if prioritized_replay:
                    experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(t))
                    (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                else:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
                    weights, batch_idxes = np.ones_like(rewards), None
                td_errors = train(obses_t, actions, rewards, obses_tp1, dones, weights)

                if prioritized_replay:
                    new_priorities = np.abs(td_errors) + prioritized_replay_eps
                    replay_buffer.update_priorities(batch_idxes, new_priorities)

            if t > learning_starts and t % target_network_update_freq == 0:
                # Update target network periodically.
                update_target()

            episode_outcomes.append(rew)
            episode_rewards.append(0.0)

            if t > learning_starts and t % target_network_update_freq == 0:
                # Update target network periodically.
                update_target()

            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            mean_100ep_succ = round(np.mean([1 if x == 100 else 0 for x in episode_outcomes[-101:-1]]), 2)
            mean_100ep_crash = round(np.mean([1 if x == -100 else 0 for x in episode_outcomes[-101:-1]]), 2)
            num_episodes = len(episode_rewards)
            if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", num_episodes)
                logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                logger.record_tabular("mean 100 episode succ", mean_100ep_succ)
                logger.record_tabular("mean 100 episode crash", mean_100ep_crash)
                logger.dump_tabular()

            if checkpoint_freq is not None and t > learning_starts and num_episodes > 100 and t % checkpoint_freq == 0 and (
                    saved_mean_reward is None or mean_100ep_reward > saved_mean_reward):
                if print_freq is not None:
                    logger.log("Saving model due to mean reward increase: {} -> {}".format(
                        saved_mean_reward, mean_100ep_reward))
                save_variables(model_file)
                model_saved = True
                saved_mean_reward = mean_100ep_reward

        if model_saved:
            if print_freq is not None:
                logger.log("Restored model with mean reward: {}".format(saved_mean_reward))
            load_variables(model_file)

    reward_data = {
        'rewards': episode_rewards,
        'outcomes': episode_outcomes
    }

    return act, reward_data
ACT_DIM = 6
class CoPilotPolicy(object):
    def __init__(self, data_dir, policy_path=None):
        self.policy = None
        self.policy_path = policy_path
        self.data_dir = data_dir
        self.scope=None
        if policy_path is not None:
            scope = os.path.basename(policy_path)
            scope = scope.split("_policy")
            self.scope = scope[0]
            self.policy = load_act(policy_path, self.scope)

    def learn(self, env, max_timesteps, copilot_scope='co_deepq', pilot_tol=0, pilot_is_human=False, **extras):

        if copilot_scope is not None:
            scope = copilot_scope
        elif copilot_scope is None:
            scope = str(uuid.uuid4())
        self.scope = scope
        self.policy, self.reward_data = learn(
            env,
            scope=scope,
            network='mlp',
            total_timesteps=max_timesteps,
            pilot_tol=pilot_tol,
            reuse=pilot_is_human,
            lr=1e-3,
            target_network_update_freq=500,
            pilot_is_human=pilot_is_human,
            gamma=0.99
        )

        self.policy_path = self.data_dir + '/' + self.scope + '_policy.pkl'
        self.policy.save_act(path=self.policy_path)

    def step(self, observation, pilot_policy, pilot_tol, pilot_actions, pilot_is_human=False):
        with tf.variable_scope(self.scope, reuse=None):
            lander_obs = np.squeeze(observation)[:9]
            masked_obs = mask_helipad(lander_obs)
            if pilot_is_human:
                pilot_action = onehot_encode(pilot_policy(lander_obs[None, :]))  # pilot knows where goal is
            else:
                pilot_action = onehot_encode(pilot_policy.step(lander_obs[None, :])) #pilot knows where goal is

            pilot_actions[ACT_DIM * 1:] = pilot_actions[:-1 * ACT_DIM]
            pilot_actions[:ACT_DIM] = pilot_action

            feed_obs = np.concatenate((masked_obs, pilot_actions))

            return self.policy._act(
                feed_obs[None, :],
                pilot_tol=pilot_tol,
                pilot_action=onehot_decode(pilot_action)
            )[0][0], pilot_actions

