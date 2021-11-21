import torch
import numpy as np
import time
import random
import string
import os

import gym
from gym.spaces import Discrete
from gym.envs.registration import register

from stable_baselines3_copy import DQN
from stable_baselines3_copy.common import set_random_seed
from stable_baselines3_copy.dqn.policies import CnnPolicy
from gym_turf import TurfEnv

def buff_to_action_dataset(buffer, batch_size):
    replay_data = buffer.sample(batch_size)
    return replay_data


def buff_to_dataset(buffer, d_min, d_max, n_couples):
    buffer.traj = buffer.dones.cumsum()[:-1]
    buffer.traj = np.insert(buffer.traj, 0, buffer.traj[0], axis=0)
    # buffer pos create discontinuity in the set of trajectories
    buffer.traj[buffer.pos:] += 1

    def get_couples(buff, d_min, d_max, n_couples):
        couples = []
        labels = []
        while len(couples) < n_couples:
            idx = np.random.randint(buff.buffer_size)
            possible_idxs = np.where(buff.traj == buff.traj[idx])[0]
            beg = possible_idxs[0]
            possible_idxs = np.where((np.abs(possible_idxs - idx) <= d_max) * (np.abs(possible_idxs - idx) >= d_min))[0]
            if len(possible_idxs) > 0:
                possible_idxs += beg
                other_idx = np.random.choice(possible_idxs)
                couples.append(np.concatenate([buff.observations[idx], buff.observations[other_idx]]))
                labels.append(float(other_idx > idx))
        return couples, labels

    couples, labels = get_couples(buffer, d_min, d_max, n_couples)
    couples, labels = np.array(couples), np.array(labels)
    return list(zip(couples, labels))


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc


def generate_buffer(size, env_name='cartpole', seed=42):
    np.random.seed(seed)

    if env_name == "turf":
        env = TurfEnv()
        env.seed(seed)
        set_random_seed(seed)
        model = DQN('CnnPolicy', env, verbose=1,
                    buffer_size=size,
                    learning_starts=size,
                    learning_rate=0.0001,
                    target_update_interval=50,
                    exploration_fraction=0.1,
                    exploration_initial_eps=1,
                    exploration_final_eps=1,
                    batch_size=32,
                    seed=seed,
                    )
        model.learn(total_timesteps=size)


    elif env_name == 'cartpole':
        env = gym.make('CartPole-v0')
        env.seed(seed)
        set_random_seed(seed)
        model = DQN('MlpPolicy', env, verbose=1,
                    buffer_size=size,
                    learning_starts=size,
                    learning_rate=0.0001,
                    target_update_interval=50,
                    exploration_fraction=0.1,
                    exploration_initial_eps=1,
                    exploration_final_eps=1,
                    batch_size=32,
                    seed=seed,
                    )
        model.learn(total_timesteps=size)

    else:
        raise NotImplementedError

    return model.replay_buffer
