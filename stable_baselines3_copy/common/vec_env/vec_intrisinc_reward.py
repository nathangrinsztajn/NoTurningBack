import pickle
import pathlib
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from collections import deque
import io
import numpy as np
import torch
from torch.optim import Adam
import os

from stable_baselines3_copy.common import logger
from stable_baselines3_copy.common.running_mean_std import RunningMeanStd
from stable_baselines3_copy.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper
from stable_baselines3_copy.common.buffers import ReplayBuffer

class VecIntrinsic(VecEnvWrapper):
    """
    An environment to compute and use intrisinc reward

    :param venv: (VecEnv) the vectorized environment to wrap
    :param feature_extractor: (torch.nn.Module) the feature extractor neural network
    :param head: (torch.nn.Module) the prediction head
    :param weight_intrinsic_reward: (float) scale of the intrinsic reward compared to the extrinsic one
    :param buffer_size: (int) how many transitions to store
    :param train_freq: (int) train the network every n steps
    :param gradient_steps: (int) how many gradient steps
    :param batch_size: (int) minibatch size
    :param learning_starts: (int) start computing the intrinsic reward only after n steps
    :param save/load: (str Union Path) save and load weights of the network used for computing intrinsic reward
    """

    def __init__(
        self, venv, feature_extractor, head, model_rev=None, weight_intrinsic_reward=1, func=lambda x: x-0.5, buffer_size=2000, train_freq=100, gradient_step=10,
    batch_size=256, lr=0.01, learning_start=500, d_min=0, d_max=50, reward_free=False, save_path=None, load=None):
        VecEnvWrapper.__init__(self, venv)
        self.feature_extractor = feature_extractor
        self.head = head
        self.model_rev = model_rev

        self.wir = weight_intrinsic_reward
        self.buffer_size = buffer_size
        self.train_freq = train_freq
        self.gradient_step = gradient_step
        self.batch_size = batch_size
        self.d_min = d_min
        self.d_max = d_max
        self.learning_start = learning_start
        self.save_path = save_path
        self.load = load
        self.func = func
        self.reward_free = reward_free

        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = Adam(list(self.feature_extractor.parameters()) + list(head.parameters()), lr=lr)
        self.counter = 0

        # replay buffer
        self.buffer = ReplayBuffer(
            self.buffer_size,
            self.venv.observation_space,
            self.venv.action_space,
        )

        self.old_obs = np.array([])
        self.rewards = np.array([])
        self.len_episode = np.zeros(self.num_envs)
        self.queu_episode = deque(maxlen=15)
        self.is_saved = False

    def save_replay_buffer(self, path: Union[str, pathlib.Path, io.BufferedIOBase]) -> None:
        """
        Save the replay buffer as a pickle file.

        :param path: (Union[str,pathlib.Path, io.BufferedIOBase]) Path to the file where the replay buffer should be saved.
            if path is a str or pathlib.Path, the path is automatically created if necessary.
        """
        assert self.buffer is not None, "The replay buffer is not defined"
        save_to_pkl(path, self.buffer, self.verbose)

    def load_replay_buffer(self, path: Union[str, pathlib.Path, io.BufferedIOBase]) -> None:
        """
        Load a replay buffer from a pickle file.

        :param path: (Union[str, pathlib.Path, io.BufferedIOBase]) Path to the pickled replay buffer.
        """
        self.buffer = load_from_pkl(path, self.verbose)
        assert isinstance(self.buffer, ReplayBuffer), "The replay buffer must inherit from ReplayBuffer class"


    def step_wait(self):
        observations, rewards, dones, infos = self.venv.step_wait()
        old_obs = self.old_obs
        self.buffer.add(old_obs, observations, self.actions, rewards, dones)

        if self.counter % self.train_freq == 0 and self.counter > self.learning_start:
            couples, labels = self._create_dataset()
            for e in range(self.gradient_step):
                loss = self.train_on_batch(couples, labels)
            logger.record('intrinsic/reversibility_loss', loss)
        else:
            loss = -1

        with torch.no_grad():
            old_obs = torch.tensor(old_obs).cuda()
            obs = torch.tensor(observations).cuda()

            if self.model_rev is None:
                old_obs = self.feature_extractor.extract_features(old_obs)
                obs = self.feature_extractor.extract_features(obs)
                p_rev = self.head(torch.cat((old_obs, obs), dim=1))
            else:
                p_rev = self.model_rev(old_obs, obs)

            p_rev = torch.sigmoid(p_rev).squeeze(1).detach().cpu().numpy()

        rewards_intrinsic = self.func(p_rev) * self.wir

        if self.reward_free:
            rewards = -rewards_intrinsic
        else:
            rewards -= rewards_intrinsic

        logger.record('intrinsic/reward_intrinsic', -np.mean(rewards_intrinsic))

        infos[0]['intrinsic/loss'] = loss
        infos[0]['intrinsic/reward'] = rewards_intrinsic
        self.counter += 1

        self.old_obs = observations
        self.rewards += rewards
        self.len_episode += 1

        if np.mean(self.queu_episode) >= 197 and not self.is_saved:
            self.save(self.save_path)
            self.is_saved = True

        for i, done in enumerate(dones):
            if done:
                logger.record("intrinsic/reward_episode", self.rewards[i])
                self.queu_episode.append(self.len_episode[i])
                self.len_episode[i] = 0
                self.rewards[i] = 0

        return observations, rewards, dones, infos

    def _create_dataset(self):
        self.buffer.traj = self.buffer.dones.cumsum()[:-1]
        self.buffer.traj = np.insert(self.buffer.traj, 0, self.buffer.traj[0], axis=0)

        # buffer pos create discontinuity in the set of trajectories
        self.buffer.traj[self.buffer.pos:] += 1

        def get_couples(buff, d_min, d_max, n_couples):
            couples = []
            labels = []
            while len(couples) < n_couples:
                idx = np.random.randint(buff.size())
                possible_idxs = np.where(buff.traj == buff.traj[idx])[0]
                beg = possible_idxs[0]
                possible_idxs = \
                np.where((np.abs(possible_idxs - idx) <= d_max) * (np.abs(possible_idxs - idx) >= d_min))[0]
                if len(possible_idxs) > 0:
                    possible_idxs += beg
                    other_idx = np.random.choice(possible_idxs)
                    couples.append(np.concatenate([buff.observations[idx], buff.observations[other_idx]]))
                    labels.append(float(other_idx > idx))
            return couples, labels

        couples, labels = get_couples(self.buffer, self.d_min, self.d_max, self.batch_size)
        couples, labels = np.array(couples), np.array(labels)
        couples, labels = torch.tensor(couples), torch.tensor(labels)
        return couples, labels

    def train_on_batch(self, couples, labels):
        obs1, obs2 = couples[:, 0].float(), couples[:, 1].float()

        if torch.cuda.is_available():
            obs1, obs2 = obs1.cuda(), obs2.cuda()
            labels = labels.cuda()

        self.optimizer.zero_grad()
        repr1, repr2 = self.feature_extractor.extract_features(obs1), self.feature_extractor.extract_features(obs2)
        repr = torch.cat([repr1, repr2], axis=1)
        pred = self.head(repr).squeeze(1)

        loss = self.criterion(pred, labels)
        loss.backward()
        self.optimizer.step()
        return loss.cpu().item()

    def reset(self) -> np.ndarray:
        """
        Reset all environments
        """
        obs = self.venv.reset()
        self.old_obs = obs
        self.rewards = np.zeros(self.num_envs)
        self.len_episode = np.zeros(self.num_envs)
        self.queu_episode = deque(maxlen=15)
        return obs

    def close(self) -> None:
        self.venv.close()

    def load(self, load_path: str) -> None:

        """
        Loads a saved nn object.
        :param load_path: (str) the path to load from.
        """

        self.head = torch.load(load_path + 'head.pt')
        self.feature_extractor = torch.load(load_path + 'feature_extractor.pt')

    def save(self, save_path: str) -> None:
        """
        Save current VecNormalize object with
        all running statistics and settings (e.g. clip_obs)

        :param save_path: (str) The path to save to
        """
        if self.model_rev is None:
            torch.save(self.head, os.path.join(save_path, 'head.pt'))
            torch.save(self.feature_extractor, os.path.join(save_path, 'feature_extractor.pt'))
        else:
            torch.save(self.model_rev, os.path.join(save_path, 'model_rev.pt'))

