import pickle
import pathlib
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import io
import numpy as np
import torch as th
from torch.optim import Adam
import os

from stable_baselines3_copy.common import logger
from stable_baselines3_copy.common.running_mean_std import RunningMeanStd
from stable_baselines3_copy.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper
from stable_baselines3_copy.common.buffers import ReplayBuffer

class VecSafe(VecEnvWrapper):
    """
    An environment to compute and use intrisinc reward

    :param venv: (VecEnv) the vectorized environment to wrap
    :param model_rev: (torch.nn.Module) the neural net corresponding to phi
    :param threshold: (float) threshold to reject action (the lower the safer)
    """

    def __init__(self, venv, model_rev, threshold=0.9):

        VecEnvWrapper.__init__(self, venv)
        self.model = model_rev
        self.threshold = threshold

        self.current_obs = np.array([])
        self.old_obs = np.array([])
        self.rewards = np.array([])
        self.len_episode = np.array([])
        self.is_saved = False
        self.real_actions = []
        self.counter = 0

    def step_wait(self):
        observations, rewards, dones, infos = self.venv.step_wait()
        self.current_obs = observations
        self.counter += 1

        return observations, rewards, dones, infos

    def reset(self):
        obs = self.venv.reset()
        self.current_obs = obs
        self.counter = 0
        return obs

    def step_async(self, actions: np.ndarray):
        actions = np.array(actions)
        with th.no_grad():
            rev_score = self.model(th.from_numpy(self.current_obs).to(self.model.device))
        irrev_idx = rev_score[:, actions].squeeze(1) > self.threshold
        if irrev_idx.sum() > 0:
            actions[irrev_idx.cpu().numpy()] = th.argmin(rev_score[irrev_idx], axis=1).cpu().numpy()
        self.real_actions = actions
        self.venv.step_async(actions)

if __name__ == "__main__":
    import gym
    import torch
    from torch import nn as nn
    from torch.nn.functional import relu

    from model import GrasslandARev

    use_gpu = torch.cuda.is_available()
    model_act = GrasslandARev()
    if use_gpu:
        model_act.cuda()
        model_act.device = "cuda"

    from stable_baselines3_copy.common.vec_env.vec_safe import VecSafe
    from stable_baselines3_copy.common.vec_env import DummyVecEnv, VecTransposeImage
    from gym_turf import TurfEnv

    # env.unwrapped.state
    env = TurfEnv()
    env = DummyVecEnv([lambda: env])
    env = VecTransposeImage(env)
    env = VecSafe(env, model_act)
    obs = env.reset()
    done = False
    observations = []
    imgs = []
    observations.append(obs)
    while not done:
        a = env.action_space.sample()
        print(a)
        obs, r, done, info = env.step([a])
        observations.append(obs)
        s = torch.FloatTensor(obs).cuda()
