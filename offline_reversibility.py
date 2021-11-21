import gym
from gym.spaces import Box
import os
import numpy as np
import torch
from torch import nn as nn
from torch.nn.functional import relu
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import pickle as pkl
import torch as th
from model import CartpoleRev, CartpoleARev, GrasslandRev, GrasslandARev
from utils import buff_to_dataset, binary_acc
from utils import generate_buffer
from stable_baselines3_copy.common.monitor import Monitor
from stable_baselines3_copy.common.vec_env import DummyVecEnv
from stable_baselines3_copy import DQN, PPO
from gym_turf import TurfEnv

def learn_rev_classifier(n_traj, env_str='cartpole', w_max=200, dataset_size=10**4, epochs=100, lr=0.01, seed=42, no_cuda=False,
                         verbose=0):

    criterion = torch.nn.BCEWithLogitsLoss()

    if env_str == 'cartpole':
        model = CartpoleRev()

    if env_str == 'turf':
        obs_space = Box(0, 255, (3, 10, 10), np.uint8)
        model = GrasslandRev(observation_space=obs_space, features_dim=64)

    use_gpu = torch.cuda.is_available() and not no_cuda
    if use_gpu:
        model.cuda()

    buffer = generate_buffer(n_traj, seed=seed, env_name=env_str)

    optimizer = Adam(model.parameters(), lr=lr)

    for e in range(1, epochs + 1):
        epoch_loss = 0
        epoch_acc = 0

        dataset = buff_to_dataset(buffer, 1, w_max, dataset_size)
        train_loader = DataLoader(dataset, batch_size=256, shuffle=True)

        for batch in train_loader:
            obs, labels = batch[0], batch[1]
            obs1, obs2 = obs[:, 0].float(), obs[:, 1].float()

            if use_gpu:
                obs1, obs2 = obs1.cuda(), obs2.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            pred = model(obs1, obs2).squeeze(1)

            loss = criterion(pred, labels)
            acc = binary_acc(pred, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        if verbose == 1:
            print(
                f'Epoch {e + 0:03}: | Loss: {epoch_loss / len(train_loader):.5f} | Acc: {epoch_acc / len(train_loader):.3f}')

    return model, buffer


def learn_rev_action(model, buffer, env_str='cartpole', epochs=10**5, lr=0.01, no_cuda=False, verbose=0):
    criterion = torch.nn.BCEWithLogitsLoss()

    if env_str == 'cartpole':
        model_act = CartpoleARev()

    if env_str == 'turf':
        obs_space = Box(0, 255, (3, 10, 10), np.uint8)
        model_act = GrasslandARev(obs_space, features_dim=64)

    use_gpu = torch.cuda.is_available() and not no_cuda
    if use_gpu:
        model_act.cuda()
        model_act.device = "cuda"

    optimizer = Adam(model_act.parameters(), lr=lr)

    for e in range(1, epochs + 1):
        epoch_loss = 0
        epoch_acc = 0

        replay_data = buffer.sample(1024)
        data_observations = replay_data.observations[replay_data.dones.squeeze(1) == 0][:128]
        data_next_observations = replay_data.next_observations[replay_data.dones.squeeze(1) == 0][:128]
        data_actions = replay_data.actions[replay_data.dones.squeeze(1) == 0][:128]

        with th.no_grad():
            # Compute the target reversibility values
            target_rev = model(data_observations, data_next_observations)
        optimizer.zero_grad()

        # Get current Q estimates
        current_rev = model_act(data_observations)
        current_rev = th.gather(current_rev, dim=1, index=data_actions.long())

        #         if use_gpu:
        #             obs1, obs2 = obs1.cuda(), obs2.cuda()
        #             labels = labels.cuda()

        loss = criterion(current_rev, torch.sigmoid(target_rev))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        if verbose == 1:
            print(f'Epoch {e + 0:03}: | Loss: {epoch_loss / len(replay_data):.5f}')
    return model_act

