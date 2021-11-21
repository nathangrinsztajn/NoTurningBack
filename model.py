import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
import numpy as np

import gym
from gym.spaces import Box
from stable_baselines3_copy.common.preprocessing import is_image_space

class GrasslandRev(nn.Module):

    def __init__(self, observation_space=Box(0, 255, (3, 10, 10), np.uint8), features_dim=64):
        super(GrasslandRev, self).__init__()
        assert is_image_space(observation_space), (
            "You should use NatureCNN "
            f"only with images not with {observation_space} "
            "(you are probably using `CnnPolicy` instead of `MlpPolicy`)"
        )
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
        self.head = Linear(2 * features_dim, 1)

    def encode(self, x):
        x = x.float() / 255.0
        return self.linear(self.cnn(x))

    def forward(self, x, y):
        x = self.encode(x)
        y = self.encode(y)
        out = torch.cat([x, y], axis=1)
        return self.head(out)


class GrasslandARev(nn.Module):

    def __init__(self, observation_space=Box(0, 255, (3, 10, 10), np.uint8), features_dim=64):
        super(GrasslandARev, self).__init__()
        self.encoder = GrasslandRev(observation_space=observation_space, features_dim=features_dim)
        self.head = nn.Linear(features_dim, 4)

    def encode(self, x):
        return self.encoder.encode(x)

    def forward(self, s):
        x = self.encode(s)
        out = self.head(x)
        return out


class ExtractorGrassland(nn.Module):

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super(ExtractorGrassland, self).__init__()
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        assert is_image_space(observation_space), (
            "You should use NatureCNN "
            f"only with images not with {observation_space} "
            "(you are probably using `CnnPolicy` instead of `MlpPolicy`)"
        )
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = observations.float() / 255.0
        return self.linear(self.cnn(x))

    def extract_features(self, x):
        return self.forward(x)


class CartpoleRev(nn.Module):

    def __init__(self):
        super(CartpoleRev, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(4, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(128, 1))

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x, y):
        x = self.encode(x)
        y = self.encode(y)
        out = torch.cat([x, y], axis=1)
        return self.head(out)


class CartpoleARev(nn.Module):

    def __init__(self):
        super(CartpoleARev, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(4, 64), nn.ReLU())
        self.head = nn.Linear(64, 2)

    def encode(self, x):
        return self.encoder(x)

    def forward(self, s):
        x = self.encode(s)
        out = self.head(x)
        return out


class ExtractorCartpole(nn.Module):
    def __init__(self):
        super(ExtractorCartpole, self).__init__()
        self.l1 = Linear(4, 64)
        self.l2 = Linear(64, 64)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x

    def extract_features(self, x):
        return self.forward(x)
