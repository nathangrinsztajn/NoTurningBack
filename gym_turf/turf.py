import gym
from gym.utils import seeding
from gym.spaces.discrete import Discrete
from gym.spaces import Box
from .utils import generate_fixed_room
from .render_utils import room_to_tiny_world_rgb
import numpy as np
import random


class TurfEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'tiny_human', 'tiny_rgb_array', 'raw']
    }

    def __init__(self,
                 dim_room=(10, 10),
                 max_steps=120,
                 grass_penalty=0,
                 goal_reward=1,
                 step_penalty=0.1,
                 tiny=True,
                 fixed=True
                 ):

        # Penalties and Rewards
        self.step_penalty = step_penalty
        self.goal_reward = goal_reward
        self.grass_penalty = grass_penalty

        # Other Settings
        self.tiny = tiny
        self.viewer = None
        self.max_steps = max_steps
        self.action_space = Discrete(len(ACTION_LOOKUP))
        self.fixed = fixed

        if self.fixed:
            self.grass, self.player_position, self.goal_position = generate_fixed_room()
            dim_room = self.grass.shape

            if not self.tiny:
                screen_height, screen_width = (dim_room[0] * 16, dim_room[1] * 16)
            else:
                screen_height, screen_width = (dim_room[0], dim_room[1])

            self.observation_space = Box(low=0, high=255, shape=(screen_height, screen_width, 3), dtype=np.uint8)
        else:
            raise NotImplementedError

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action, observation_mode='rgb_array'):

        if self.tiny:
            observation_mode = 'tiny_rgb_array'

        assert action in ACTION_LOOKUP
        assert observation_mode in ['rgb_array', 'tiny_rgb_array', 'raw']

        old_observation = self.render(mode=observation_mode)
        self.num_env_steps += 1

        moved_player, ruined_grass = self._move(action)

        reward = (self.player_position == self.goal_position).all() * self.goal_reward - self.step_penalty -\
                 self.grass_penalty * ruined_grass

        done = self._check_if_done()

        # Convert the observation to RGB frame
        observation = self.render(mode=observation_mode)

        info = {
            "action.name": ACTION_LOOKUP[action],
            "action.moved_player": moved_player,
            "action.ruined_grass": ruined_grass,
            "old_obs": old_observation,
            'ruined grasses': self.ruined_grasses,
        }
        if done:
            info["maxsteps_used"] = self._check_if_maxsteps()
            info["goal reached"] = self.player_position == self.goal_position

        return observation, reward, done, info

    def _move(self, action):
        """
        Moves the player to the next field, if it is not occupied.
        :param action:
        :return: Boolean, indicating a change of the room's state
        """
        change = CHANGE_COORDINATES[action]
        new_position = self.player_position + change
        current_position = self.player_position.copy()

        ruined_grass = False

        # Move player if the field in the moving direction is either
        # an empty field or an empty box target.
        if ((new_position < 0).sum() == 0) and ((new_position >= self.grass.shape[0]).sum() == 0):
            if self.grass[current_position[0], current_position[1]] == 0:
                self.grass[current_position[0], current_position[1]] = 1
                ruined_grass = True
                self.ruined_grasses += 1
            self.player_position = new_position
            return True, ruined_grass
        return False, False

    def _check_if_done(self):
        return (self.player_position == self.goal_position).all() or self._check_if_maxsteps()

    def _check_if_maxsteps(self):
        return self.max_steps == self.num_env_steps

    def reset(self, render_mode='rgb_array'):

        if self.tiny:
            render_mode = 'tiny_rgb_array'

        if self.fixed:
            self.grass, self.player_position, self.goal_position = generate_fixed_room()
        else:
            raise NotImplementedError

        self.num_env_steps = 0
        self.ruined_grasses = 0

        starting_observation = self.render(render_mode)
        return starting_observation

    def render(self, mode='human', close=None, scale=1):
        assert mode in RENDERING_MODES

        img = self.get_image(mode, scale)

        if 'rgb_array' in mode:
            return img

        elif 'human' in mode:
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

        elif 'raw' in mode:
            return self.grass, self.player_position, self.goal_position

        else:
            raise NotImplementedError

    def get_image(self, mode, scale=1):

        if mode.startswith('tiny_'):
            img = room_to_tiny_world_rgb(self.grass, self.player_position)
        else:
            raise NotImplementedError

        return img

    def close(self):
        if self.viewer is not None:
            self.viewer.close()

    def set_maxsteps(self, num_steps):
        self.max_steps = num_steps

    def get_action_lookup(self):
        return ACTION_LOOKUP

    def get_action_meanings(self):
        return ACTION_LOOKUP


ACTION_LOOKUP = {
    0: 'move up',
    1: 'move down',
    2: 'move left',
    3: 'move right',
}

CHANGE_COORDINATES = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1)
}

RENDERING_MODES = ['rgb_array', 'human', 'tiny_rgb_array', 'tiny_human', 'raw']
