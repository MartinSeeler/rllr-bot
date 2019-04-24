# core modules
import pkg_resources
import random

# 3rd party modules
from gym import spaces
import gym
from gym.utils import seeding
import numpy as np


class LightRidersEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self._seed = -1
        self.seed()
        self.rows = 16
        self.cols = 16
        self.grid = np.zeros((self.rows, self.cols))
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(0, 4, (self.rows * self.cols + 1,))
        self.p0_position = [0, 0]
        self.p1_position = [0, 0]
        self.me_first = True

    def _is_outside(self, pos):
        return pos[0] < 0 or pos[0] >= self.rows or pos[1] < 0 or pos[1] >= self.cols

    def _is_wall(self, pos):
        return self.grid[pos[0], pos[1]] == 1

    def _is_loosing_position(self, pos):
        return self._is_outside(pos) or self._is_wall(pos)

    def _get_obs(self):
        """
        0 == Free Space
        1 == Wall
        2 == Me
        3 == Enemy
        """
        obs = np.copy(self.grid)
        if not self._is_loosing_position(self.p0_position):
            obs[self.p0_position[0], self.p0_position[1]] = 2 if self.me_first else 3
        if not self._is_loosing_position(self.p1_position):
            obs[self.p1_position[0], self.p1_position[1]] = 3 if self.me_first else 2
        return np.append(np.array([0 if self.me_first else 1]), obs.reshape(-1))

    def render(self, mode='human', close=False):
        g = self._get_obs()[1:].reshape(self.rows, self.cols)
        for i in range(self.rows):
            print("---" * self.cols)
            for j in range(self.cols):
                a = g[i, j]
                c = '.'
                if a == 1:
                    c = 'x'
                elif a == 2:
                    c = 'u'
                elif a == 3:
                    c = 'e'
                print(c.upper().center(3), end="")
            print("")

    def reset(self):
        self.grid = np.zeros((self.rows, self.cols))
        p0_start_x = np.random.randint(1, self.rows - 1)
        p0_start_y = np.random.randint(1, self.cols // 2 - 1)
        self.p0_position = [p0_start_x, p0_start_y]
        self.p1_position = [p0_start_x, self.cols - 1 - p0_start_y]
        self.me_first = np.random.rand() <= 0.5

        print(self.me_first)

    def step(self, action):
        """

        Parameters
        ----------
        action :

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        self.take_action(action)

        reward = self.get_reward()
        ob = self._get_obs()
        episode_over = self._is_loosing_position(self.p0_position) or self._is_loosing_position(self.p1_position)
        return ob, reward, episode_over, {}

    def _update_pos(self, pos, action):
        if action == 0:
            # UP
            return [pos[0] - 1, pos[1]]
        elif action == 1:
            # DOWN
            return [pos[0] + 1, pos[1]]
        elif action == 2:
            # LEFT
            return [pos[0], pos[1] - 1]
        elif action == 3:
            # RIGHT
            return [pos[0], pos[1] + 1]
        else:
            return pos

    def _is_action_possible(self, position, action):
        return not self._is_loosing_position(self._update_pos(position, action))

    def take_action(self, action):
        """
        0 == UP
        1 == DOWN
        2 == LEFT
        3 == RIGHT
        """
        all_actions = [0, 1, 2, 3]
        if self.me_first:
            # set current position to wall
            self.grid[self.p0_position[0], self.p0_position[1]] = 1
            # update my position based on action
            self.p0_position = self._update_pos(self.p0_position, action)

            # calculate enemy move
            valid_enemy_actions = list(filter(lambda x: self._is_action_possible(self.p1_position
                                                                                 , x), all_actions))
            enemy_action = np.random.choice(all_actions)
            if len(valid_enemy_actions) > 0:
                enemy_action = np.random.choice(valid_enemy_actions)

            # set current enemy position to wall
            self.grid[self.p1_position[0], self.p1_position[1]] = 1
            # update my position based on action
            self.p1_position = self._update_pos(self.p1_position, enemy_action)
        else:

            # calculate enemy move
            valid_enemy_actions = list(filter(lambda x: self._is_action_possible(self.p0_position
                                                                                 , x), all_actions))
            enemy_action = np.random.choice(all_actions)
            if len(valid_enemy_actions) > 0:
                enemy_action = np.random.choice(valid_enemy_actions)

            # set current enemy position to wall
            self.grid[self.p0_position[0], self.p0_position[1]] = 1
            # update my position based on action
            self.p0_position = self._update_pos(self.p0_position, enemy_action)

            # set current position to wall
            self.grid[self.p1_position[0], self.p1_position[1]] = 1
            # update my position based on action
            self.p1_position = self._update_pos(self.p1_position, action)

    def get_reward(self):
        """ Reward is given for XY. """
        if self._is_loosing_position(self.p0_position) and self._is_loosing_position(self.p1_position):
            # DRAW
            return -50
        if self._is_loosing_position(self.p0_position):
            # Player 0 lost
            if self.me_first:
                # we lost
                return -100
            else:
                return 100
        if self._is_loosing_position(self.p1_position):
            # Player 0 lost
            if self.me_first:
                # we lost
                return 100
            else:
                return -100
        return -0.1
