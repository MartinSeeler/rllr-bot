from __future__ import print_function, division
# from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


# import numpy as np


class Grid:  # Environment
    def __init__(self, rows, cols, start):
        self.rows = rows
        self.cols = cols
        self.i = start[0]
        self.j = start[1]

    def set(self, rewards, actions):
        # rewards should be a dict of: (i, j): r (row, col): reward
        # actions should be a dict of: (i, j): A (row, col): list of possible actions
        self.rewards = rewards
        self.actions = actions

    def set_state(self, s):
        self.i = s[0]
        self.j = s[1]

    def current_state(self):
        return (self.i, self.j)

    def is_terminal(self, s):
        return s not in self.actions

    def move(self, action):
        # check if legal move first
        if action in self.actions[(self.i, self.j)]:
            if action == 'up':
                self.i -= 1
            elif action == 'down':
                self.i += 1
            elif action == 'right':
                self.j += 1
            elif action == 'left':
                self.j -= 1
        # return a reward (if any)
        return self.rewards.get((self.i, self.j), 0)

    def undo_move(self, action):
        # these are the opposite of what U/D/L/R should normally do
        if action == 'up':
            self.i += 1
        elif action == 'down':
            self.i -= 1
        elif action == 'right':
            self.j -= 1
        elif action == 'left':
            self.j += 1
        # raise an exception if we arrive somewhere we shouldn't be
        # should never happen
        assert (self.current_state() in self.all_states())

    def game_over(self):
        # returns true if game is over, else false
        # true if we are in a state where no actions are possible
        return (self.i, self.j) not in self.actions

    def all_states(self):
        # possibly buggy but simple way to get all states
        # either a position that has possible next actions
        # or a position that yields a reward
        return set(self.actions.keys()) | set(self.rewards.keys())
