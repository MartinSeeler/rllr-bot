import numpy as np


def get_valid_actions(field, x, y, rows, cols):
    possible_actions = []
    if (x + 1) < rows and (field[x + 1, y] != 'x'):
        possible_actions.append('down')
    if (x - 1) >= 0 and (field[x - 1, y] != 'x'):
        possible_actions.append('up')
    if (y + 1) < cols and (field[x, y + 1] != 'x'):
        possible_actions.append('right')
    if (y - 1) >= 0 and (field[x, y - 1] != 'x'):
        possible_actions.append('left')
    return possible_actions


def get_reward(field, x, y):
    return 0.1


class LRGrid:  # Environment
    def __init__(self, field_data, rows, cols, my_id, enemy_id):
        self.rows = rows
        self.cols = cols
        state_arr = field_data.split(",")
        self.field = np.array(state_arr).reshape(rows, cols)
        self.my_id = my_id
        self.enemy_id = enemy_id
        self.my_position = np.argwhere(self.field == self.my_id)[0]
        self.enemy_position = np.argwhere(self.field == self.enemy_id)[0]
        # calculate possible actions and rewards per field
        self.actions = {}
        self.rewards = {}
        for x in range(self.rows):
            for y in range(self.cols):
                self.rewards.update({(x, y): get_reward(self.field, x, y)})
                actions_valid = get_valid_actions(self.field, x, y, self.rows, self.cols)
                if (len(actions_valid) > 0) and (self.field[x, y] in ['.', str(self.my_id), str(self.enemy_id)]):
                    self.actions.update({(x, y): actions_valid})

    def current_state(self):
        return (self.my_position[0], self.my_position[1])

    def enemy_state(self):
        return (self.enemy_position[0], self.enemy_position[1])

    def is_terminal(self, s):
        return s not in self.actions

    def enemy_lost(self):
        return self.enemy_state() not in self.actions

    def i_lost(self):
        return self.current_state() not in self.actions

    def game_over(self):
        return self.enemy_lost() or self.i_lost()

    def move(self, action, enemy_action=None):
        # check if legal move first
        if self.i_lost():
            return -100, self
        if self.enemy_lost():
            return 100, self
        my_pos_next = np.copy(self.my_position)
        enemy_pos_next = np.copy(self.enemy_position)
        #        if action in self.actions[self.current_state()]:
        if action == 'up':
            my_pos_next[0] -= 1
        elif action == 'down':
            my_pos_next[0] += 1
        elif action == 'right':
            my_pos_next[1] += 1
        elif action == 'left':
            my_pos_next[1] -= 1

        if (
                my_pos_next[0] < 0 or my_pos_next[0] >= self.rows) or (
                my_pos_next[1] < 0 or my_pos_next[1] >= self.cols) or self.field[my_pos_next[0], my_pos_next[1]] in [
            'x', self.enemy_id]:
            # we ran into wall, that's bad
            return -100, LRGrid(self.my_id + ",x," + self.enemy_id, 1, 3, self.my_id, self.enemy_id)
        if self.enemy_state() not in self.actions:
            # enemy can't move
            return 100, LRGrid(self.my_id + ",x," + self.enemy_id, 1, 3, self.my_id, self.enemy_id)
        if enemy_action == None:
            enemy_action = np.random.choice(self.actions[self.enemy_state()])
        if enemy_action == 'up':
            enemy_pos_next[0] -= 1
        elif enemy_action == 'down':
            enemy_pos_next[0] += 1
        elif enemy_action == 'right':
            enemy_pos_next[1] += 1
        elif enemy_action == 'left':
            enemy_pos_next[1] -= 1
        next_field_data = np.copy(self.field)
        next_field_data[self.current_state()[0], self.current_state()[1]] = 'x'
        next_field_data[my_pos_next[0], my_pos_next[1]] = self.my_id
        next_field_data[self.enemy_state()[0], self.enemy_state()[1]] = 'x'
        next_field_data[enemy_pos_next[0], enemy_pos_next[1]] = self.enemy_id
        # return a reward (if any)
        next_grid = LRGrid(','.join(next_field_data.reshape(-1)), self.rows, self.cols, self.my_id, self.enemy_id)
        return self.rewards.get(self.current_state(), -100), next_grid

    def all_states(self):
        # possibly buggy but simple way to get all states
        # either a position that has possible next actions
        # or a position that yields a reward
        return set(self.actions.keys()) | set(self.rewards.keys())
