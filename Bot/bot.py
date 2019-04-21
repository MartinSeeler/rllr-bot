from Bot.grid import Grid
import numpy as np
import sys


def getValidActions(state, x, y, rows, cols):
    possible_actions = []
    if (x + 1) < rows and (state[x + 1, y] == '.'):
        possible_actions.append('down')
    if (x - 1) >= 0 and (state[x - 1, y] == '.'):
        possible_actions.append('up')
    if (y + 1) < cols and (state[x, y + 1] == '.'):
        possible_actions.append('right')
    if (y - 1) >= 0 and (state[x, y - 1] == '.'):
        possible_actions.append('left')
    return possible_actions


def isOutside(state, x, y, rows, cols):
    if x < 0 or x >= rows or y < 0 or y >= cols:
        return True
    return False


def isWall(state, x, y, rows, cols):
    if isOutside(state, x, y, rows, cols):
        return False
    if state[x, y] == 'x':
        return True
    return False


def getReward(state, x, y, rows, cols):
    reward = -0.1
    if isWall(state, x, y, rows, cols):
        # this is a wall
        reward -= 5
    if isOutside(state, x + 1, y, rows, cols):
        reward -= 0.1
    if isOutside(state, x - 1, y, rows, cols):
        reward -= 0.1
    if isOutside(state, x, y + 1, rows, cols):
        reward -= 0.1
    if isOutside(state, x, y - 1, rows, cols):
        reward -= 0.1
    if isWall(state, x + 1, y, rows, cols):
        reward -= 0.5
    if isWall(state, x - 1, y, rows, cols):
        reward -= 0.5
    if isWall(state, x, y + 1, rows, cols):
        reward -= 0.5
    if isWall(state, x, y - 1, rows, cols):
        reward -= 0.5
    return reward


def stateToGrid(stateDate, rows, cols, my_bot_id, enemy_bot_id):
    state_arr = stateDate.split(",")
    state = np.array(state_arr).reshape(rows, cols)
    my_bot_pos = np.argwhere(state == my_bot_id)[0]
    enemy_bot_pos = np.argwhere(state == enemy_bot_id)[0]
    g = Grid(rows, cols, my_bot_pos)
    actions = {}
    rewards = {}
    for x in range(rows):
        for y in range(cols):
            rewards.update({(x, y): getReward(state, x, y, rows, cols)})
            actions_valid = getValidActions(state, x, y, rows, cols)
            # print(x,y,actions_valid)
            if (len(actions_valid) > 0) and (state[x, y] in ['.', str(my_bot_id)]):
                actions.update({(x, y): actions_valid})
            elif len(actions_valid) == 0 and (state[x, y] in ['.']):
                rewards.update({(x, y): -10})

    g.set(rewards, actions)
    return g


def print_values(V, g):
    sys.stderr.write("\nVALUES\n")
    for i in range(g.rows):
        sys.stderr.write("------------------------------\n")
        for j in range(g.cols):
            v = V.get((i, j), 0)
            if v >= 0:
                sys.stderr.write(" %.2f|" % v)
            else:
                sys.stderr.write("%.2f|" % v)  # -ve sign takes up an extra space
        sys.stderr.write("\n")
        sys.stderr.flush()


def print_policy(P, g):
    sys.stderr.write("\nPOLICY\n")
    for i in range(g.rows):
        sys.stderr.write("------------------------------\n")
        for j in range(g.cols):
            a = P.get((i, j), ' ')
            sys.stderr.write(a.center(7) + " |")
        sys.stderr.write("\n")
        sys.stderr.flush()


def gameToGrid(game):
    return stateToGrid(game.field_data, game.field_height, game.field_width, str(game.my_botid),
                       str(game.other_botid))


def gridToPolicy(grid):
    SMALL_ENOUGH = 1e-2
    GAMMA = 0.9
    # state -> action
    # we'll randomly choose an action and update as we learn
    policy = {}
    for s in grid.actions.keys():
        policy[s] = np.random.choice(grid.actions[s])

    # initial policy
    # print("initial policy:")
    print_policy(policy, grid)

    # initialize V(s)
    states = grid.all_states()

    ### uniformly random actions ###
    # initialize V(s) = 0
    SMALL_ENOUGH = 5e-1  # threshold for convergence
    V = {}
    for s in states:
        V[s] = 0
        gamma = 1.0  # discount factor
    # repeat until convergence
    runs = 0
    while runs < 5:
        biggest_change = 0
        for s in states:
            old_v = V[s]
            # V(s) only has value if it's not a terminal state
            if s in grid.actions:

                new_v = 0  # we will accumulate the answer
                p_a = 1.0 / len(grid.actions[s])  # each action has equal probability
                for a in grid.actions[s]:
                    grid.set_state(s)
                    r = grid.move(a)
                    new_v += p_a * (r + gamma * V[grid.current_state()])
                V[s] = new_v
                biggest_change = max(biggest_change, np.abs(old_v - V[s]))
        # print(biggest_change)
        if biggest_change < SMALL_ENOUGH:
            break
        runs += 1
    # print("values for uniformly random actions:")
    print_values(V, grid)
    # find a policy that leads to optimal value function
    for s in policy.keys():
        best_a = None
        best_value = float('-inf')
        # loop through all possible actions to find the best current action
        for a in grid.actions[s]:
            grid.set_state(s)
            r = grid.move(a)
            v = r + GAMMA * V[grid.current_state()]
            if v > best_value:
                best_value = v
                best_a = a
        policy[s] = best_a

    # our goal here is to verify that we get the same answer as with policy iteration
    # print("values:")
    # print_values(V, grid)
    # print("policy:")
    print_policy(policy, grid)
    return policy


class Bot:

    def __init__(self, game):
        # self.game = game
        grid = gameToGrid(game)
        current_state = grid.current_state();
        policy = gridToPolicy(grid)
        self.next_move = policy[current_state]

    def do_turn(self):
        return self.next_move
