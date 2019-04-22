from Bot.grid import LRGrid
import numpy as np
import sys

GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('up', 'down', 'left', 'right')


def max_dict(d):
    # returns the argmax (key) and max (value) from a dictionary
    # put this into a function since we are using it so often
    max_key = None
    max_val = float('-inf')
    for k, v in d.items():
        if v > max_val:
            max_val = v
            max_key = k
    return max_key, max_val


def random_action(a, eps=0.5):
    # choose given a with probability 1 - eps + eps/4
    # choose some other a' != a with probability eps/4
    p = np.random.random()
    # if p < (1 - eps + eps/len(ALL_POSSIBLE_ACTIONS)):
    #   return a
    # else:
    #   tmp = list(ALL_POSSIBLE_ACTIONS)
    #   tmp.remove(a)
    #   return np.random.choice(tmp)
    #
    # this is equivalent to the above
    if p < (1 - eps):
        return a
    else:
        return np.random.choice(ALL_POSSIBLE_ACTIONS)


def play_game(grid, policy):
    # returns a list of states and corresponding returns
    # in this version we will NOT use "exploring starts" method
    # instead we will explore using an epsilon-soft policy
    s = grid.current_state()
    a = random_action(policy[s])

    # be aware of the timing
    # each triple is s(t), a(t), r(t)
    # but r(t) results from taking action a(t-1) from s(t-1) and landing in s(t)
    states_actions_rewards = [(s, a, 0)]
    while True:
        r, grid = grid.move(a)
        s = grid.current_state()
        if grid.game_over():
            if grid.i_lost():
                r -= 100
            if grid.enemy_lost():
                r += 100
            states_actions_rewards.append((s, None, r))
            break
        else:
            a = random_action(policy[s])  # the next state is stochastic
            states_actions_rewards.append((s, a, r))

    # calculate the returns by working backwards from the terminal state
    G = 0
    states_actions_returns = []
    first = True
    for s, a, r in reversed(states_actions_rewards):
        # the value of the terminal state is 0 by definition
        # we should ignore the first state we encounter
        # and ignore the last G, which is meaningless since it doesn't correspond to any move
        if first:
            first = False
        else:
            states_actions_returns.append((s, a, G))
        G = r + GAMMA * G
    states_actions_returns.reverse()  # we want it to be in order of state visited
    return states_actions_returns


class Bot:

    def __init__(self, game):
        # self.game = game
        grid = LRGrid(game.field_data, game.field_height, game.field_width, str(game.my_botid),
                      str(game.other_botid))
        # state -> action
        # initialize a random policy
        policy = {}
        for s in grid.actions.keys():
            policy[s] = np.random.choice(grid.actions[s])
        # initialize Q(s,a) and returns
        Q = {}
        returns = {}  # dictionary of state -> list of returns we've received
        states = grid.all_states()
        for s in states:
            if s in grid.actions:  # not a terminal state
                Q[s] = {}
                for a in ALL_POSSIBLE_ACTIONS:
                    Q[s][a] = 0
                    returns[(s, a)] = []
            else:
                # terminal state or state we can't otherwise get to
                pass

        deltas = []
        for t in range(30):
            # generate an episode using pi
            biggest_change = 0
            states_actions_returns = play_game(grid, policy)

            # calculate Q(s,a)
            seen_state_action_pairs = set()
            for s, a, G in states_actions_returns:
                # check if we have already seen s
                # called "first-visit" MC policy evaluation
                sa = (s, a)
                if sa not in seen_state_action_pairs:
                    old_q = Q[s][a]
                    returns[sa].append(G)
                    Q[s][a] = np.mean(returns[sa])
                    biggest_change = max(biggest_change, np.abs(old_q - Q[s][a]))
                    seen_state_action_pairs.add(sa)
            deltas.append(biggest_change)

            # calculate new policy pi(s) = argmax[a]{ Q(s,a) }
            for s in policy.keys():
                a, _ = max_dict(Q[s])
                policy[s] = a

        self.next_move = policy[grid.current_state()]

    def do_turn(self):
        return self.next_move
