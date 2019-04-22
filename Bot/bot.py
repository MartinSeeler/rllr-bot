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
        # initialize Q(s,a)
        Q = {}
        states = grid.all_states()
        for s in states:
            Q[s] = {}
            for a in ALL_POSSIBLE_ACTIONS:
                Q[s][a] = 0

        # let's also keep track of how many times Q[s] has been updated
        update_counts = {}
        update_counts_sa = {}
        for s in states:
            update_counts_sa[s] = {}
            for a in ALL_POSSIBLE_ACTIONS:
                update_counts_sa[s][a] = 1.0
        # repeat until convergence
        ALPHA = 0.1
        t = 1.0
        deltas = []
        for it in range(np.min([game.round, 25])):
            my_grid = grid
            if it % 3 == 0:
                t += 1
            # if it % 2000 == 0:
            # print("it:", it)

            # instead of 'generating' an epsiode, we will PLAY
            # an episode within this loop
            s = my_grid.current_state()  # start state

            # the first (s, r) tuple is the state we start in and 0
            # (since we don't get a reward) for simply starting the game
            # the last (s, r) tuple is the terminal state and the final reward
            # the value for the terminal state is by definition 0, so we don't
            # care about updating it.
            a, _ = max_dict(Q[s])
            biggest_change = 0
            while not my_grid.game_over():
                a = random_action(a, eps=0.5 / t)  # epsilon-greedy
                # random action also works, but slower since you can bump into walls
                # a = np.random.choice(ALL_POSSIBLE_ACTIONS)
                r, my_grid = my_grid.move(a)
                if my_grid.i_lost():
                    r -= 100
                if my_grid.enemy_lost():
                    r += 100
                s2 = my_grid.current_state()

                # adaptive learning rate
                alpha = ALPHA / update_counts_sa[s][a]
                update_counts_sa[s][a] += 0.005

                # we will update Q(s,a) AS we experience the episode
                old_qsa = Q[s][a]
                # the difference between SARSA and Q-Learning is with Q-Learning
                # we will use this max[a']{ Q(s',a')} in our update
                # even if we do not end up taking this action in the next step
                a2, max_q_s2a2 = max_dict(Q[s2])
                Q[s][a] = Q[s][a] + alpha * (r + GAMMA * max_q_s2a2 - Q[s][a])
                biggest_change = max(biggest_change, np.abs(old_qsa - Q[s][a]))

                # we would like to know how often Q(s) has been updated too
                update_counts[s] = update_counts.get(s, 0) + 1

                # next state becomes current state
                s = s2
                a = a2

            deltas.append(biggest_change)

        # determine the policy from Q*
        # find V* from Q*
        policy = {}
        V = {}
        for s in grid.actions.keys():
            a, max_q = max_dict(Q[s])
            policy[s] = a
            V[s] = max_q

        self.next_move = policy[grid.current_state()]

    def do_turn(self):
        return self.next_move
