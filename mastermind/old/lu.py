'''Reinforcement Learning approach adapted from Lu, et. all
Paper: http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7926576'''
import random
import time
from collections import defaultdict
from copy import copy
from mastermind import *
from rl import *


class LuStateManager(object):

    def __init__(self, learning_rate=0.1, discount=0.9):
        self.q = defaultdict(int)
        self.learning_rate = learning_rate
        self.discount = discount

    def get_starting_state(self):
        return [1 for i in xrange(NUM_OPTIONS**NUM_PEGS)]

    def get_action(self, state, epsilon=0.1):
        best_actions = []
        best_q_val = -100000000000000000000
        for i, val in enumerate(state):
            if val == 1:
                action = i
                q_val = self.get_q(state, action)
                if q_val > best_q_val:
                    best_q_val = q_val
                    best_actions = [action]
                elif q_val == best_q_val:
                    best_actions.append(action)

        # randomly choose action at random to explore
        if random.random() < epsilon:
            return random.randint(0, NUM_OPTIONS**NUM_PEGS-1)
        else:
            return random.choice(best_actions)

    def get_q(self, state, action):
        state = map(str, state)
        s = ''.join(state) + ' ' + str(action)
        return self.q[s]

    def set_q(self, state, action, value):
        state = map(str, state)
        s = ''.join(state) + ' ' + str(action)
        self.q[s] = value

    def update_q(self, state, action, reward, next_q):
        this_q = self.get_q(state, action)
        self.set_q(state, action, this_q + self.learning_rate*(reward + self.discount*next_q - this_q))
        return self.get_q(state, action)

    def update_state(self, state, action, result):
        state = copy(state)
        guess = get_guess(action)
        for i, val in enumerate(state):
            if val == 1 and validate_attempt(guess, get_guess(i)) != result:
                state[i] = 0
        return state

    def learn(self, actions):
        next_q = 0
        for (state, action, result) in actions[::-1]:
            if result == (4, 4):
                next_q = self.update_q(state, action, 0, next_q)
            else:
                next_q = self.update_q(state, action, -1, next_q)

'''
Results (after 2000 epochs):
3 pegs, 6 options: 4.12962962963
4 pegs, 6 options: '''


if __name__ == '__main__':
    state_manager = LuStateManager()
    ml = MastermindRL(state_manager, num_epochs=2500, log_name='small/lu{}')
    ml.train()
    ml.test()
