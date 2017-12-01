'''Reinforcement Learning approach adapted from Lu, et. all
Paper: http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7926576'''
import random
import time
from collections import defaultdict
from copy import copy
from game import *

all_guesses = list(generate_all_targets(4, 6))
def get_guess(i):
    '''gets the guess at index i'''
    return all_guesses[i]

class SARSA(object):

    def __init__(self, learning_rate=0.1, discount=0.9):
        self.q = defaultdict(int)
        self.learning_rate = learning_rate
        self.discount = discount
    
    def choose_action(self, state):
        '''chooses an action based on current state'''
        # TODO see if should consider all actions or just from candidate set
        best_actions = []
        best_q_val = -100000000000000000000
        for i, val in enumerate(state):
            if val == 1:
                action = i#get_guess(i)
                q_val = self.get_q(state, action)
                if q_val > best_q_val:
                    best_q_val = q_val
                    best_actions = [action]
                elif q_val == best_q_val:
                    best_actions.append(action)
        return random.choice(best_actions)

    def get_q(self, state, action):
        state = map(str, state)
        s = ''.join(state) + ' ' + str(action)
        return self.q[s]

    def set_q(self, state, action, value):
        state = map(str, state)
        s = ''.join(state) + ' ' + str(action)
        self.q[s] = value

    def update(self, state, action, reward, next_q):
        this_q = self.get_q(state, action)
        self.set_q(state, action, this_q + self.learning_rate*(reward + self.discount*next_q - this_q))



class Episode(object):

    def __init__(self, sarsa=None, slots=4, options=6, opponent=None):
        self.sarsa = sarsa or SARSA()
        self.state = [1 for i in xrange(options**slots)]
        self.actions_taken = []
        self.qs = []
        self.reward = 0
        self.opponent = opponent or Opponent()
        self.epsilon = 0.1
        self.last_length = 1296
        self._is_done = False

    def _do_action(self):
        # first choose an action (index of guess)
        action = self.sarsa.choose_action(self.state)
        if random.random() < self.epsilon:
            action = random.choice([i for i, val in enumerate(self.state) if val == 1])
        self.actions_taken.append(action)
        guess = get_guess(action)
        q = self.sarsa.get_q(self.state, action)
        self.qs.append(q)

        # take the action
        result = self.opponent.action(guess)
        old_state = copy(self.state)
        self._update_state(guess, result)
        if self.last_length == sum(self.state):
            self._is_done = True
        else:
            self.last_length = sum(self.state)
        return q, old_state, action

    def _update_state(self, guess, result):
        for i, val in enumerate(self.state):
            if val == 1 and validate_attempt(guess, get_guess(i)) != result:
                self.state[i] = 0

    def run(self):
        old_q, old_state, old_action = self._do_action()
        count = 1
        while True:
            next_q, next_state, next_action = self._do_action()
            if self._is_done:
                self.sarsa.update(old_state, old_action, 0, next_q)
                break
            else:
                # penalize not being done
                self.sarsa.update(old_state, old_action, -1, next_q)

        return map(get_guess, self.actions_taken), self.qs



class Opponent(object):

    def __init__(self, answer=None):
        self.game = Mastermind(target=answer, num_pegs=4, num_options=6)

    def action(self, guess):
        return self.game.guess(guess)


class MastermindRL(object):

    def __init__(self, num_epochs=4):
        self.sarsa = SARSA()
        self.num_epochs = num_epochs

    def run(self):
        for epoch in xrange(self.num_epochs):
            start = time.time()
            for i in xrange(6**4):
                answer = get_guess(i)
                opponent = Opponent(answer=answer)
                episode = Episode(sarsa=self.sarsa, opponent=opponent)
                actions_taken, qs = episode.run()

                with open('lu_results{}.txt'.format(epoch), 'a') as f:
                    f.write('Game {}: {}, {}, {}\n'.format(i, answer, actions_taken, qs))
            q = self.sarsa.q
            print 'Epoch {} finished in {}'.format(epoch, time.time()-start)
            print len(self.sarsa.q)
            # with open('epoch{}'.format(epoch), 'w') as f:
            #     f.write(str(self.sarsa.q))


if __name__ == '__main__':
    MastermindRL(num_epochs=250).run()