import random
import time
from copy import copy
from collections import defaultdict
from game import *


NUM_PEGS = 4
NUM_OPTIONS = 6
all_guesses = list(generate_all_targets(NUM_PEGS, NUM_OPTIONS))
def get_guess(i):
    '''gets the guess at index i'''
    return all_guesses[i]


class Episode(object):

    def __init__(self, state_manager, opponent, train=True):
        self.state_manager = state_manager
        self.opponent = opponent
        self.state = self.state_manager.get_starting_state()
        self.trace = [] # list of (state, action, result)
        self.epsilon = 0.1
        self.train = train
        if not train:
            self.epsilon = 0

    def run(self):
        # keep running until is done
        count = 0
        while not self.opponent.is_finished and count < 10:
            count += 1
            state = copy(self.state)
            action = self.state_manager.get_action(state, epsilon=self.epsilon)
            result = self.opponent.action(action)
            self.trace.append((state, action, result))
            self.state = self.state_manager.update_state(state, action, result)

        # now update all of the states
        if self.train:
            self.state_manager.learn(self.trace)
        return self.trace


class Opponent(object):

    def __init__(self, answer=None):
        self.game = Mastermind(target=answer, num_pegs=NUM_PEGS, num_options=NUM_OPTIONS)
        self.is_finished = False

    def action(self, guess):
        guess = get_guess(guess)
        result = self.game.guess(guess)
        if result == (NUM_PEGS, NUM_PEGS):
            self.is_finished = True
        return result


class MastermindRL(object):

    def __init__(self, state_manager, episode=Episode, num_epochs=1, log_name='results{}.txt'):
        self.state_manager = state_manager
        self.episode = episode
        self.num_epochs = num_epochs
        self.log_name = log_name

    def train(self):
        for epoch in xrange(self.num_epochs):
            start = time.time()
            for i in xrange(NUM_OPTIONS**NUM_PEGS):
                answer = get_guess(i)
                self._run_episode(answer, train=True, i=i, epoch=epoch)
            print 'Epoch {} finished in {}'.format(epoch, time.time()-start)
            print 'Size of dict:', len(self.state_manager.q)

    def _run_episode(self, answer, train=True, i=0, epoch=0):
        opponent = Opponent(answer=answer)
        episode = self.episode(self.state_manager, opponent, train=train)
        actions_taken = map(lambda x: get_guess(x[1]), episode.run())
        with open(self.log_name.format(epoch), 'a') as f:
            f.write('Game {}: {}, {}\n'.format(i, answer, actions_taken))

    def test(self):
        start = time.time()
        total_taken = 0.
        for i in xrange(NUM_OPTIONS**NUM_PEGS):
            answer = get_guess(i)
            opponent = Opponent(answer=answer)
            episode = self.episode(self.state_manager, opponent, train=False)
            actions_taken = map(lambda x: get_guess(x[1]), episode.run())
            total_taken += len(actions_taken)
        print 'Test finished in {}'.format(time.time()-start)
        print 'Average Number of Turns:', total_taken/NUM_OPTIONS**NUM_PEGS




