import random
from player import Player
from solver import solve


class RandomPlayer(Player):

    def make_guess(self):
        options = range(self.num_options)
        guess = [random.choice(options) for i in xrange(self.num_pegs)]
        if tuple(guess) in self.used:
            return self.make_guess()
        else:
            return guess


class RandomPlayerSolver(Player):

    def make_guess(self):
        if len(self.attempts) >= 7:
            return self._educated_guess()
        else:
            return self._random_guess()

    def _random_guess(self):
        options = range(self.num_options)
        guess = [random.choice(options) for i in xrange(self.num_pegs)]
        if tuple(guess) in self.used:
            return self._random_guess()
        else:
            return guess

    def _educated_guess(self):
        '''uses the solver with all of its guesses to try to make a better guess'''
        return solve(
            num_pegs=self.num_pegs, num_options=self.num_options,
            feedbacks=self.attempts
        )
