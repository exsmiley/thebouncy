import random
from player import Player


class RandomPlayer(Player):

    def make_guess(self):
        options = range(self.num_options)
        guess = [random.choice(options) for i in xrange(self.num_pegs)]
        if tuple(guess) in self.used:
            return self.make_guess()
        else:
            return guess
