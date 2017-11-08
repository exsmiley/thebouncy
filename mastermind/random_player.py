import random
from game import Mastermind
from solver import solve


class RandomPlayer(object):

    def __init__(self, game):
        self.game = game
        self.num_pegs = game.num_pegs
        self.num_options = game.num_options
        self.guesses = []

    def make_guess(self):
        options = range(self.num_options)
        guess = [random.choice(options) for i in xrange(self.num_pegs)]
        exist, match = game.guess(guess)
        guess = (guess, exist, match)
        self.guesses.append(guess)
        return guess

    def make_x_guesses(self, x):
        '''makes x number of guesses'''
        for _ in xrange(x):
            self.make_guess()
        return self.guesses

    def educated_guess(self):
        '''uses the solver with all of its guesses to try to make a better guess'''
        guess = solve(
            num_pegs=self.num_pegs, num_options=self.num_options,
            feedbacks=self.guesses
        )
        exist, match = game.guess(guess)
        guess = (guess, exist, match)
        self.guesses.append(guess)
        return guess


if __name__ == '__main__':
    game = Mastermind()
    print 'target:', game.target
    player = RandomPlayer(game)
    print 'random 7 guesses:', player.make_x_guesses(7)
    print 'smart guess:', player.educated_guess()
