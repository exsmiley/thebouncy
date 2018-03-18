import random
from player import Player
from mastermind import random_guess
from solver import MastermindSolver


class RandomPlayer(Player):

    def make_guess(self):
        guess = random_guess()
        if tuple(guess) in self.used:
            return self.make_guess()
        else:
            return guess

class RandomSolverPlayer(RandomPlayer):
    def __init__(self):
        super(RandomSolverPlayer, self).__init__()
        self.solver = MastermindSolver()

    def make_guess(self):
        if len(self.attempts) > 3:
            is_unique, soln = self.solver.is_unique_solution()
            
            if is_unique:
                # print "solver solved...", soln
                return soln

        guess = super(RandomSolverPlayer, self).make_guess()
        return guess

    def add_feedback(self, guess, feedback):
        # overwrite super since store slightly differently
        self.used.add(tuple(guess))
        self.attempts.append((guess, feedback))
        self.solver.add_feedback((guess, feedback))

    def reset(self):
        super(RandomSolverPlayer, self).reset()
        self.solver.reset()
