import math
import random
from copy import copy
import tqdm
from player import Player
from mastermind import validate_attempt, generate_all_targets
from solver import solve


class FiveGuessPlayer(Player):
    '''uses the Knuth 5 Guess algorithm (https://en.wikipedia.org/wiki/Mastermind_(board_game))
    => just minimax so never terminates for big problems...'''

    def __init__(self):
        super(FiveGuessPlayer, self).__init__()
        self._setup()

    def _setup(self):
        self.all_possible = list(generate_all_targets(self.num_pegs, self.num_options))
        self.remaining_answers = copy(self.all_possible)

    def make_guess(self):
        if len(self.attempts) == 0 and self.num_pegs % 2 == 0:
            return [0 for i in xrange(int(math.floor(self.num_pegs/2.)))] + [1 for i in xrange(int(math.ceil(self.num_pegs/2.)))]
        if len(self.remaining_answers) == 1:
            return self.remaining_answers[0]
        else:
            # print 'strategy', len(self.remaining_answers), len(self.all_possible)
            return self.strategy()

    def add_feedback(self, guess, feedback):
        super(FiveGuessPlayer, self).add_feedback(guess, feedback)
        
        # only keep answers that give the same feedback
        still_remaining = []
        for target in self.remaining_answers:
            if validate_attempt(target, guess) == feedback:
                still_remaining.append(target)
        self.remaining_answers = still_remaining
        print 'made {} guesses with {}/{} remaining'.format(len(self.attempts), len(self.remaining_answers), len(self.all_possible))


    def strategy(self):
        '''runs minimax'''
        possible_scores = [(i,j) for i in xrange(self.num_pegs) for j in xrange(self.num_pegs) if i >= j]
        best_count = -1
        best_guesses = []
        remaining = set(self.remaining_answers)
        for i, guess in enumerate(self.all_possible):
            eliminated_counts = []
            for score in possible_scores:
                eliminated_count = 0
                for answer in self.remaining_answers:
                    if validate_attempt(guess, answer) != score:
                        eliminated_count += 1
                eliminated_counts.append(eliminated_count)
            eliminated_count = min(eliminated_counts)

            # if i % (len(self.all_possible)/10) == 0:
            #     print '{}/{}'.format(i, len(self.all_possible)), best_guess, best_count

            if eliminated_count > best_count and tuple(guess) not in self.used:
                best_count = eliminated_count
                best_guesses = [guess]
            elif eliminated_count == best_count and tuple(guess) not in self.used:
                best_guesses.append(guess)

        # prioritize guesses that are in the remaining answers
        potential_win_guesses = [guess for guess in best_guesses if guess in remaining]

        if len(potential_win_guesses) > 0:
            return random.choice(potential_win_guesses)
        else:
            return random.choice(best_guesses)

    def reset(self):
        super(FiveGuessPlayer, self).reset()
        self._setup()


class SwaszekPlayer(Player):
    '''uses Swaszek's strategy of enumeration + filtering + choose random'''

    def __init__(self):
        super(SwaszekPlayer, self).__init__()
        self._setup()

    def _setup(self):
        self.all_possible = list(generate_all_targets(self.num_pegs, self.num_options))#all_possible
        self.remaining_answers = self.all_possible

    def make_guess(self):
        return random.choice(self.remaining_answers)

    def add_feedback(self, guess, feedback):
        # print tuple(guess) in self.remaining_answers
        super(SwaszekPlayer, self).add_feedback(guess, feedback)

        # only keep answers that give the same feedback
        still_remaining = []
        for target in self.remaining_answers:
            if validate_attempt(target, guess) == feedback:
                still_remaining.append(target)
        self.remaining_answers = still_remaining

    def reset(self):
        super(SwaszekPlayer, self).reset()
        self._setup()


class MaxEntropyPlayer(FiveGuessPlayer):
    '''uses the Max Entropy strategy'''

    def make_guess(self):
        if len(self.attempts) == 0 and self.num_pegs < self.num_options:
            return [i for i in xrange(self.num_pegs)]
        if len(self.remaining_answers) == 1:
            return self.remaining_answers[0]
        else:
            return self.strategy()

    def strategy(self):
        possible_scores = [(i,j) for i in xrange(self.num_pegs) for j in xrange(self.num_pegs) if i >= j]
        max_entropy = 0
        best_guesses = []
        remaining = set(self.remaining_answers)
        for i, guess in enumerate(self.all_possible):
            entropy = 0
            for score in possible_scores:
                part_size = 0
                for answer in self.remaining_answers:
                    if validate_attempt(guess, answer) != score:
                        part_size += 1
                
                if part_size > 0:
                    I = math.log(1.0*len(self.remaining_answers)/part_size)/math.log(2)
                    p = 1.0*part_size/len(self.remaining_answers)
                    entropy += I*p


            if entropy > max_entropy and tuple(guess) not in self.used:
                max_entropy = entropy
                best_guesses = [guess]
            elif entropy == max_entropy and tuple(guess) not in self.used:
                best_guesses.append(guess)

        # prioritize guesses that are in the remaining answers
        potential_win_guesses = [guess for guess in best_guesses if guess in remaining]

        if len(potential_win_guesses) > 0:
            return random.choice(potential_win_guesses)
        else:
            return random.choice(best_guesses)


class MaxPartsPlayer(FiveGuessPlayer):
    '''uses the Max Parts strategy'''
    def make_guess(self):
        if len(self.attempts) == 0 and self.num_pegs < self.num_options:
            return [i for i in xrange(self.num_pegs)]
        if len(self.remaining_answers) == 1:
            return self.remaining_answers[0]
        else:
            return self.strategy()

    def strategy(self):
        possible_scores = [(i,j) for i in xrange(self.num_pegs) for j in xrange(self.num_pegs) if i >= j]
        max_parts = 0
        best_guesses = []
        remaining = set(self.remaining_answers)
        for i, guess in enumerate(self.all_possible):
            part_size = 0
            for score in possible_scores:
                for answer in self.remaining_answers:
                    if validate_attempt(guess, answer) != score:
                        part_size += 1

            if part_size > max_parts and tuple(guess) not in self.used:
                max_parts = part_size
                best_guesses = [guess]
            elif part_size == max_parts and tuple(guess) not in self.used:
                best_guesses.append(guess)

        # prioritize guesses that are in the remaining answers
        potential_win_guesses = [guess for guess in best_guesses if guess in remaining]

        if len(potential_win_guesses) > 0:
            return random.choice(potential_win_guesses)
        else:
            return random.choice(best_guesses)


class CEGISPlayer(Player):

    def make_guess(self):
        if len(self.attempts) >= 1:
            return self._educated_guess()
        else:
            first_half = self.num_pegs/2 if self.num_pegs % 2 == 0 else self.num_pegs/2+1
            return [1 for i in xrange(first_half)] + [2 for i in xrange(self.num_pegs/2)] #self._random_guess()

    def _random_guess(self):
        options = range(self.num_options)
        guess = [random.choice(options) for i in xrange(self.num_pegs)]
        if tuple(guess) in self.used:
            return self._random_guess()
        else:
            return guess

    def _educated_guess(self):
        '''uses the solver with all of its guesses to try to make a better guess'''
        attempts = [(x, (y, z)) for (x, y, z) in self.attempts]
        return solve(
            num_pegs=self.num_pegs, num_options=self.num_options,
            feedbacks=attempts
        )


if __name__ == '__main__':
    # p = FiveGuessPlayer()
    # print p.make_guess()
    p = MaxEntropyPlayer()
    p.add_feedback([0,1,2,3], (1,0))
    p.add_feedback([3,4,3,2], (0,0))
    p.add_feedback([2,5,5,5], (2,1))
    print p.make_guess()
    p.add_feedback([5,0,4,5], (3,1))
    # p.add_feedback([0,1,4,0], (2,1))
    # p.add_feedback([4,0,1,5], (2,0))
    # p.add_feedback([2,4,1,2], (0,0))
    # p.add_feedback([3,2,5,4], (1,0))
    # p.add_feedback([2,0,5,5], (3,0))
    # p.add_feedback([4,5,5,5], (2,1))
    # p.add_feedback([5,5,5,1], (2,2))
