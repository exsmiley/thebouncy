import math
import random
from copy import copy
from player import Player
from game import validate_attempt, generate_all_targets


class FiveGuessPlayer(Player):
    '''uses the Knuth 5 Guess algorithm (https://en.wikipedia.org/wiki/Mastermind_(board_game))
    => just minimax so never terminates for big problems...'''

    def __init__(self, num_pegs=4, num_options=6):
        super(FiveGuessPlayer, self).__init__(num_pegs=num_pegs, num_options=num_options)
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
            return self.strategy()

    def add_feedback(self, guess, feedback):
        super(FiveGuessPlayer, self).add_feedback(guess, feedback)
        
        # only keep answers that give the same feedback
        still_remaining = []
        for target in self.remaining_answers:
            if validate_attempt(target, guess) == feedback:
                still_remaining.append(target)
        self.remaining_answers = still_remaining

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

    def __init__(self, num_pegs=4, num_options=10):
        super(SwaszekPlayer, self).__init__(num_pegs=num_pegs, num_options=num_options)
        self._setup()

    def _setup(self):
        self.all_possible = list(generate_all_targets(self.num_pegs, self.num_options))#all_possible
        self.remaining_answers = self.all_possible

    def make_guess(self):
        return random.choice(self.remaining_answers)

    def add_feedback(self, guess, feedback):
        super(SwaszekPlayer, self).add_feedback(guess, feedback)

        # only keep answers that give the same feedback
        still_remaining = []
        for target in self.remaining_answers:
            if validate_attempt(target, guess) == feedback:
                still_remaining.append(target)
        self.remaining_answers = still_remaining
        # print 'made {} guesses with {}/{} remaining'.format(len(self.attempts), len(self.remaining_answers), len(self.all_possible))

    def reset(self):
        super(SwaszekPlayer, self).reset()
        self._setup()


class EntropyPlayer(FiveGuessPlayer):
    '''uses the Max Entropy strategy
    TODO: It's broken....'''

    def strategy(self):
        possible_scores = [(i,j) for i in xrange(self.num_pegs) for j in xrange(self.num_pegs) if i >= j]
        best_entropy = 0
        best_guesses = []
        remaining = set(self.remaining_answers)
        for i, guess in enumerate(self.all_possible):
            eliminated_counts = []
            for score in possible_scores:
                eliminated_count = 0
                for answer in self.remaining_answers:
                    prob = validate_attempt(guess, answer) != score:
                        eliminated_count += 1
                eliminated_counts.append(eliminated_count)

            total_entropy = 0
            probs = map(lambda x: 1.0*x/len(self.remaining_answers), eliminated_counts)
            for prob in probs:
                if prob > 0:
                    total_entropy += -1*prob*math.log(prob, 2)

            # if i % (len(self.all_possible)/10) == 0:
            #     print '{}/{}'.format(i, len(self.all_possible)), best_guess, best_count

            if total_entropy > best_entropy and tuple(guess) not in self.used:
                best_count = eliminated_count
                best_guesses = [guess]
            elif total_entropy == best_entropy and tuple(guess) not in self.used:
                best_guesses.append(guess)

        # prioritize guesses that are in the remaining answers
        potential_win_guesses = [guess for guess in best_guesses if guess in remaining]

        if len(potential_win_guesses) > 0:
            return random.choice(potential_win_guesses)
        else:
            return random.choice(best_guesses)



if __name__ == '__main__':
    p = FiveGuessPlayer(num_pegs=4, num_options=3)
    print p.make_guess()