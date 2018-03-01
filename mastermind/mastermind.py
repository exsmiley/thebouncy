import random
import itertools
import numpy as np


NUM_PEGS = 4
NUM_OPTIONS = 6
ENCODER_VECTOR_LENGTH = NUM_PEGS*NUM_OPTIONS + (NUM_PEGS+1)*2
EMBEDDED_LENGTH = 50


def generate_all_targets(num_pegs, num_options):
    '''returns iterator over all targets'''
    return itertools.product(range(num_options), repeat=num_pegs)


# TODO maybe figure out how to generate this in the function and cache it
ALL_GUESSES = list(generate_all_targets(NUM_PEGS, NUM_OPTIONS))
def get_guess(i):
    '''gets the guess at index i'''
    return ALL_GUESSES[i]

def random_guess():
    return random.choice(ALL_GUESSES)


def guess_to_vector(guess):
    vec = [0 for i in xrange(NUM_OPTIONS)]*NUM_PEGS

    for i, option in enumerate(guess):
        index = i*NUM_OPTIONS + option
        vec[index] = 1

    return vec


def feedback_to_vector(feedback):
    num_exist, num_match = feedback

    vec = [0 for i in xrange(NUM_PEGS+1)]*2

    vec[num_exist] = 1
    vec[num_match+NUM_PEGS+1] = 1

    return vec


def unencoded_vector(guess, feedback):
    return guess_to_vector(guess) + feedback_to_vector(feedback)


def random_numbers():
    options = range(NUM_OPTIONS)
    return [random.choice(options) for i in xrange(NUM_PEGS)]

def get_counts(arr):
    '''get counts for each of the numbers'''
    counts = {i: 0 for i in xrange(NUM_OPTIONS)}

    for num in arr:
        counts[num] += 1

    return counts


def validate_attempt(target, attempt):
    '''attempt is an array trying to match the target'''
    num_exist = 0
    num_match = 0

    # first compare counts to get num_exist
    target_counts = get_counts(target)

    for num in attempt:
        if target_counts[num] > 0:
            target_counts[num] -= 1
            num_exist += 1

    # compare arrays to get num_match
    for i in xrange(len(attempt)):
        if attempt[i] == target[i]:
            num_match += 1

    return num_exist, num_match


class Mastermind(object):

    def __init__(self, target=None):
        '''target is the goal array to match'''
        if target is None:
            target = random_numbers()
        self.target = target
        assert len(target) == NUM_PEGS
        self.num_pegs = NUM_PEGS
        self.num_options = NUM_OPTIONS

    def guess(self, attempt):
        return validate_attempt(self.target, attempt)

    def is_winning_feedback(self, feedback):
        return feedback[1] == NUM_PEGS


if __name__ == '__main__':
    # game = Mastermind([4, 2, 3, 4])
    # print game.guess([4, 5, 6, 7]), (1, 1)
    # print game.guess([4, 3, 3, 1]), (2, 2)

    target = [2, 4, 2, 5]
    # game = Mastermind(target)
    # print game.guess(target)

    guess = [3, 4, 1, 2]
    print validate_attempt(target, guess) == validate_attempt(guess, target)
    import time
    start = time.time()
    for i in xrange(1296):
        validate_attempt(target, guess)
    t = time.time()-start
    print t
    print t*1296
