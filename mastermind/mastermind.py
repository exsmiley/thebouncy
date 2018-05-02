import random
import itertools
import numpy as np


NUM_PEGS = 4
NUM_OPTIONS = 6
ENCODER_VECTOR_LENGTH = NUM_PEGS*NUM_OPTIONS + NUM_PEGS + NUM_PEGS*(NUM_PEGS+1)+2 # add some for the unknown bit
EMBEDDED_LENGTH = ENCODER_VECTOR_LENGTH  #50


BRAIN_INPUT_LENGTH = 10*ENCODER_VECTOR_LENGTH
OUTPUT_LENGTH = NUM_PEGS*NUM_OPTIONS

def generate_all_targets(num_pegs, num_options):
    '''returns iterator over all targets'''
    return itertools.product(range(num_options), repeat=num_pegs)


# TODO maybe figure out how to generate this in the function and cache it
ALL_GUESSES = list(generate_all_targets(NUM_PEGS, NUM_OPTIONS))
NUM_ALL_GUESSES = len(ALL_GUESSES)
def get_guess(i):
    '''gets the guess at index i'''
    return ALL_GUESSES[i]

def random_guess():
    return random.choice(ALL_GUESSES)


def guess_to_vector(guess):
    vec = [0 for i in range(NUM_OPTIONS+1)]*NUM_PEGS

    for i, option in enumerate(guess):
        index = i*(NUM_OPTIONS+1) + option + 1
        vec[index] = 1

    return vec


def truth_to_vector(truth):
    vec = [0 for i in range(OUTPUT_LENGTH)]

    for i, option in enumerate(truth):
        index = i*(NUM_OPTIONS) + option
        vec[index] = 1

    return vec


def feedback_to_vector(feedback):
    num_exist, num_match = feedback

    vec = [0 for i in range(NUM_PEGS*(NUM_PEGS+1)+2)]
    # print feedback, num_exist+4*num_match+1, len(vec)
    vec[num_exist+4*num_match+1] = 1

    return vec

def feedback_index(feedback):
    num_exist, num_match = feedback
    return num_exist+4*num_match+1


def unencoded_vector(guess, feedback):
    return guess_to_vector(guess) + feedback_to_vector(feedback)


def random_numbers():
    options = range(NUM_OPTIONS)
    return [random.choice(options) for i in range(NUM_PEGS)]

def get_counts(arr):
    '''get counts for each of the numbers'''
    counts = {i: 0 for i in range(NUM_OPTIONS)}

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
    for i in range(len(attempt)):
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

    def get_brain_truth(self):
        return truth_to_vector(self.target)


class MastermindEnv(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.guess_feedbacks = []
        self.won = False
        self.game = Mastermind()
        self.forbidden = set()
        return 0, []

    def forbid(self):
        return self.forbidden

    def win(self):
        return self.won

    def can_move(self):
        return self.won or len(self.guess_feedbacks) < 10

    def step(self, action):
        guess = ALL_GUESSES[i]
        self.forbidden.add(action)
        feedback = self.game.guess(guess)
        self.won = self.game.is_winning_feedback(feedback)
        reward = 1 if self.won else 0
        self.guess_feedbacks.append((action, feedback))
        return (len(self.guess_feedbacks), self.guess_feedbacks), reward, not self.can_move()



class StateXform:
  def __init__(self):
    self.length = BRAIN_INPUT_LENGTH
  def state_to_np(self, time_state):
    _, action_feedback = time_state
    action_feedback = action_feedback[:10]
    state = np.zeros(shape=(1, self.length))
    for i, (action, feedback) in enumerate(action_feedback):
        state[:,(i)*EMBEDDED_LENGTH:(i+1)*EMBEDDED_LENGTH] = np.array(unencoded_vector(action, feedback)).reshape(-1, ENCODER_VECTOR_LENGTH)
    return state

class ActionXform:
  def __init__(self):
    self.possible_actions = list(range(NUM_ALL_GUESSES))
    self.length = NUM_ALL_GUESSES
  def idx_to_action(self, idx):
    return get_guess(idx)
  def action_to_idx(self, a):
    # TODO optimize
    for i, g in enumerate(ALL_GUESSES):
        if g == a:
            return i
    return -1
  def action_to_1hot(self, a):
    ret = np.zeros(L*L)
    ret[a] = 1.0
    return ret


if __name__ == '__main__':
    # game = Mastermind([4, 2, 3, 4])
    # print game.guess([4, 5, 6, 7]), (1, 1)
    # print game.guess([4, 3, 3, 1]), (2, 2)
    print([(x, i) for (x, i) in enumerate(ALL_GUESSES)])
    target = [2, 4, 2, 5]
    game = Mastermind(target)
    print(game.get_brain_truth())
    # print game.guess(target)

    guess = [3, 4, 1, 2]
    print(validate_attempt(target, guess) == validate_attempt(guess, target))
