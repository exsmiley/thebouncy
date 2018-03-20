import random
import numpy as np
import itertools


NUM_ZOOMBINIS = 16
MAX_MISTAKES = 6

ZOOMBINI_VECTOR_LENGTH = 1 + 5*4 + 2 # first bit is unknown bit
CANDIDATE_LENGTH = 5*4
INPUT_LENGTH = (NUM_ZOOMBINIS)*ZOOMBINI_VECTOR_LENGTH + CANDIDATE_LENGTH
FEEDBACK_LENGTH = 2


class Zoombini(object):

    def __init__(self, features=None):
        self.has_passed = False
        self.is_top = None
        if features:
            self.hair = features['hair']
            self.eyes = features['eyes']
            self.nose = features['nose']
            self.feet = features['feet']
        else:
            self.hair = random.randint(0, 4)
            self.eyes = random.randint(0, 4)
            self.nose = random.randint(0, 4)
            self.feet = random.randint(0, 4)

    def get_vector(self):
        vec = [0 for i in xrange(ZOOMBINI_VECTOR_LENGTH)]
        if self.is_top is None:
            vec[0] = 1
        else:
            vec[1+self.hair] = 1
            vec[6+self.eyes] = 1
            vec[11+self.nose] = 1
            vec[16+self.feet] = 1

            # feedback
            if self.is_top:
                vec[21] = 1
            else:
                vec[22] = 1
        return vec

    def get_candidate_vector(self):
        vec = [0 for i in xrange(CANDIDATE_LENGTH)]
        vec[self.hair] = 1
        vec[5+self.eyes] = 1
        vec[10+self.nose] = 1
        vec[15+self.feet] = 1
        return vec

    def __str__(self):
        return '<Zoombini object - hair: {} eyes: {} nose: {} feet: {} has_passed: {} >'.format(
            self.hair, self.eyes, self.nose, self.feet, self.has_passed
        )


class Bridge(object):

    def __init__(self):
        # generate the condition
        self.attr = random.choice(['hair', 'eyes', 'nose', 'feet'])
        self.attr_num = random.randint(0, 5)
        self.top_true = random.randint(0, 1)


    def check_pass(self, zoombini, top):
        '''returns true if the zoombini can cross the bridge

        Args:
            zoombini: Zoombini instance
            top: boolean saying if it's trying to go top
        '''
        satisfies = getattr(zoombini, self.attr) == self.attr_num
        return (satisfies and self.top_true == top) or (not satisfies and self.top_true != top)

    def __str__(self):
        bridge_spot = 'Top' if self.top_true else 'Bottom'
        return 'Bridge Condition: {} bridge says only {} num {}'.format(bridge_spot, self.attr, self.attr_num)


class Game(object):
    '''Game based on the Zoombini's Allergic Cliffs level'''

    def __init__(self, zoombinis=None, bridge=None):
        if zoombinis:
            self.zoombinis = zoombinis
        else:
            # init zoombinis randomly
            self.zoombinis = [Zoombini() for _ in xrange(NUM_ZOOMBINIS)]

        self.new_game(bridge)
        self.mistakes = 0

    def new_game(self, bridge=None):
        if bridge:
            self.bridge = bridge
        else:
            self.bridge = Bridge()

    def send_zoombini(self, index, top):
        zoombini = self.zoombinis[index]
        if self.bridge.check_pass(zoombini, top):
            zoombini.has_passed = True
        else:
            self.mistakes += 1

        # collected information about the zoombini
        zoombini.is_top = (zoombini.has_passed and top) or (not zoombini.has_passed and not top)

        return zoombini.has_passed

    def has_won(self):
        num_passed = sum(map(lambda x: x.has_passed, self.zoombinis))
        return num_passed == len(self.zoombinis)

    def can_move(self):
        return self.mistakes < MAX_MISTAKES and not self.has_won()

    def get_state_vector(self, candidate_index):
        # TODO figure out how to sort zoombinis for "set" behavior
        zoombinis_vecs = map(lambda x: x.get_vector(), self.zoombinis)
        vec = list(itertools.chain.from_iterable(zoombinis_vecs))
        vec += self.zoombinis[candidate_index].get_candidate_vector()
        return vec

    def __str__(self):
        return ('Zoombini Game' +
        '\n{}'.format(str(self.bridge)) +
        '\nZoombinis: {}'.format(map(str, self.zoombinis)) +
        '\nNum Mistakes: {}'.format(self.mistakes)
    )


if __name__ == '__main__':
    g = Game()
    print g
