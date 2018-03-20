import random


NUM_ZOOMBINIS = 16
MAX_MISTAKES = 6

class Zoombini(object):

    def __init__(self, features=None):
        self.has_passed = False
        if features:
            self.hair = features['hair']
            self.eyes = features['eyes']
            self.nose = features['nose']
            self.feet = features['feet']
        else:
            self.hair = random.randint(0, 5)
            self.eyes = random.randint(0, 5)
            self.nose = random.randint(0, 5)
            self.feet = random.randint(0, 5)

    def __str__(self):
        return '<Zoombini object - hair: {} eyes: {} nose: {} feet: {} >'.format(
            self.hair, self.eyes, self.nose, self.feet
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
        return (satisfies and top_true == top) or (not satisfies and top_true != top)


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

    def has_won(self):
        num_passed = sum(map(lambda x: x.has_passed, self.zoombinis))
        return num_passed == len(self.zoombinis)

    def can_move(self):
        return self.mistakes < MAX_MISTAKES


if __name__ == '__main__':
    z = Zoombini()
    print z