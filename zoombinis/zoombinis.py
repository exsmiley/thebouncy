import random
import numpy as np
import itertools


NUM_ZOOMBINIS = 2
MAX_MISTAKES = 1
NUM_BRIDGES = 2

ZOOMBINI_AGENT_VECTOR_LENGTH = 5*4 + 2*NUM_BRIDGES
ZOOMBINI_BRAIN_VECTOR_LENGTH = 5*4 + 3*NUM_BRIDGES
AGENT_INPUT_LENGTH = (NUM_ZOOMBINIS)*ZOOMBINI_AGENT_VECTOR_LENGTH
BRAIN_INPUT_LENGTH = (NUM_ZOOMBINIS)*ZOOMBINI_BRAIN_VECTOR_LENGTH
OUTPUT_LENGTH = (NUM_ZOOMBINIS)*(NUM_BRIDGES)
FEEDBACK_LENGTH = 2


class Zoombini(object):

    def __init__(self, features=None):
        self.has_passed = False
        self.rejected_bridges = []
        self.accepted_bridge = None
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

    def get_agent_vector(self):
        # actual format is [hair|eyes|nose|feet|rejections|accepted]
        vec = [0 for i in range(ZOOMBINI_AGENT_VECTOR_LENGTH)]

        vec[self.hair] = 1
        vec[5+self.eyes] = 1
        vec[10+self.nose] = 1
        vec[15+self.feet] = 1

        # feedback
        for i in self.rejected_bridges:
            vec[20+i] = 1

        if self.accepted_bridge is not None:
            vec[20+NUM_BRIDGES+self.accepted_bridge] = 1

        return vec

    def get_brain_vector(self):
        vec = [0 for i in range(ZOOMBINI_BRAIN_VECTOR_LENGTH)]

        vec[self.hair] = 1
        vec[5+self.eyes] = 1
        vec[10+self.nose] = 1
        vec[15+self.feet] = 1

        for i in range(NUM_BRIDGES):
            if i == self.accepted_bridge:
                vec[20+i*3] = 1
            elif i in self.rejected_bridges:
                vec[20+i*3+1] = 1
            else:
                vec[20+i*3+2] = 1

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

    def zoombini_bridge(self, zoombini):
        for i in range(NUM_BRIDGES):
            if self.check_pass(zoombini, i):
                return i

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
            self.zoombinis = [Zoombini() for _ in range(NUM_ZOOMBINIS)]

        self.new_game(bridge)
        self.truth = list(map(lambda z: self.bridge.zoombini_bridge(z), self.zoombinis))
        self.mistakes = 0

    def new_game(self, bridge=None):
        if bridge:
            self.bridge = bridge
        else:
            self.bridge = Bridge()

    def send_zoombini(self, index, top):
        zoombini = self.zoombinis[index]
        if self.bridge.check_pass(zoombini, top) and not zoombini.has_passed:
            zoombini.has_passed = True
            zoombini.accepted_bridge = top
        # can't repeat sending same zoombini
        elif zoombini.has_passed:
            self.mistakes += 1
            return False
        else:
            self.mistakes += 1
            zoombini.rejected_bridges.append(top)

        return zoombini.has_passed

    def has_won(self):
        num_passed = sum(map(lambda x: x.has_passed, self.zoombinis))
        return num_passed == len(self.zoombinis)

    def score(self):
        return sum(map(lambda x: x.has_passed, self.zoombinis))

    def can_move(self):
        return self.mistakes <= MAX_MISTAKES and not self.has_won()

    def get_agent_state(self):
        zoombinis_vecs = map(lambda x: x.get_agent_vector(), self.zoombinis)
        vec = list(itertools.chain.from_iterable(zoombinis_vecs))
        return vec

    def get_brain_state(self):
        # TODO figure out how to sort to keep zoombinis consistent with truth
        zoombinis_vecs = map(lambda x: x.get_brain_vector(), self.zoombinis)
        vec = list(itertools.chain.from_iterable(zoombinis_vecs))
        return vec

    def get_brain_truth(self):
        return self.truth

    def __str__(self):
        return ('Zoombini Game' +
        '\n{}'.format(str(self.bridge)) +
        '\nZoombinis: {}'.format(list(map(str, self.zoombinis))) +
        '\nNum Mistakes: {}'.format(self.mistakes)
    )

class GameEnv(object):

    def __init__(self):
        self.game = Game()
        self.state_size = AGENT_INPUT_LENGTH
        self.action_size = NUM_ZOOMBINIS*NUM_BRIDGES
        self.actions = set()

    def reset(self):
        # print('Start game')
        self.game = Game()
        self.actions = set()
        return np.array(self.game.get_agent_state())#.reshape(1, -1)

    def step(self, action, verbose=False):
        # action is an int
        # action/num_bridges is the zoombini, action % num_bridges is the bridge
        self.actions.add(action)
        zoombini = action//NUM_BRIDGES
        bridge = action % NUM_BRIDGES

        passed = self.game.send_zoombini(zoombini, bridge)

        state = np.array(self.game.get_agent_state())#.reshape(1, -1)
        reward = 1 if passed else 0
        done = not self.game.can_move()

        if verbose:
            passed_str = 'PASSED' if passed else 'failed'
            print('Sending Zoombini {} to {} and it {}'.format(zoombini, bridge, passed_str))

        return state, reward, done

    def check_already_made(self, action):
        return action in self.actions

if __name__ == '__main__':
    g = Game()
    print(g)
    print(g.send_zoombini(0, 1))
    print(g.can_move(), 'move')
    print(g.send_zoombini(0, 0))
    print(g.can_move(), 'move')
    print(g.send_zoombini(1, 1))
    print(g.can_move(), 'move')
    print(g.send_zoombini(1, 0))
    print(g.can_move(), 'move')
