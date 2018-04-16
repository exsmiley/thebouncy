import random
import numpy as np
import itertools
import functools


NUM_ZOOMBINIS = 16
MAX_MISTAKES = 6
NUM_BRIDGES = 2

ZOOMBINI_AGENT_VECTOR_LENGTH = 5*4 + 2*NUM_BRIDGES
ZOOMBINI_BRAIN_VECTOR_LENGTH = 5*4 + 3*NUM_BRIDGES
AGENT_INPUT_LENGTH = (NUM_ZOOMBINIS)*ZOOMBINI_AGENT_VECTOR_LENGTH
BRAIN_INPUT_LENGTH = (NUM_ZOOMBINIS)*ZOOMBINI_BRAIN_VECTOR_LENGTH
OUTPUT_LENGTH = (NUM_ZOOMBINIS)*(NUM_BRIDGES)
FEEDBACK_LENGTH = 2


class Zoombini(object):

    def __init__(self, features=None):
        self.reset()
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
            elif i in self.rejected_bridges or self.has_passed:
                vec[20+i*3+1] = 1
            # unknown
            else:
                vec[20+i*3+2] = 1

        return vec

    def json(self):
        return {
            "hair": self.hair,
            "eyes": self.eyes,
            "nose": self.nose,
            "feet": self.feet,
            "has_passed": self.has_passed,
            "rejected": self.rejected_bridges,
            "accepted": self.accepted_bridge
        }

    def reset(self):
        self.has_passed = False
        self.rejected_bridges = []
        self.accepted_bridge = None

    def __str__(self):
        return '<Zoombini object - hair: {} eyes: {} nose: {} feet: {} has_passed: {} rejected: {} accepted: {}>'.format(
            self.hair, self.eyes, self.nose, self.feet, self.has_passed, self.rejected_bridges, self.accepted_bridge
        )


def sort_zoombinis(zoombinis):
    def sort_func(a, b):
        if a.hair != b.hair:
            return a.hair - b.hair
        if a.eyes != b.eyes:
            return a.eyes - b.eyes
        if a.nose != b.eyes:
            return a.nose - b.nose
        return a.feet - b.feet
    return sorted(zoombinis, key=functools.cmp_to_key(sort_func))


class Bridge(object):

    def __init__(self):
        # generate the conditions
        self.else_bridge = random.randint(0, NUM_BRIDGES-1)
        conds = set()
        self.bridges = {}
        for i in range(NUM_BRIDGES):
            if i != self.else_bridge:
                cond = self.gen_condition()
                while str(cond) in conds:
                    cond = self.gen_condition()
                self.bridges[i] = cond
                conds.add(str(cond))

    def gen_condition(self):
        attr = random.choice(['hair', 'eyes', 'nose', 'feet'])
        num = random.randint(0, 5)
        return {'attr': attr, 'num': num}


    def check_pass(self, zoombini, bridge):
        '''returns true if the zoombini can cross the bridge

        Args:
            zoombini: Zoombini instance
            top: boolean saying if it's trying to go top
        '''
        if bridge == self.else_bridge:
            # iterate through all and check
            for i in range(NUM_BRIDGES):
                if i == self.else_bridge:
                    continue
                attr = self.bridges[i]['attr']
                num = self.bridges[i]['num']
                if getattr(zoombini, attr) == num:
                    return False
            return True
        else:
            # only need to check one
            attr = self.bridges[bridge]['attr']
            num = self.bridges[bridge]['num']
            return getattr(zoombini, attr) == num


    def zoombini_bridge(self, zoombini):
        for i in range(NUM_BRIDGES):
            if self.check_pass(zoombini, i):
                return i

    def __str__(self):
        return 'Bridges: {} Else: {}'.format(self.bridges, self.else_bridge)


class Game(object):
    '''Game based on the Zoombini's Allergic Cliffs level'''

    def __init__(self, zoombinis=None, bridge=None):
        if zoombinis:
            self.zoombinis = zoombinis
        else:
            # init zoombinis randomly
            # self.zoombinis = [Zoombini() for _ in range(NUM_ZOOMBINIS)]
            self.zoombinis = sort_zoombinis([Zoombini() for _ in range(NUM_ZOOMBINIS)])

        # bridge = Bridge()
        # bridge.attr = 'hair'
        # bridge.attr_num = 1
        # bridge.bridge = 0

        self.new_game(bridge)
        self.truth = list(map(lambda z: self.bridge.zoombini_bridge(z), self.zoombinis))
        self.mistakes = 0
        self.known = {}
        # print(self.truth)

    def new_game(self, bridge=None):
        if bridge:
            self.bridge = bridge
        else:
            self.bridge = Bridge()

    def send_zoombini(self, index, top):
        # print('sending', index, top)
        top = 1 if top else 0
        zoombini = self.zoombinis[index]
        if self.bridge.check_pass(zoombini, top) and not zoombini.has_passed:
            zoombini.has_passed = True
            zoombini.accepted_bridge = top
        # can't repeat sending same zoombini
        elif zoombini.has_passed:
            # self.mistakes += 1
            return False
        else:
            self.mistakes += 1
            zoombini.rejected_bridges.append(top)
        did_pass = 'passed' if zoombini.has_passed else 'failed'
        self.known[(index, top)] = zoombini.has_passed
        # print('sending {} to {} and it {}'.format(index, top, did_pass))

        return zoombini.has_passed

    def has_won(self):
        num_passed = sum(map(lambda x: x.has_passed, self.zoombinis))
        return num_passed == len(self.zoombinis)

    def score(self):
        return sum(map(lambda x: x.has_passed, self.zoombinis))

    def can_move(self):
        return self.mistakes < MAX_MISTAKES and not self.has_won()

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
        vecs = [0 for _ in range(len(self.truth)*NUM_BRIDGES)]
        for i, t in enumerate(self.truth):
            vecs[NUM_BRIDGES*i+t] = 1
        # return self.truth
        return vecs

    def reset(self):
        self.mistakes = 0
        for zoombini in self.zoombinis:
            zoombini.reset()

    def get_invalid_moves(self):
        inds = []
        for i in range(len(self.zoombinis)):
            if self.zoombinis[i].has_passed:
                for j in range(NUM_BRIDGES):
                    inds.append(NUM_BRIDGES*i+j)
        return inds

    def zoombinis_json(self):
        zoombinis = [z.json() for z in self.zoombinis]
        for i in range(len(zoombinis)):
            zoombinis[i]['id'] = i
        return zoombinis

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
        self.winning_threshold = NUM_ZOOMBINIS-0.3

    def reset(self, game=None):
        # print('Start game')
        self.game = game if game else Game()
        self.actions = set()
        return np.array(self.game.get_agent_state())#.reshape(1, -1)

    def step(self, action, verbose=False, reward_shaper=None):
        # action is an int
        # action/num_bridges is the zoombini, action % num_bridges is the bridge
        self.actions.add(action)
        zoombini = action//NUM_BRIDGES
        bridge = action % NUM_BRIDGES

        already_passed = self.game.zoombinis[zoombini].has_passed

        passed = self.game.send_zoombini(zoombini, bridge)

        state = np.array(self.game.get_agent_state())#.reshape(1, -1)
        reward = 1 if passed else 0
        done = not self.game.can_move()

        if verbose:
            passed_str = 'PASSED' if passed else 'failed'
            print('Sending Zoombini {} to {} and it {}'.format(zoombini, bridge, passed_str))
        if reward_shaper:
            if already_passed:
                reward2 = 0
            else:
                reward2 = reward_shaper.get_reward(self.game.get_brain_state(), action)
            return state, reward, done, reward2
        else:
            return state, reward, done

    def check_valid_move(self, action):
        zoombini = action//NUM_BRIDGES
        return not self.game.zoombinis[zoombini].has_passed

    def get_invalid_moves(self):
        return self.game.get_invalid_moves()

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
