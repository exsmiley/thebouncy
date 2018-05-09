import random
import numpy as np
import itertools
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# make parent directory available
import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from utils import *


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

    def __init__(self, zoombinis):
        # generate the conditions
        self.else_bridge = random.randint(0, NUM_BRIDGES-1)
        conds = set()
        self.bridges = {}
        for i in range(NUM_BRIDGES):
            if i != self.else_bridge:
                cond = self.gen_condition(zoombinis)
                while str(cond) in conds:
                    cond = self.gen_condition(zoombinis)
                self.bridges[i] = cond
                conds.add(str(cond))

    def gen_condition(self, zoombinis):
        # attr = random.choice(['hair', 'eyes', 'nose', 'feet'])
        # num = random.randint(0, 5)
        # return {'attr': attr, 'num': num}
        attrs = ['hair', 'eyes', 'nose', 'feet']
        probs = []
        vals = []
        for attr in attrs:
            counts = [0 for i in range(5)]
            for z in zoombinis:
                val = getattr(z, attr)
                counts[val] += 1
            counts = np.array(counts) / len(zoombinis)
            selected_val = np.random.choice(range(5), p=counts)
            probs.append(counts[selected_val])
            vals.append(selected_val)
        
        probs = np.array(probs) / np.sum(probs)

        attr_num = np.random.choice(len(attrs), p=probs)
        return {'attr': attrs[attr_num], 'num': vals[attr_num]}


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
        self.max_steps = 24
        self.steps = 0
        self.known = {}
        # print(self.truth)

    def new_game(self, bridge=None):
        if bridge:
            self.bridge = bridge
        else:
            self.bridge = Bridge(self.zoombinis)

    def send_zoombini(self, index, top):
        self.steps += 1
        top = 1 if top else 0
        zoombini = self.zoombinis[index]
        if not zoombini.has_passed and self.bridge.check_pass(zoombini, top):
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
        # return self.steps < self.max_steps and not self.has_won()
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

    def get_mask_truth(self):
        vecs = [0 for _ in range(len(self.truth)*NUM_BRIDGES)]
        mask = [0 for _ in range(len(self.truth)*NUM_BRIDGES)]

        for (index, top), _ in self.known.items():
            truth = self.truth[index]
            vecs[NUM_BRIDGES*index+truth] = 1
            for i in range(NUM_BRIDGES):
                mask[NUM_BRIDGES*index+i] = 1

        return vecs, mask

    def reset(self):
        self.mistakes = 0
        self.steps = 0
        self.known = {}
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
        self.truth = self.game.get_brain_truth()
        return np.array(self.game.get_brain_state()), self.truth, self.game.get_mask_truth()

    def win(self):
        return self.game.has_won()

    def step(self, action):#, verbose=False, reward_shaper=None):
        # action is an int
        # action/num_bridges is the zoombini, action % num_bridges is the bridge
        self.actions.add(action)
        zoombini = action//NUM_BRIDGES
        bridge = action % NUM_BRIDGES

        already_passed = self.game.zoombinis[zoombini].has_passed

        passed = self.game.send_zoombini(zoombini, bridge)

        state = np.array(self.game.get_brain_state())#.reshape(1, -1)
        reward = 1 if passed else 0
        done = not self.game.can_move()

        state = state, self.truth, self.game.get_mask_truth()

        return state, reward, done

    def forbid(self):
        return self.game.get_invalid_moves()

class StateXform:
  def __init__(self):
    self.length = BRAIN_INPUT_LENGTH

  def state_to_np(self, state):
    board_mask, _, _ = state
    # ret =  np.concatenate((board_mask, mask_truth))
    return np.array(board_mask)

class StateXformTruth:
  def __init__(self):
    self.length = BRAIN_INPUT_LENGTH + OUTPUT_LENGTH

  def state_to_np(self, state):
    board_mask, board_truth, _ = state
    ret =  np.concatenate((board_mask, board_truth))
    return ret

class OracleXform:
    def __init__(self, oracle):
        self.length = BRAIN_INPUT_LENGTH + OUTPUT_LENGTH
        self.oracle = oracle

    def state_to_np(self, state):
        board_mask, _, _ = state
        oracle_prediction = self.oracle.predict(state)
        ret =  np.concatenate((board_mask, oracle_prediction))

        return ret

class FutureXform:
    def __init__(self):
        self.length = OUTPUT_LENGTH

    def state_to_np(self, state):
        _, _, mask_and_truth = state

        return mask_and_truth

class ActionXform:
  def __init__(self):
    self.possible_actions = list(range(OUTPUT_LENGTH))
    self.length = OUTPUT_LENGTH
  def idx_to_action(self, idx):
    return self.possible_actions[idx]
  def action_to_idx(self, a):
    return a
  def action_to_1hot(self, a):
    ret = np.zeros(self.length)
    ret[a] = 1.0
    return ret


class Oracle(nn.Module):

    def __init__(self, state_xform, future_xform, n_hidden):
        super(Oracle, self).__init__()

        state_length, future_length = state_xform.length, future_xform.length
        self.state_xform, self.future_xform = state_xform, future_xform

        self.fc1 = nn.Linear(BRAIN_INPUT_LENGTH, n_hidden)
        self.fc2 = nn.Linear(n_hidden, OUTPUT_LENGTH)

        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    def predict(self, state):
        state, _, _ = state
        state = Variable(torch.FloatTensor(state).to(device).view(-1, BRAIN_INPUT_LENGTH))
        vecs = self.forward(state).cpu().data.numpy()[0]
        
        for i in range(0, len(vecs), NUM_BRIDGES):
            total = 0
            max_j = 0
            max_j_val = 0
            for j in range(i, i+NUM_BRIDGES):
                # total += vecs[j]
                if vecs[j] > max_j_val:
                    max_j = j
                    max_j_val = vecs[j]
                vecs[j] = 0
            vecs[max_j] = 1
            # for j in range(i, i+NUM_BRIDGES):
                # vecs[j] /= total

        return list(vecs)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.sigmoid(x)

    def train(self, oracle_data):
        s_batch = to_torch(np.array([self.state_xform.state_to_np(tr.s_i)\
                for tr in oracle_data]))
        mask_and_truth = [self.future_xform.state_to_np(tr.s_t)\
                for tr in oracle_data]
        f_batch = to_torch(np.array([np.array(m)*np.array(t) for (m,t) in mask_and_truth]))

        future_prediction = self(s_batch)

        # Optimize the model
        self.optimizer.zero_grad()
        loss = self.criterion(future_prediction, f_batch)
        loss.backward()
        self.optimizer.step()

def measure_oracle(oracle, oracle_datas):
    total_score = 0
    obtained_score = 0
    for od in oracle_datas:
        current_ob, target_ob = od.s_i, od.s_t
        prediction = np.array(oracle.predict(current_ob))
        target, mask = oracle.future_xform.state_to_np(target_ob)
        omitted_score = (len(mask)-sum(mask))//2
        prediction = prediction*np.array(mask)
        target = np.array(target)*np.array(mask)
        prediction = np.reshape(prediction, (OUTPUT_LENGTH//2, 2))
        target = np.reshape(target, (OUTPUT_LENGTH//2, 2))
        argmax_pred = np.argmax(prediction, axis=1)
        argmax_target = np.argmax(target, axis=1)

        for (pred, target) in zip(argmax_pred, argmax_target):
            total_score += 1
            if pred == target:
                obtained_score += 1

        # don't give it a benefit for masked scores
        total_score -= omitted_score
        obtained_score -= omitted_score

    return obtained_score / total_score



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
