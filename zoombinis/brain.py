import random
import tqdm
import numpy as np
from zoombinis import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


# tunable hyperparameters
NUM_RUNS = 50000
SAMPLE_SIZE = 20
LEARNING_RATE = 1e-3
MOMENTUM = 0.9
TO_TRAIN = True
USE_OLD = False
REWARD_SCALE_FACTOR = 100 # amount to divide entropy by

OLD_CHKPT = None
if USE_OLD or not TO_TRAIN:
    OLD_CHKPT = 'models/brain'


class Brain(nn.Module):

    def __init__(self, chkpt=None, ):
        super(Brain, self).__init__()

        self.fc1 = nn.Linear(BRAIN_INPUT_LENGTH, 1000)
        self.fc2 = nn.Linear(1000, OUTPUT_LENGTH)

        if chkpt is not None:
            self.load(chkpt)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.sigmoid(x)
        # vecs = torch.split(self.fc2(x), NUM_BRIDGES, dim=1)
        return vecs

    def get_entropies(self, state):
        entropies = []
        probs = self.get_probabilities(state)
        
        for i in range(len(probs)):
            val = -probs[i]*np.log10(probs[i]) if probs[i] > 1e-8 else 0
            entropies.append(val)

        return entropies

    def get_probabilities(self, state):
        state = Variable(torch.FloatTensor(state).view(-1, BRAIN_INPUT_LENGTH))
        vecs = self.forward(state).data.numpy()[0]
        
        for i in range(0, len(vecs), NUM_BRIDGES):
            total = 0
            for j in range(i, i+NUM_BRIDGES):
                total += vecs[j]
            for j in range(i, i+NUM_BRIDGES):
                vecs[j] /= total

        return list(vecs)

    def predict(self, state):
        return self.get_probabilities(state)

    def get_probabilities_total(self, state, known):
        state2 = Variable(torch.FloatTensor(state).view(-1, BRAIN_INPUT_LENGTH))
        vecs = self.forward(state2).data.numpy()[0]
        new_vecs = np.zeros(NUM_ZOOMBINIS*(NUM_BRIDGES+1))

        # make certain ones that we know
        for k, v in known.items():
            zoombini, bridge = k
            action = NUM_BRIDGES*zoombini+bridge
            if v:
                vecs[action] = 1
            else:
                vecs[action] = 0

        for i in range(0, len(vecs), NUM_BRIDGES):
            total = 0
            for j in range(i, i+NUM_BRIDGES):
                total += vecs[j]
            for j in range(i, i+NUM_BRIDGES):
                vecs[j] /= total

        vec_count = 0
        z_count = 0
        for i in range(len(new_vecs)):
            if (i+1) % (NUM_BRIDGES+1) == 0:
                if z_count in known:
                    new_vecs[i] = 1
                z_count += 1
            else:
                new_vecs[i] = vecs[vec_count]
                vec_count += 1

        vecs = np.array(new_vecs)

        return list(vecs)

    def load(self, name='models/brain'):
        self.load_state_dict(torch.load(name))
        print('Loaded model from {}!'.format(name))

    def save(self, name='models/brain'):
        torch.save(self.state_dict(), name)
        print('Saved model to {}!'.format(name))

    # def train(self, traces):
    #     pass


class BrainTrainer(object):

    def __init__(self, chkpt=None):
        self.model = Brain(chkpt=chkpt)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.BCELoss()
        self.chkpt_name = 'models/brain'

        self.state_buffer = []
        self.feedback_buffer = []

    def run(self):
        self.model.train()
        for i in range(NUM_RUNS):
            states, feedbacks = self.play_game()
            self.state_buffer.extend(states)
            self.feedback_buffer.extend(feedbacks)

            if (i+1) % 10 == 0:
                print('Episode {}...'.format(i+1))
                self.train()

        self.model.save(self.chkpt_name)

    def train(self):
        all_indices = [i for i in range(len(self.state_buffer))]
        indices = set(random.sample(all_indices, SAMPLE_SIZE))

        states = [state for i, state in enumerate(self.state_buffer) if i in indices]
        feedbacks = [fb for i, fb in enumerate(self.feedback_buffer) if i in indices]

        states = torch.FloatTensor(states).view(-1, BRAIN_INPUT_LENGTH)
        feedbacks = torch.FloatTensor(feedbacks).view(-1, OUTPUT_LENGTH)#int(OUTPUT_LENGTH/NUM_BRIDGES), -1)
        
        states, feedbacks = Variable(states), Variable(feedbacks)

        # actually train now
        self.optimizer.zero_grad()
        output = self.model.forward(states)

        loss = self.criterion(output, feedbacks)
        loss.backward()
        self.optimizer.step()

        print('Loss:', loss.data.numpy())

        # reset buffers
        self.state_buffer = []
        self.feedback_buffer = []


    def play_game(self):
        game = Game()
        sent_indices = set()
        states = []
        feedbacks = []

        while game.can_move():
            index = random.randint(0, NUM_ZOOMBINIS-1)

            while index in sent_indices:
                index = random.randint(0, NUM_ZOOMBINIS-1)

            game.send_zoombini(index, random.randint(0, 1))
            state = game.get_brain_state()

            states.append(state)
            feedbacks.append(game.get_brain_truth())

        return states, feedbacks

    def test(self):
        game = Game()
        print(game)
        sent_indices = set()
        states = []
        feedbacks = []

        print('TRUTH')
        print(game.get_brain_truth())
        old_state = []

        while game.can_move():
            index = random.randint(0, NUM_ZOOMBINIS-1)

            while index in sent_indices:
                index = random.randint(0, NUM_ZOOMBINIS-1)

            state = game.get_brain_state()
            pred = self.model.get_probabilities(state)
            # print('\nSTATE:', state)
            # print('\nPRED:', pred)
            # print('ENTROPIES:', self.model.get_entropies(state))

            result = game.send_zoombini(index, random.randint(0, 1))
            # print(game)
            # print(result, index)
            # print(game.get_brain_state())

            # if old_state == state:
            #     raise Exception('deerpp')
            # old_state = state


class EntropyRewardOracle(object):

    def __init__(self):
        self.brain = Brain(chkpt='models/brain')

    def get_reward(self, state, action):
        entropies = self.brain.get_entropies(state)
        return entropies[action]/REWARD_SCALE_FACTOR


if __name__ == '__main__':
    if TO_TRAIN:
        trainer = BrainTrainer(chkpt=OLD_CHKPT)
        trainer.run()
        trainer.test()
    else:
        trainer = BrainTrainer(chkpt='models/brain')
        trainer.test()