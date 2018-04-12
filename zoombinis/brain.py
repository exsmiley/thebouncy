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
NUM_RUNS = 10000
SAMPLE_SIZE = 20
LEARNING_RATE = 3e-2
MOMENTUM = 0.9
TO_TRAIN = False
USE_OLD = False
REWARD_SCALE_FACTOR = 10 # amount to divide entropy by

OLD_CHKPT = None
if USE_OLD:
    OLD_CHKPT = 'models/brain'


class Brain(nn.Module):

    def __init__(self, chkpt=None):
        super(Brain, self).__init__()

        self.fc1 = nn.Linear(BRAIN_INPUT_LENGTH, 5000)
        self.fc2 = nn.Linear(5000, OUTPUT_LENGTH)

        if chkpt is not None:
            self.load(chkpt)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        vecs = torch.split(self.fc2(x), NUM_BRIDGES, dim=1)
        return vecs

    def get_entropies(self, state):
        probs = self.get_probabilities(state)

        entropies = []
        for prob in probs:
            entropies.extend([
                -p*np.log2(p)
                if p > 1e-8
                else 0
                for p in prob
            ])
        return entropies

    def get_probabilities(self, state):
        state = Variable(torch.FloatTensor(state).view(-1, BRAIN_INPUT_LENGTH))
        vecs = self.forward(state)
        # need softmaxes
        vecs = tuple(map(lambda v: F.softmax(v+1e-8, dim=-1), vecs))
        return list(map(lambda x: list(x.data.numpy()[0]), vecs))

    def load(self, name='models/brain'):
        self.load_state_dict(torch.load(name))
        print('Loaded model from {}!'.format(name))

    def save(self, name='models/brain'):
        torch.save(self.state_dict(), name)
        print('Saved model to {}!'.format(name))


class BrainTrainer(object):

    def __init__(self, chkpt=None):
        self.model = Brain(chkpt=chkpt)
        self.optimizer = optim.SGD(self.model.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.CrossEntropyLoss()

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

        self.model.save()

    def train(self):
        all_indices = [i for i in range(len(self.state_buffer))]
        indices = set(random.sample(all_indices, SAMPLE_SIZE))

        states = [state for i, state in enumerate(self.state_buffer) if i in indices]
        feedbacks = [fb for i, fb in enumerate(self.feedback_buffer) if i in indices]

        states = torch.FloatTensor(states).view(-1, BRAIN_INPUT_LENGTH)
        feedbacks = torch.LongTensor(feedbacks).view(int(OUTPUT_LENGTH/NUM_BRIDGES), -1)
        
        states, feedbacks = Variable(states), Variable(feedbacks)

        # actually train now
        self.optimizer.zero_grad()
        output = self.model(states)
        losses = []
        for i, zoombini in enumerate(output):
            loss = self.criterion(zoombini, feedbacks[i])
            losses.append(loss)

        total_loss = sum(losses)#/len(losses)
        total_loss.backward()
        self.optimizer.step()

        print('Loss:', sum(total_loss.data.numpy()))#sum(losses)/len(losses))

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

            state = game.get_brain_state()
            game.send_zoombini(index, random.randint(0, 1))

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

        while game.can_move():
            index = random.randint(0, NUM_ZOOMBINIS-1)

            while index in sent_indices:
                index = random.randint(0, NUM_ZOOMBINIS-1)

            state = game.get_brain_state()
            pred = self.model.get_probabilities(state)
            print('\nSTATE:', state)
            print('\nPRED:', pred)
            print('ENTROPIES:', self.model.get_entropies(state))
            game.send_zoombini(index, random.randint(0, 1))


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