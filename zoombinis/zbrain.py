import os
import numpy as np
import random
import time
import tqdm
import pickle
from copy import copy
from zoombinis import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# tunable hyperparameters
NUM_RUNS = 10000
LEARNING_RATE = 1e-3
MOMENTUM = 0.99
to_train = True

# supress warnings...
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'


class ZBrain(nn.Module):

    def __init__(self, chkpt=None):
        super(ZBrain, self).__init__()

        self.fc1 = nn.Linear(INPUT_LENGTH, 2500)
        self.fc2 = nn.Linear(2500, FEEDBACK_LENGTH)

        if chkpt is not None:
            self.load(chkpt)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=-1)
        return x

    def get_entropy(self, state):
        state = Variable(torch.FloatTensor(state))

        probs = [
            p*np.log2(p)
            for p in self.forward(state).data
            if p > 1e-8
        ]
        return -sum(probs)

    def get_probabilities(self, state):
        state = Variable(torch.FloatTensor(state))
        return list(self.forward(state).data)

    def load(self, name='models/brain'):
        self.load_state_dict(torch.load(name))

    def save(self, name='models/brain'):
        torch.save(self.state_dict(), name)
        print('Saved model to {}!'.format(name))


class ZBrainTrainer(object):

    def __init__(self, chkpt=None):
        self.model = ZBrain()
        self.optimizer = optim.SGD(self.model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
        self.criterion = nn.CrossEntropyLoss()

        self.state_buffer = []
        self.feedback_buffer = []

    def run(self):
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
        indices = set(random.sample(all_indices, 20))

        states = [state for i, state in enumerate(self.state_buffer) if i in indices]
        feedbacks = [fb for i, fb in enumerate(self.feedback_buffer) if i in indices]

        states = torch.FloatTensor(states).view(-1, INPUT_LENGTH)
        # print(states.shape[0])
        feedbacks = torch.LongTensor(feedbacks)
        
        states, feedbacks = Variable(states), Variable(feedbacks)

        # actually train now
        self.optimizer.zero_grad()
        output = self.model(states)
        loss = self.criterion(output, feedbacks)
        loss.backward()
        self.optimizer.step()

        print('Loss:', loss.data[0])

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

            state = game.get_state_vector(index)

            send_top = True if random.randint(0, 1) else False
            feedback = game.send_zoombini(index, send_top)

            if (feedback and send_top) or (not feedback and not send_top):
                feedback_vec = 0
            else:
                feedback_vec = 1

            if feedback:
                sent_indices.add(index)

            states.append(state)
            feedbacks.append(feedback_vec)

        return states, feedbacks

    def test(self):
        game = Game()
        print(game)
        sent_indices = set()

        while game.can_move():
            index = random.randint(0, NUM_ZOOMBINIS-1)

            while index in sent_indices:
                index = random.randint(0, NUM_ZOOMBINIS-1)

            state = game.get_state_vector(index)
            state = np.array(state).reshape(-1, INPUT_LENGTH)

            probabilities = self.model.get_feedback_layer(state)
            entropy = self.model.get_entropy(state)
            print('\nGoing to send index', index, game.zoombinis[index])
            print('Probabilities: {}\nEntropy: {}'.format(probabilities, entropy))

            send_top = True if random.randint(0, 1) else False
            feedback = game.send_zoombini(index, send_top)
            print(feedback, send_top)
            if (feedback and send_top) or (not feedback and not send_top):
                feedback_vec = [1, 0]
            else:
                feedback_vec = [0, 1]

            if feedback:
                sent_indices.add(index)



if __name__ == '__main__':
    if to_train:
        trainer = ZBrainTrainer()
        trainer.run()
        # trainer.test()
    else:
        trainer = ZBrainTrainer(chkpt='models/brain')
        trainer.test()
