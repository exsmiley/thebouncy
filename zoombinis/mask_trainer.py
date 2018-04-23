import random
import tqdm
import numpy as np
from zoombinis import *
from baselines import *
from brain import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class MaskBrainTrainer(object):

    def __init__(self, chkpt=None):
        self.model = Brain(chkpt=chkpt)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.BCELoss()
        self.chkpt_name = 'models/brain_mask'

        self.state_buffer = []
        self.feedback_buffer = []
        self.mask_buffer = []

    def run(self):
        self.model.train()
        for i in range(NUM_RUNS):
            states, feedbacks, masks = self.play_game()
            self.state_buffer.extend(states)
            self.feedback_buffer.extend(feedbacks)
            self.mask_buffer.extend(masks)

            if (i+1) % 10 == 0:
                print('Episode {}...'.format(i+1))
                self.train()

        self.model.save(self.chkpt_name)

    def train(self):
        all_indices = [i for i in range(len(self.state_buffer))]
        indices = set(random.sample(all_indices, SAMPLE_SIZE))

        states = [state for i, state in enumerate(self.state_buffer) if i in indices]
        feedbacks = [fb for i, fb in enumerate(self.feedback_buffer) if i in indices]
        masks = [mask for i, mask in enumerate(self.mask_buffer) if i in indices]

        states = torch.FloatTensor(states).view(-1, BRAIN_INPUT_LENGTH)
        masks = torch.FloatTensor(masks).view(-1, OUTPUT_LENGTH)
        feedbacks = torch.FloatTensor(feedbacks).view(-1, OUTPUT_LENGTH)#int(OUTPUT_LENGTH/NUM_BRIDGES), -1)
        
        states, feedbacks, masks = Variable(states), Variable(feedbacks), Variable(masks)

        # actually train now
        self.optimizer.zero_grad()
        output = self.model.forward(states)*masks

        # print(output.data.numpy()[0])
        # print(masks.data.numpy()[0])
        # print((output*masks).data.numpy()[0])
        # quit()

        loss = self.criterion(output, feedbacks)
        loss.backward()
        self.optimizer.step()

        print('Loss:', sum(loss.data.numpy()))

        # reset buffers
        self.state_buffer = []
        self.feedback_buffer = []
        self.mask_buffer = []


    def play_game(self):
        game = Game()
        sent_indices = set()
        states = []

        while game.can_move():
            index = random.randint(0, NUM_ZOOMBINIS-1)

            while index in sent_indices:
                index = random.randint(0, NUM_ZOOMBINIS-1)

            game.send_zoombini(index, random.randint(0, 1))
            state = game.get_brain_state()

            states.append(state)

        feedback, mask = game.get_mask_truth()
        feedbacks = [feedback for i in range(len(states))]
        masks = [mask for i in range(len(states))]

        return states, feedbacks, masks

class MaskEntropyTrainer(MaskBrainTrainer):

    def __init__(self):
        super(MaskEntropyTrainer, self).__init__()
        self.chkpt_name = 'models/brain_entropy_mask'
        self.player = MaxEntropyPlayer()
        self.brain = Brain()
        self.player.brain = self.brain

    def play_game(self):
        states, game = self.player.play_game_trainer_mask()
        feedback, mask = game.get_mask_truth()
        feedbacks = [feedback for i in range(len(states))]
        masks = [mask for i in range(len(states))]
        return states, feedbacks, masks

if __name__ == '__main__':
    MaskBrainTrainer().run()