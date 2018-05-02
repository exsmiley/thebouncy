import random
import tqdm
import numpy as np
from mastermind import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# make parent directory available
import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from utils import *


# tunable hyperparameters
NUM_RUNS = 100000
SAMPLE_SIZE = 100
LEARNING_RATE = 1e-3
MOMENTUM = 0.9
TO_TRAIN = True
USE_OLD = False
PLAY = True
REWARD_SCALE_FACTOR = 100 # amount to divide entropy by

OLD_CHKPT = None
if USE_OLD or not TO_TRAIN:
    OLD_CHKPT = 'models/brain'


class Brain(nn.Module):

    def __init__(self, chkpt=None):
        super(Brain, self).__init__()

        self.fc1 = nn.Linear(BRAIN_INPUT_LENGTH, 3000)
        self.fc2 = nn.Linear(3000, OUTPUT_LENGTH)

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
        for i in range(0, len(vecs), NUM_OPTIONS):
            total = 0
            for j in range(i, i+NUM_OPTIONS):
                total += vecs[j]
            for j in range(i, i+NUM_OPTIONS):
                vecs[j] /= total

        return list(vecs)

    def get_action_probabilities(self, state):
        pred = self.get_probabilities(state)
        split_pred = list(map(list, np.split(np.array(pred), 4)))

        all_preds = []

        for guess in ALL_GUESSES:
            prob = 1
            for i, peg in enumerate(guess):
                prob *= split_pred[i][peg]
            all_preds.append(prob)

        return all_preds


    def load(self, name='models/brain'):
        self.load_state_dict(torch.load(name))
        print('Loaded model from {}!'.format(name))

    def save(self, name='models/brain'):
        torch.save(self.state_dict(), name)
        print('Saved model to {}!'.format(name))


class BrainTrainer(object):

    def __init__(self, chkpt=None):
        self.model = Brain(chkpt=chkpt)
        self.state_xform = StateXform()
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
            # self.state_buffer.append(states[-1])
            # self.feedback_buffer.append(feedbacks[-1])

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
        env = MastermindEnv()
        sent_indices = set()
        states = []
        feedbacks = []
        count = 0

        while env.can_move() or len(states) < 5:
            index = random.randint(0, NUM_ALL_GUESSES-1)

            while index in sent_indices:
                index = random.randint(0, NUM_ALL_GUESSES-1)

            action = get_guess(index)

            time_state, _, _ = env.step(action)
            state = self.state_xform.state_to_np(time_state)

            states.append(state)
            feedbacks.append(env.game.get_brain_truth())

        return states, feedbacks

    def test(self):
        env = MastermindEnv()
        sent_indices = set()
        states = []
        feedbacks = []

        print('TRUTH')
        print(env.game.target)
        print(env.game.get_brain_truth())
        old_state = []

        while env.can_move():
            index = random.randint(0, NUM_ALL_GUESSES-1)

            while index in sent_indices:
                index = random.randint(0, NUM_ALL_GUESSES-1)

            action = get_guess(index)
            time_state, _, _ = env.step(action)
            state = self.state_xform.state_to_np(time_state)
            
            pred = self.model.get_probabilities(state)

            print(list(map(list, np.split(np.array(pred), 4))))



class BrainAgent(object):

    def __init__(self, state_xform, action_xfrom, num_random=0):
        self.brain = Brain(chkpt='models/brain')
        self.state_xform = state_xform
        self.action_xfrom = action_xfrom
        self.num_random = num_random
        self.made_moves = []

    def reset(self):
        self.made_moves = []

    def act(self, x, forbid, det=False):
        if len(x) < self.num_random:
            move = random_guess()
            return tuple(move), {tuple(move): 1.0}
        state = self.state_xform.state_to_np(x)


        # pred = self.brain.get_probabilities(state)
        # split_pred = list(map(list, np.split(np.array(pred), 4)))

        # guess = []
        # prob = 1

        # for p in split_pred:
        #     val = np.argmax(p)
        #     prob *= p[val]
        #     guess.append(val)

        # action = tuple(guess)

        pred = self.brain.get_action_probabilities(state)

        for move in self.made_moves:
            pred[move] = 0

        pred = np.array(pred)

        pred = pred / sum(pred)

        if det:
            action = np.argmax(pred)
        else:
            action = np.random.choice(range(NUM_ALL_GUESSES), p=pred)
        self.made_moves.append(action)

        return action, pred



if __name__ == '__main__':
    if PLAY:
        state_xform, action_xform = StateXform(), ActionXform()
        ac_actor = BrainAgent(state_xform, action_xform)
        game_bound = 11
        score = 0
        num_games = 100

        for i in tqdm.tqdm(range(num_games)):
            env = MastermindEnv()
            ac_actor.reset()
            trace = play_game(env, ac_actor, game_bound)
            print(trace)
            score += sum([tr.r for tr in trace])

        print('avg score:', score/num_games)

        quit()
    if TO_TRAIN:
        trainer = BrainTrainer(chkpt=OLD_CHKPT)
        trainer.run()
        trainer.test()
    else:
        trainer = BrainTrainer(chkpt='models/brain')
        trainer.test()

