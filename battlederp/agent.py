import argparse
import pickle
import numpy as np
import random
import traceback
from itertools import count
from battleship import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
np.set_printoptions(suppress=True)

import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from utils import *


GAMMA = 0.99
# SEED = 543
LOG_INTERVAL = 100


class Derp(nn.Module):
    def __init__(self, state_xform, action_xform):
        super(Derp, self).__init__()
        # input_length = 4
        state_length, action_length = state_xform.length, action_xform.length
        self.state_xform, self.action_xform = state_xform, action_xform
        input_length = state_length
        layer_size = 128
        # layer_size = 1000
        self.affine1 = nn.Linear(input_length, layer_size)
        self.action_head = nn.Linear(layer_size, action_length)
        self.value_head = nn.Linear(layer_size, 1)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=-1), state_values

    def load(self, name='derp'):
        self.load_state_dict(torch.load(name))
        print('Loaded model from {}!'.format(name))

    def save(self, name='derp'):
        torch.save(self.state_dict(), name)
        print('Saved model to {}!'.format(name))


class DerpPlayer(object):

    def __init__(self, state_xform, action_xform):
        self.model = Derp(state_xform, action_xform)
        self.state_xform = state_xform
        self.action_xform = action_xform
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def act(self, x, disallowed=None):
        state = self.state_xform.state_to_np(x)
        state = Variable(torch.from_numpy(state).float())
        action_pr, sv = self.model.forward(state)
        action_pr = action_pr.data.numpy()
        action_pr2 = action_pr.copy()

        if disallowed:
            action_pr2 += 1e-10
            for i in disallowed:
                action_pr2[i] = 0
            action_pr2 /= sum(action_pr2)

        action_id = np.random.choice(range(len(action_pr)), p=action_pr2)
        return self.action_xform.idx_to_action(action_id), action_pr, sv.data[0]

    def value_estimator(self, x):
        state = self.state_xform.state_to_np(x)
        action_pr, sv = self.model.forward(state)
        return sv

    def learn(self, tups):
        rewards = []
        action_pr = []
        policy_losses = []
        value_losses = []
        for tr in tups:
            rewards.append(tr.r)
            action_pr.append((Variable(torch.from_numpy(np.log([tr.p])), requires_grad=True), Variable(torch.Tensor(np.array([tr.v])), requires_grad=True)))

        rewards = torch.Tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
        for (log_prob, value), r in zip(action_pr, rewards):
            advantage = r - value
            policy_losses.append(-log_prob * advantage)
            value_losses.append(F.smooth_l1_loss(value, Variable(torch.Tensor([r]))))
        self.optimizer.zero_grad()
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        loss.backward()
        self.optimizer.step()



if __name__ == '__main__':
    state_xform, action_xform = StateXform(), ActionXform()
    derp_actor = DerpPlayer(state_xform, action_xform)
    buff = Buffer(10000)
    game_bound = L*L
    avg_num_moves = []
    avg_moves = game_bound

    running_reward = 0

    for i in range(10000):
        env = GameEnv()
        trace = play_game(env, derp_actor, game_bound)
        # print ([tr.a for tr in trace])
        total_reward = sum([tr.r for tr in trace])
        avg_moves = avg_moves*0.99 + len(trace)*0.01
        avg_num_moves.append(running_reward)
        running_reward = running_reward * 0.99 + total_reward * 0.01
        disc_trace = get_discount_trace(trace, derp_actor.value_estimator)
        [buff.add(tr) for tr in disc_trace]

        tr_sample = [buff.sample() for _ in range(20)]
        derp_actor.learn(tr_sample)

        if (i+1) % LOG_INTERVAL == 0:
            print('Game {}: reward {} num_moves {}'.format(i+1, running_reward, len(trace)))
            print ([(tr.a, tr.p) for tr in trace])
    derp_actor.model.save()
    import matplotlib.pyplot as plt
    plt.plot(avg_num_moves)
    plt.show()

