import math
import random
import numpy as np
from collections import namedtuple
from itertools import count

import cv2
import torch
import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from battleship import L

# make parent directory available
import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from utils import to_torch, to_torch_int, to_torch_byte, device

# simple cross entropy cost (might be numerically unstable if pred has 0)
def xentropy_cost(x_target, x_pred):
    assert x_target.size() == x_pred.size(), \
            "size fail ! "+str(x_target.size()) + " " + str(x_pred.size())
    logged_x_pred = torch.log(x_pred)
    cost_value = -torch.sum(x_target * logged_x_pred)
    return cost_value

def measure_oracle(oracle, oracle_datas):
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    total_err = 0
    for od in oracle_datas:
        ob_i, truth_i = od

        for ii in range(L):
            for jj in range(L):
                if ob_i[ii][jj] != 2:
                    assert ob_i[ii][jj] == truth_i[ii][jj], "inconsistent ob"


        prediction = oracle.predict(od)

        # print ("input ")
        # print (ob_i)

        # print ("prediction ")
        pred = np.reshape(prediction[:,1], (L,L))
        # print (pred)

        # print ("target ")
        # print (truth_i)

        error = np.abs(pred - truth_i)
        # print ("error " )
        error = np.sum(error)

        total_err += error
    return total_err / len(oracle_datas) / (L * L)
                
def consistency_check(oracle_data):
    for tr in oracle_data:
        ins = tr.s_i
        out = tr.s_t
        ob_i, truth_i = ins
        ob_t, truth_t = out
        assert (np.sum(truth_i == truth_t)) == L*L, "truth is wrong"
        for ii in range(L):
            for jj in range(L):
                if ob_i[ii][jj] != 2:
                    assert ob_i[ii][jj] == truth_i[ii][jj], "inconsistent ob"

class Oracle(nn.Module):

    def __init__(self, state_xform, future_xform, action_xform, n_hidden):
        super(Oracle, self).__init__()

        state_length, future_length = state_xform.length, L*L*2
        self.state_xform, self.future_xform = state_xform, future_xform
        self.action_xform = action_xform

        self.enc1  = nn.Linear(state_length, n_hidden)
        self.enc2  = nn.Linear(n_hidden, n_hidden)
        self.head = nn.Linear(n_hidden, future_length)

        self.all_opt = torch.optim.RMSprop(self.parameters(), lr=0.0001)

    def predict(self, state):
        state = to_torch(np.array([self.state_xform.state_to_np(state)]))
        prediction = self(state).data.cpu().numpy()[0]
        return prediction

    def compress(self, state):
        x = to_torch(np.array([self.state_xform.state_to_np(state)]))
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        compressed = x.data.cpu().numpy()[0]
        return compressed

    def forward(self, x):
        batch_size = x.size()[0]
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = self.head(x)
        x = x.view(-1, self.future_xform.n_obs, 2)
        x = F.softmax(x, dim=2)
        smol_const = to_torch(np.array([1e-6]))
        x = x + smol_const.expand(x.size())
        return x

    def train(self, oracle_data):

        consistency_check(oracle_data)

        s_batch = to_torch(np.array([self.state_xform.state_to_np(tr.s_i)\
                for tr in oracle_data]))
        f_batch = to_torch(np.array([self.future_xform.truth_to_np(tr.s_t)\
                for tr in oracle_data]))

        future_prediction = self(s_batch)
        loss = xentropy_cost(f_batch, future_prediction)
        # Optimize the model
        self.all_opt.zero_grad()
        loss.backward()
        for param in self.parameters():
            param.grad.data.clamp_(-1, 1)
        self.all_opt.step()

    def best_not_forbid(self, x, forbid):
        with torch.no_grad():
            pred = self.predict(x)
            pred = pred[:,1]
            for forbid_idx in forbid:
                pred[forbid_idx] = -99999
            action_id = np.argmax(pred)
            return action_id, pred[action_id]

    # max probability move
    def max_pr(self, x, forbid, epi):
        if random.random() < epi:
            without_forbid = set(self.action_xform.possible_actions).difference(forbid)
            return random.choice(list(without_forbid))
        else:
            action_id = self.best_not_forbid(x, forbid)[0]
            return self.action_xform.idx_to_action(action_id)

    # max entropy move
    def max_ent(self, x, forbid, epi):
        if random.random() < epi:
            without_forbid = set(self.action_xform.possible_actions).difference(forbid)
            return random.choice(list(without_forbid))
        else:
            with torch.no_grad():
                pred = self.predict(x)
                pred = np.abs(pred[:,1] - 0.5)
                for forbid_idx in forbid:
                    pred[forbid_idx] = -99999
                action_id = np.argmax(pred)
                return self.action_xform.idx_to_action(action_id)

    def act(self, x, forbid, epi):
        max_id, max_pr = self.best_not_forbid(x, forbid)
        #     return self.max_pr(x, forbid, epi)
        if max_pr > 0.51:
            return self.max_pr(x, forbid, epi)
        else:
            return self.max_ent(x, forbid, epi)

