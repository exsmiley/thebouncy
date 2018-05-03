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
    total_score = 0
    obtained_score = 0
    for od in oracle_datas:
        current_ob, target_ob = od.s_i, od.s_t
        prediction = oracle.predict(current_ob)
        target = oracle.future_xform.state_to_np(target_ob)
        target = np.reshape(target, (L*L, 2))
        argmax_pred = np.argmax(prediction, axis=1)
        argmax_target = np.argmax(target, axis=1)

        for jjj in range(len(target)):
            if sum(target[jjj]) > 0:
                total_score += 1
                obtained_score += 1 if argmax_pred[jjj] == argmax_target[jjj] else 0
    return obtained_score / total_score
                

class Oracle(nn.Module):

    def __init__(self, state_xform, future_xform):
        super(Oracle, self).__init__()

        state_length, future_length = state_xform.length, future_xform.length
        self.state_xform, self.future_xform = state_xform, future_xform

        self.enc1  = nn.Linear(state_length, state_length * 10)
        self.bn1 = nn.BatchNorm1d(state_length * 10)
        self.enc2  = nn.Linear(state_length * 10, state_length * 10)
        self.bn2 = nn.BatchNorm1d(state_length * 10)
        self.head = nn.Linear(state_length * 10, future_length)

        self.all_opt = torch.optim.RMSprop(self.parameters(), lr=0.001)

    def predict(self, state):
        state = to_torch(np.array([self.state_xform.state_to_np(state)]))
        prediction = self(state).data.cpu().numpy()[0]
        return prediction

    def forward(self, x):
        batch_size = x.size()[0]
        def optional_bn(x, bn, size):
            return x if size == 1 else bn(x)
        x = optional_bn(self.enc1(x), self.bn1, batch_size)
        x = optional_bn(self.enc2(x), self.bn2, batch_size)
        x = self.head(x)
        x = x.view(-1, self.future_xform.n_obs, 2)
        x = F.softmax(x, dim=2)
        smol_const = to_torch(np.array([1e-6]))
        x = x + smol_const.expand(x.size())
        return x

    def train(self, oracle_data):

        s_batch = to_torch(np.array([self.state_xform.state_to_np(tr.s_i)\
                for tr in oracle_data]))
        f_batch = to_torch(np.array([self.future_xform.target_state_to_np(tr.s_t)\
                for tr in oracle_data]))

        future_prediction = self(s_batch)
        loss = xentropy_cost(f_batch, future_prediction)
        # Optimize the model
        self.all_opt.zero_grad()
        loss.backward()
        for param in self.parameters():
            param.grad.data.clamp_(-1, 1)
        self.all_opt.step()


