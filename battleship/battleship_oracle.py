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

# make parent directory available
import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from utils import to_torch, to_torch_int, to_torch_byte, device

class Oracle(nn.Module):

    def __init__(self, state_xform, future_state_xform):
        super(Oracle, self).__init__()

        assert 0, "#TODO"
        state_length, action_length = state_xform.length, action_xform.length
        self.state_xform, self.action_xform = state_xform, action_xform

        self.enc1  = nn.Linear(state_length, state_length * 10)
        self.bn1 = nn.BatchNorm1d(state_length * 10)
        self.enc2  = nn.Linear(state_length * 10, state_length * 10)
        self.bn2 = nn.BatchNorm1d(state_length * 10)
        self.head = nn.Linear(state_length * 10, action_length)

    def forward(self, x):
        batch_size = x.size()[0]
        def optional_bn(x, bn, size):
            return x if size == 1 else bn(x)
        x = optional_bn(self.enc1(x), self.bn1, batch_size)
        x = optional_bn(self.enc2(x), self.bn2, batch_size)
        return self.head(x)

    def act(self, x, epi):
        if random.random() < epi:
            return random.choice(self.action_xform.possible_actions)
        else:
            with torch.no_grad():
                x = self.state_xform.state_to_np(x)
                x = to_torch(np.expand_dims(x,0))
                q_values = self(x)
                action_id = q_values.max(1)[1].data.cpu().numpy()[0]
                return self.action_xform.idx_to_action(action_id)

    def train_oracle(oracle_data):

        s_batch = to_torch(np.array([policy_net.state_xform.state_to_np(tr.s)\
                for tr in transitions]))
        a_batch = to_torch_int(np.array([[policy_net.action_xform.action_to_idx(tr.a)]\
                for tr in transitions]))
        r_batch = to_torch(np.array([(tr.r)\
                for tr in transitions]))
        ss_batch = to_torch(np.array([policy_net.state_xform.state_to_np(tr.ss)\
                for tr in transitions]))
        fin_batch = torch.Tensor([tr.last for tr in transitions]).byte().to(device)

        # Q[s,a]
        all_sa_values = policy_net(s_batch)
        sa_values = all_sa_values.gather(1, a_batch)

        # V[ss] = max_a(Q[ss,a]) if ss not last_state else 0.0
        all_ssa_values = target_net(ss_batch)
        best_ssa_values = all_ssa_values.max(1)[0].detach()
        ss_values = best_ssa_values.masked_fill_(fin_batch, 0.0)
        
        # Q-target[s,a] = reward[s,a] + discount * V[ss]
        target_sa_values = r_batch + (ss_values * self.GAMMA)
        # Compute Huber loss |Q[s,a] - Q-target[s,a]|
        loss = F.smooth_l1_loss(sa_values, target_sa_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()


