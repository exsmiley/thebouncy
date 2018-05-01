import math
import random
import numpy as np
from collections import namedtuple
from itertools import count

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

Tr = namedtuple('Tr', ('state', 'action', 'next_state', 'reward', 'last'))

from utils import to_torch

def dqn_play_game(env, actor, bnd, epi):
    '''
    get a roll-out trace of an actor acting on an environment
    env : the environment
    actor : the actor
    bnd : the length of game upon which we force termination
    epi : the episilon greedy exploration
    '''
    s = env.reset()
    trace = []
    done = False
    i_iter = 0

    while not done:
        action = actor.act(s, epi)
        ss, r, done = env.step(action)
        # set a bound on the number of turns
        i_iter += 1
        if i_iter > bnd: 
            done = True
            # r = env.get_final_reward()

        trace.append( Tr(s, action, ss, r, done) )
        s = ss

    return trace

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, state_xform, action_xform):
        super(DQN, self).__init__()
        state_length, action_length = state_xform.length, action_xform.length
        self.state_xform, self.action_xform = state_xform, action_xform

        self.enc  = nn.Linear(state_length, state_length * 10)
        self.bn = nn.BatchNorm1d(state_length * 10)
        self.head = nn.Linear(state_length * 10, action_length)

    def forward(self, x):
        batch_size = x.size()[0]
        x = F.relu(self.enc(x)) if batch_size == 1 else F.relu(self.bn(self.enc(x)))
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

class Trainer:

    def __init__(self, game_bound):
        self.BATCH_SIZE = 128
        self.GAMMA = 0.999
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 200
        self.TARGET_UPDATE = 10
        self.num_episodes = 1000

        self.game_bound = game_bound

    def compute_epi(self, steps_done):
        e_s = self.EPS_START
        e_t = self.EPS_END
        e_decay = self.EPS_DECAY
        epi = e_t + (e_s - e_t) * math.exp(-1. * steps_done / e_decay)
        return epi

    def optimize_model(self, memory):
        # if we don't even have enough stuff just don't learn
        if len(memory) < self.BATCH_SIZE:
            return
        transitions = memory.sample(BATCH_SIZE)
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()


    def train(self, policy_net, target_net, env_maker):
        # policy_net = DQN().to(device)
        # target_net = DQN().to(device)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()
        optimizer = optim.RMSprop(policy_net.parameters())
        memory = ReplayMemory(10000)

        for i_episode in range(self.num_episodes):
            epi = self.compute_epi(i_episode) 
            print (" = = = = = = ", i_episode, " ", epi)
            trace = dqn_play_game(env_maker(), policy_net, self.game_bound, epi) 
            print (trace)
#                 # Move to the next state
#                 state = next_state
# 
#                 # Perform one step of the optimization (on the target network)
#                 optimize_model()
#                 if done:
#                     episode_durations.append(t + 1)
#                     plot_durations()
#                     break
#             # Update the target network
#             if i_episode % TARGET_UPDATE == 0:
#                 target_net.load_state_dict(policy_net.state_dict())
# 



