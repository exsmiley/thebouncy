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

Tr = namedtuple('Tr', ('s', 'a', 'ss', 'r', 'last'))

from utils import to_torch, to_torch_int, to_torch_byte, device

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

def measure_dqn(env_class, agent, bnd):
    score = 0.0
    for i in range(100):
        env = env_class()
        trace = dqn_play_game(env, agent, bnd, 0.0)
        score += sum([tr.r for tr in trace])
    return score / 100

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, tr):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = tr
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

    def __init__(self, params):
        self.BATCH_SIZE           = params["BATCH_SIZE"]
        self.GAMMA                = params["GAMMA"]
        self.EPS_START            = params["EPS_START"]
        self.EPS_END              = params["EPS_END"]
        self.EPS_DECAY            = params["EPS_DECAY"]
        self.TARGET_UPDATE        = params["TARGET_UPDATE"]
        self.UPDATE_PER_ROLLOUT   = params["UPDATE_PER_ROLLOUT"]
        self.LEARNING_RATE        = params["LEARNING_RATE"]
        self.REPLAY_SIZE          = params["REPLAY_SIZE"]
        self.num_initial_episodes = params["num_initial_episodes"]
        self.num_episodes         = params["num_episodes"]
        self.game_bound           = params["game_bound"]

    def compute_epi(self, steps_done):
        e_s = self.EPS_START
        e_t = self.EPS_END
        e_decay = self.EPS_DECAY
        epi = e_t + (e_s - e_t) * math.exp(-1. * steps_done / e_decay)
        return epi

    def optimize_model(self, policy_net, target_net, transitions, optimizer):

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


    def train(self, policy_net, target_net, env_maker):
        # policy_net = DQN().to(device)
        # target_net = DQN().to(device)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()
        optimizer = optim.RMSprop(policy_net.parameters(), lr = self.LEARNING_RATE)
        memory = ReplayMemory(self.REPLAY_SIZE)

        # collect a lot of initial random trace epi = 1
        for i in range(self.num_initial_episodes):
            trace = dqn_play_game(env_maker(), policy_net, self.game_bound, 1.0) 
            for tr in trace:
                memory.push(tr)

        for i_episode in tqdm.tqdm(range(self.num_episodes)):
            epi = self.compute_epi(i_episode) 

            # collect trace
            trace = dqn_play_game(env_maker(), policy_net, self.game_bound, epi) 
            for tr in trace:
                memory.push(tr)

            # perform 
            if len(memory) > self.BATCH_SIZE:
                for j_train in range(self.UPDATE_PER_ROLLOUT):
                    transitions = memory.sample(self.BATCH_SIZE)
                    self.optimize_model(policy_net, target_net, transitions, optimizer)
 
            # periodically bring target network up to date
            if i_episode % self.TARGET_UPDATE == 0:
                print (" copying over to target network ! ! ! !")
                target_net.load_state_dict(policy_net.state_dict())

            # periodically print out some diagnostics
            if i_episode % 100 == 0:
                print (" ============== i t e r a t i o n ============= ", i_episode)
                print (" episilon ", epi)
                print (" measure ", measure_dqn(env_maker, policy_net, self.game_bound))
                print (" replay size ", len(memory))


