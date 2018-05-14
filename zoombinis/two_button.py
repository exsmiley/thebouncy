import argparse
import pickle
import numpy as np
import random
import traceback
from itertools import count
from collections import namedtuple
from zoombinis import *
from brain import EntropyRewardOracle, Brain
from baselines import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
np.set_printoptions(suppress=True)

USE_SHAPER = False
PIPELINE = True

GAMMA = 0.99
SEED = 543
LOG_INTERVAL = 10

torch.manual_seed(SEED)

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        # input_length = 4
        input_length = BRAIN_INPUT_LENGTH
        layer_size = 128
        # layer_size = 1000
        self.affine1 = nn.Linear(input_length, layer_size)
        self.action_head = nn.Linear(layer_size, 2)
        self.value_head = nn.Linear(layer_size, 1)

        self.saved_actions = []
        self.rewards = []
        self.probs_list = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=-1), state_values

    def load(self, name='models/actor'):
        self.load_state_dict(torch.load(name))
        print('Loaded model from {}!'.format(name))

    def save(self, name='models/actor'):
        torch.save(self.state_dict(), name)
        print('Saved model to {}!'.format(name))

    def select_action(self, state):
        state = torch.from_numpy(state).float()
        probs, state_value = self.forward(Variable(state))
    
        m = Categorical(probs)
        action = m.sample()
        self.saved_actions.append(SavedAction(m.log_prob(action), state_value))
        return action.data


def finish_episode(model, optimizer):
    R = 0
    saved_actions = model.saved_actions
    policy_losses = []
    value_losses = []
    rewards = []
    for r in model.rewards[::-1]:
        R = r + GAMMA * R
        rewards.insert(0, R)
    rewards = torch.Tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + float(np.finfo(np.float32).eps))
    for (log_prob, value), r in zip(saved_actions, rewards):
        advantage = r - value.data[0]
        policy_losses.append(-log_prob * advantage)
        value_losses.append(F.smooth_l1_loss(value, Variable(torch.Tensor([r]))))

    optimizer.zero_grad()
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    loss.backward()
    optimizer.step()
    del model.rewards[:]
    del model.saved_actions[:]


def main(model, running_reward_list):
    env = GameEnv()
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    max_ent_player = MaxEntropyPlayer()
    max_prob_player = MaxProbabilityPlayer()

    running_reward = 1

    for i_episode in count(1):
        state = env.reset()
        total_reward = 0
        made_actions = set()
        for t in range(30):  # Don't infinite loop while learning
            invalid_moves = env.game.get_invalid_moves()

            action = model.select_action(state[0])
            player = max_ent_player if action else max_prob_player
            actual_action = player.act(env.game, set(invalid_moves))

            state, reward, done = env.step(actual_action)

            total_reward += reward

            # if reward <= 0:
            #     reward = -1

            if done and not env.game.has_won():
                reward = -100

            model.rewards.append(reward)
            if done:
                # made_actions = []
                break

        if i_episode == 0:
            running_reward = running_reward
        else:
            running_reward = running_reward * 0.99 + total_reward * 0.01

        running_reward_list.append(running_reward)
        finish_episode(model, optimizer)
        if i_episode % LOG_INTERVAL == 0:
            print('Episode {}\tLast length/reward: {:5d}/{}\tAverage reward: {:.2f}'.format(
                i_episode, t+1, total_reward, running_reward))
        if running_reward > env.winning_threshold:#env.spec.reward_threshold:
            print("Solved {}! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(i_episode, running_reward, t))
            break


class TwoButtonPlayer(object):

    def __init__(self):
        self.model = Policy()
        self.brain = Brain(chkpt='models/brain_entropy_mask')
        self.model.load()
        self.max_ent_player = MaxEntropyPlayer()
        self.max_prob_player = MaxProbabilityPlayer()

    def play(self, game):
        # print(game)
        # print(game.truth)
        truth = game.get_brain_truth()
        scores = []
        running_scores = []
        actual_score = 0
        moved = []
        state = game.get_brain_state()
        while game.can_move():
            invalid_moves = game.get_invalid_moves() + moved
            action = self.model.select_action(np.array(state))
            player = self.max_ent_player if action else self.max_prob_player
            actual_action = player.act(game, set(invalid_moves))
            moved.append(action)

            state = game.get_brain_state()
            probs = self.brain.get_probabilities(state)
            score = 0

            for i in range(0, len(truth), NUM_BRIDGES):
                truths = truth[i:i+NUM_BRIDGES]
                preds = probs[i:i+NUM_BRIDGES]
                if np.argmax(truths) == np.argmax(preds):
                    score += 1

            scores.append(score)

            zoombini = actual_action//NUM_BRIDGES
            bridge = actual_action % NUM_BRIDGES
            passed = game.send_zoombini(zoombini, bridge)
            if passed:
                actual_score += 1
            running_scores.append(actual_score)

        return game.has_won(), scores, actual_score, running_scores


if __name__ == '__main__':
    try:
        model = Policy()
        running_reward_list = []
        main(model, running_reward_list)
    except:
        traceback.print_exc()
    finally:
        model.save()
        import matplotlib.pyplot as plt
        plt.plot(running_reward_list)
        plt.show()

