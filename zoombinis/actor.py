import argparse
import pickle
import numpy as np
import random
import traceback
from itertools import count
from collections import namedtuple
from zoombinis import *
from brain import EntropyRewardOracle, Brain

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
    def __init__(self, pipeline=False):
        super(Policy, self).__init__()
        # input_length = 4
        input_length = AGENT_INPUT_LENGTH
        if pipeline:
            input_length = NUM_ZOOMBINIS*(NUM_BRIDGES+1)#OUTPUT_LENGTH
        layer_size = 128
        # layer_size = 1000
        self.affine1 = nn.Linear(input_length, layer_size)
        self.action_head = nn.Linear(layer_size, OUTPUT_LENGTH)
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
        return action.data[0]

    def select_action2(self, state, invalid_moves):
        # zeros out any unavailable options and rebalances to make sum to 1
        state = torch.from_numpy(state).float()
        probs, state_value = self.forward(Variable(state))
        # print(invalid_moves)
        # print(probs)
        m = Categorical(probs)
        # action = m.sample()

        probs2 = probs.data.numpy()
        # if np.isnan(np.min(probs2)):
        #     print(self.probs_list)
        #     print(probs2, 'probs2')
        #     print(probs.data.numpy(), 'probs')
        #     quit()
        # probs2 = np.nan_to_num(probs2)+1e-20
        invalid_moves2 = set(invalid_moves)
        for ind in invalid_moves:
            probs2[ind] = 1e-30

        for i in range(len(probs2)):
            if i in invalid_moves2:
                continue
            probs2[i] = max(probs2[i], 1e-25)
        probs2 = probs2 / np.sum(probs2)
        # print(probs2)
        action = np.random.choice(len(probs2), p=probs2)
        action = Variable(torch.from_numpy(np.array([action])))
        # print(probs2, action)
        # print(probs2)
        # if len(self.probs_list) > 10:
        #     self.probs_list.pop(0)
        # self.probs_list.append((probs.data.numpy(), invalid_moves))
        
        self.saved_actions.append(SavedAction(m.log_prob(action), state_value))
        return action.data[0]

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
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
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

    if USE_SHAPER:
        reward_shaper = EntropyRewardOracle()
    if PIPELINE:
        brain = Brain('models/brain')

    running_reward = 1

    for i_episode in range(10000):
        state = env.reset()
        total_reward = 0
        made_actions = []
        for t in range(100):  # Don't infinite loop while learning
            invalid_moves = env.get_invalid_moves()

            if PIPELINE:
                state = np.array(brain.get_probabilities_total(env.game.get_brain_state(), env.game.known))
                # state = np.array(env.game.get_brain_truth())

            action = model.select_action2(state, invalid_moves+made_actions)
            made_actions.append(action)

            if USE_SHAPER:
                state, reward, done, additional = env.step(action, reward_shaper=reward_shaper)
            else:   
                state, reward, done = env.step(action)
                additional = 0

            total_reward += reward

            # if done and not env.game.has_won():
            #     reward = -100

            model.rewards.append(reward+additional)
            if done:
                made_actions = []
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


class ActorPlayer(object):

    def __init__(self):
        self.policy = Policy()
        self.brain = Brain(chkpt='models/brain')
        self.policy.load()

    def play(self, game):
        # print(game)
        # print(game.truth)
        truth = game.get_brain_truth()
        scores = []
        running_scores = []
        actual_score = 0
        moved = []
        while game.can_move():
            invalid_moves = game.get_invalid_moves() + moved
            state = np.array(game.get_agent_state())
            action = self.policy.select_action2(state, invalid_moves)
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

            zoombini = action//NUM_BRIDGES
            bridge = action % NUM_BRIDGES
            passed = game.send_zoombini(zoombini, bridge)
            if passed:
                actual_score += 1
            running_scores.append(actual_score)

        return game.has_won(), scores, actual_score, running_scores

class ActorShapedPlayer(ActorPlayer):

    def __init__(self):
        self.policy = Policy()
        self.brain = Brain(chkpt='models/brain')
        self.policy.load('models/actor_shaped')

class ActorPipelinePlayer(ActorPlayer):

    def __init__(self):
        self.policy = Policy(pipeline=True)
        self.brain = Brain(chkpt='models/brain')
        self.policy.load('models/actor_pipeline')

    def play(self, game):
        # print(game)
        # print(game.truth)
        truth = game.get_brain_truth()
        scores = []
        running_scores = []
        actual_score = 0
        moved = []
        while game.can_move():
            invalid_moves = game.get_invalid_moves() + moved
            state = np.array(self.brain.get_probabilities_total(game.get_brain_state(), game.known))
            action = self.policy.select_action2(state, invalid_moves)
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

            zoombini = action//NUM_BRIDGES
            bridge = action % NUM_BRIDGES
            passed = game.send_zoombini(zoombini, bridge)
            if passed:
                actual_score += 1

            running_scores.append(actual_score)

        return game.has_won(), scores, actual_score, running_scores


class ActorPipelinePlayer2(ActorPlayer):

    def __init__(self):
        self.policy = Policy(pipeline=True)
        self.brain = Brain(chkpt='models/brain_pipelined')
        self.entropy_brain = Brain(chkpt='models/brain')
        self.policy.load('models/actor_pipelined2')

    def play(self, game):
        truth = game.get_brain_truth()
        scores = []
        running_scores = []
        actual_score = 0
        moved = []
        while game.can_move():
            invalid_moves = game.get_invalid_moves() + moved
            state = np.array(self.brain.get_probabilities_total(game.get_brain_state(), game.known))
            action = self.policy.select_action2(state, invalid_moves)
            moved.append(action)

            state = game.get_brain_state()
            probs = self.entropy_brain.get_probabilities(state)
            score = 0

            for i in range(0, len(truth), NUM_BRIDGES):
                truths = truth[i:i+NUM_BRIDGES]
                preds = probs[i:i+NUM_BRIDGES]
                if np.argmax(truths) == np.argmax(preds):
                    score += 1

            scores.append(score)

            zoombini = action//NUM_BRIDGES
            bridge = action % NUM_BRIDGES
            passed = game.send_zoombini(zoombini, bridge)
            if passed:
                actual_score += 1

            running_scores.append(actual_score)

        return game.has_won(), scores, actual_score, running_scores



if __name__ == '__main__':
    try:
        model = Policy(pipeline=PIPELINE)
        running_reward_list = []
        main(model, running_reward_list)
    except:
        traceback.print_exc()
    finally:
        if USE_SHAPER:
            model.save('models/actor_shaped')
        elif PIPELINE:
            model.save('models/actor_pipeline')
        else:
            model.save()
        import matplotlib.pyplot as plt
        plt.plot(running_reward_list)
        plt.show()

