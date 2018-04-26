import random
import numpy as np
from collections import namedtuple
Tr = namedtuple('Tr', 's a ss r, v')

def play_game(env, actor, bnd):
  '''
  get a roll-out trace of an actor acting on an environment
  '''
  s = env.reset()
  trace = []
  done = False
  i_iter = 0

  while not done:
    action, action_pr = actor.act(s)
    action_pr = round(action_pr[action], 2)
    ss, r, done = env.step(action)
    trace.append( Tr(s, action, ss, r, action_pr) )
    s = ss
    # set a bound on the number of turns
    i_iter += 1
    if i_iter > bnd: done = True

  return trace

def get_discount_trace(trace, value_estimator):
  '''
  get the discounted future rewards
  [0, 1, 0, 0, 1] => [0.99*(1 + 0.99^3), 1 + 0.99 ^3, 0.99^2, 0.99, 1]
  '''
  rewards = [tr.r for tr in trace]
  discount_reward = [0.0]
  for i in range(1, len(rewards)+1):
    future_r = 0.99 * discount_reward[-i]
    cur_r  = rewards[-i]
    discount_reward.insert(0, cur_r + future_r)

  # normalize data
  discount_reward = discount_reward[:-1]
  discount_reward = np.array(discount_reward)
  discount_reward -= np.mean(discount_reward)
  discount_reward /= (np.std(discount_reward) + 1e-3)

  discount_trace = []
  for i, tr in enumerate(trace):
    discount_trace.append( Tr(tr.s, tr.a, tr.ss, discount_reward[i], value_estimator(tr.s)) )

  return discount_trace


class RandomActor:
  '''
  a random actor that uniformly choose the actions disregarding the state
  '''
  def __init__(self, possible_actions):
    self.possible_actions = possible_actions

  def act(self, state):
    return random.choice(self.possible_actions)

class Buffer:
  '''
  a buffer that supports sample and add, the type of things being added is left
  to the user
  '''
  
  def __init__(self, buff_len):
    self.buff_len = buff_len
    self.buff = []

  def trim(self):
    if len(self.buff) > self.buff_len:
      self.buff = self.buff[self.buff_len // 10:]

  def add(self, tup):
    self.trim()
    self.buff.append(tup)

  def sample(self):
    return random.choice(self.buff)

  def sample_k(self, k):
    return random.choices(self.buff, k=k)


if __name__ == "__main__":
  buff = Buffer(1000)
  for i in range(10000):
    buff.add(i)

  print (buff.sample())

# ============= torch related utils =============

import torch
from torch.autograd import Variable
def to_torch(x, req = False, cuda=True):
  dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor
  x = Variable(torch.from_numpy(x).type(dtype), requires_grad = req)
  return x

def to_torch_int(x, req = False, cuda=True):
  dtype = torch.cuda.LongTensor if cuda else torch.LongTensor
  x = Variable(torch.from_numpy(x).type(dtype), requires_grad = req)
  return x

