import random

def play_game(env, actor, bnd):
  '''
  get a roll-out trace of an actor acting on an environment
  '''
  s = env.reset()
  trace = []
  done = False
  i_iter = 0

  while not done:
    action = actor.act(s)
    ss, r, done = env.step(action)
    trace.append( (s, action, ss, r) )
    s = ss
    # set a bound on the number of turns
    i_iter += 1
    if i_iter > bnd: done = True

  return trace

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


if __name__ == "__main__":
  buff = Buffer(1000)
  for i in range(10000):
    buff.add(i)

  print (buff.sample())

# ============= torch related utils =============

import torch
from torch.autograd import Variable
def to_torch(x, req = False):
  x = Variable(torch.from_numpy(x).type(torch.cuda.FloatTensor), requires_grad = req)
  return x


