import numpy as np
import random

# make parent directory available
import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from utils import *
from ac_agent import *

class GameEnv(object):

  def __init__(self):
    self.goal = 5

  def win(self):
    self.goal == self.cur_pos

  def reset(self):
    self.cur_pos = random.randint(0, 10)
    return self.cur_pos

  def step(self, action):
    update = 1 if action == 0 else -1
    state = (self.cur_pos + update) % 10
    self.cur_pos = state
    done = self.win()
    reward = 1.0 if done else -0.1
    return state, reward, done

class StateXform:
  def __init__(self):
    self.length = L*L*3
  def state_to_np(self, state):
    ret = np.zeros(shape=(L*L, 3), dtype=np.float32)
    ret_idx = np.resize(state, L*L)
    for i in range(L*L):
      ret[i, int(ret_idx[i])] = 1.0
    ret = np.resize(ret, L*L*3)
    return ret

class ActionXform:
  def __init__(self):
    self.possible_actions = list(range(L*L))
    self.length = L*L
  def idx_to_action(self, idx):
    return self.possible_actions[idx]
  def action_to_1hot(self, a):
    ret = np.zeros(L*L)
    ret[a] = 1.0
    return ret

if __name__ == "__main__":
  env = GameEnv()
  r_actor = RandomActor(range(2))
  trace = play_game(env, r_actor, 20)
  print (trace)

