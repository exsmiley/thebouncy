import cv2
import numpy as np
import math

# make parent directory available
import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from _dqn import *
from battleship import *

def measure_dqn(agent, bnd):
  score = 0.0
  for i in range(100):
    env = GameEnv()
    trace = dqn_play_game(env, agent, bnd, 0.0)
    score += sum([tr.r for tr in trace])
  return score / 100

if __name__ == "__main__":
    print ("HEYA")
    state_xform, action_xform = StateXform(), ActionXform()
    dqn_policy = DQN(state_xform, action_xform).to(device)
    dqn_target = DQN(state_xform, action_xform).to(device)

    game_bound = L*L*0.75
    trainer = Trainer(game_bound) 
    trainer.train(dqn_policy, dqn_target, GameEnv, game_bound, measure_dqn)
