import cv2
import numpy as np
import math

# make parent directory available
import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from dqn import *
from zoombinis import *

def measure_dqn(agent, bnd):
  return measure_dqn_reward(GameEnv, agent, bnd)

if __name__ == "__main__":
    print ("HEYA")
    state_xform, action_xform = StateXform(), ActionXform()
    dqn_policy = DQN(state_xform, action_xform).to(device)
    dqn_target = DQN(state_xform, action_xform).to(device)

    params = {
        "BATCH_SIZE" : 128,
        "GAMMA" : 0.5 ,
        "EPS_START" : 0.99,
        "EPS_END" : 0.05,
        "EPS_DECAY" : 10000,
        "TARGET_UPDATE" : 5000 ,
        "UPDATE_PER_ROLLOUT" : 5,
        "LEARNING_RATE" : 0.001,
        "REPLAY_SIZE" : 100000 ,
        "num_initial_episodes" : 100,
        "num_episodes" : 100000,
        "game_bound" : 20,
        }
    trainer = Trainer(params) 
    trainer.train(dqn_policy, dqn_target, GameEnv)