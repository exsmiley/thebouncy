import cv2
import numpy as np
import math

# make parent directory available
import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from dqnexp import *
from battleship import *
from battleship_oracle import *

if __name__ == "__main__":
    print ("HEYA")
    state_xform, action_xform = StateXform(), ActionXform()
    truth_xform = StateXformTruth()
    n_hidden = 128

    dqn_policy = DQN(state_xform, action_xform, n_hidden).to(device)
    dqn_target = DQN(state_xform, action_xform, n_hidden).to(device)

    params = {
            "BATCH_SIZE" : 50,
            "GAMMA" : 0.9 ,
            "EPS_START" : 0.9,
            "EPS_END" : 0.05,
            "EPS_DECAY" : 500,
            "TARGET_UPDATE" : 10 ,
            "UPDATE_PER_ROLLOUT" : 1,
            "LEARNING_RATE" : 0.0001,
            "REPLAY_SIZE" : 10000 ,
            "num_initial_episodes" : 0,
            "num_episodes" : 2001,
            "game_bound" : L*L*0.5,
            }

    trainer = JointTrainer(params)
    trainer.policy_only(dqn_policy, dqn_target, GameEnv)
