import cv2
import numpy as np
import math

# make parent directory available
import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from dqn import *
from battleship import *
from battleship_oracle import *

if __name__ == "__main__":
    print ("HEYA")
    state_xform, action_xform = StateXform(), ActionXform()
    truth_xform = StateXformTruth()
    n_hidden = 1000

    oracle = Oracle(state_xform, truth_xform, n_hidden).to(device)
    oracle_xform = OracleXform(oracle)

    dqn_policy = DQN(state_xform, action_xform, n_hidden).to(device)
    dqn_target = DQN(state_xform, action_xform, n_hidden).to(device)

    # dqn_policy = DQN(oracle_xform, action_xform, n_hidden).to(device)
    # dqn_target = DQN(oracle_xform, action_xform, n_hidden).to(device)

    # dqn_policy = DQN(truth_xform, action_xform, n_hidden).to(device)
    # dqn_target = DQN(truth_xform, action_xform, n_hidden).to(device)

    params = {
            "BATCH_SIZE" : 128,
            "GAMMA" : 0.9 ,
            "EPS_START" : 0.9,
            "EPS_END" : 0.05,
            "EPS_DECAY" : 1000,
            "TARGET_UPDATE" : 100 ,
            "UPDATE_PER_ROLLOUT" : 10,
            "LEARNING_RATE" : 0.001,
            "REPLAY_SIZE" : 200000 ,
            "num_initial_episodes" : 100,
            "num_episodes" : 5500,
            "game_bound" : L*L*0.75,
            }

    trainer = JointTrainer(params)
    trainer.pre_train(dqn_policy, dqn_target, oracle, measure_oracle, GameEnv)
