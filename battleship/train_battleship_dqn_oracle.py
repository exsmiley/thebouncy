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
    oracle = Oracle(state_xform, state_xform).to(device)
    oracle_xform = OracleXform(oracle)
    dqn_policy = DQN(oracle_xform, action_xform).to(device)
    dqn_target = DQN(oracle_xform, action_xform).to(device)

    params = {
            "BATCH_SIZE" : 128,
            "GAMMA" : 0.5 ,
            "EPS_START" : 0.99,
            "EPS_END" : 0.05,
            "EPS_DECAY" : 10000,
            "TARGET_UPDATE" : 100 ,
            "UPDATE_PER_ROLLOUT" : 5,
            "LEARNING_RATE" : 0.001,
            "REPLAY_SIZE" : 1000000 ,
            "num_initial_episodes" : 1000,
            "num_episodes" : 100000,
            "game_bound" : L*L*0.75,
            }

    trainer = JointTrainer(params)
    trainer.pre_train(dqn_policy, dqn_target, oracle, measure_oracle, GameEnv)
