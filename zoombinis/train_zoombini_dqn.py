import cv2
import numpy as np
import math

# make parent directory available
import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from dqn import *
from zoombinis import *


if __name__ == "__main__":
    print ("ZOOMINIIIISSSSSS")
    n_hidden = 128
    state_xform, action_xform, future_xform = StateXform(), ActionXform(), FutureXform()

    oracle = Oracle(state_xform, future_xform, n_hidden).to(device)
    oracle_xform = OracleXform(oracle)

    dqn_policy = DQN(oracle_xform, action_xform, n_hidden).to(device)
    dqn_target = DQN(oracle_xform, action_xform, n_hidden).to(device)
    

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
    trainer = JointTrainer(params)
    trainer.pre_train(dqn_policy, dqn_target, oracle, measure_oracle, GameEnv)
