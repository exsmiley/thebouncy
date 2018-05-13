import cv2
import numpy as np
import math

# make parent directory available
import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from dqn import *
from zoombinis import *
from brain import *
from baselines import MaxEntropyPlayer

USE_ORACLE = False


if __name__ == "__main__":
    print ("ZOOMBINIS")
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
        "num_initial_episodes" : 5000,
        "num_episodes" : 100000,
        "game_bound" : 20,
        "entropy_scale_factor": 0.1,
        }
    n_hidden = 128
    state_xform, action_xform, future_xform = StateXform(), ActionXform(), FutureXform()

    if USE_ORACLE:
        oracle = Oracle(state_xform, future_xform, n_hidden).to(device)
        state_xform = OracleXform(oracle)

    dqn_policy = DQN(state_xform, action_xform, n_hidden).to(device)
    dqn_target = DQN(state_xform, action_xform, n_hidden).to(device)
    
    if USE_ORACLE:
        trainer = JointTrainer(params)
        trainer.pre_train(dqn_policy, dqn_target, oracle, measure_oracle, GameEnv)
    else:
        trainer = Trainer(params)
        trainer.train(dqn_policy, dqn_target, GameEnv)

    
