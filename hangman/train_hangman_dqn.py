import cv2
import numpy as np
import math

# make parent directory available
import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from dqn import *
from hangman import *



if __name__ == "__main__":
    print ("HANGMAN")
    params = {
        "BATCH_SIZE" : 50,
        "GAMMA" : 0.5 ,
        "EPS_START" : 0.9,
        "EPS_END" : 0.05,
        "EPS_DECAY" : 2500,
        "TARGET_UPDATE" : 20 ,
        "UPDATE_PER_ROLLOUT" : 1,
        "LEARNING_RATE" : 0.0001,
        "REPLAY_SIZE" : 10000 ,
        "num_initial_episodes" : 0,
        "num_episodes" : 10001,
        "game_bound" : 20,
        }
    n_hidden = 128
    state_xform, action_xform = StateXform(), ActionXform()

    dqn_policy = DQN(state_xform, action_xform, n_hidden).to(device)
    dqn_target = DQN(state_xform, action_xform, n_hidden).to(device)

    trainer = Trainer(params)
    trainer.train(dqn_policy, dqn_target, HangmanEnv)

    
