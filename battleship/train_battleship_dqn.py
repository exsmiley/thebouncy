import cv2
import numpy as np
import math

# make parent directory available
import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from dqn import *
from battleship import *

if __name__ == "__main__":
    print ("HEYA")
    state_xform, action_xform = StateXform(), ActionXform()
    dqn_policy = DQN(state_xform, action_xform).cuda()
    dqn_target = DQN(state_xform, action_xform).cuda()

    game_bound = L*L*0.75
    trainer = Trainer(game_bound) 

    trainer.train(dqn_policy, dqn_target, GameEnv)
# 
# 
#     for i in range(1000000):
#         if i % 100 == 0:
#             print (" ================= MEASURE    :    ", measure(q_actor, game_bound))
#             print (" state size ", len(q_actor.Q))
#             # print (" everything ? ", q_actor.Q)
#         env = GameEnv()
#         trace = play_game(env, q_actor, game_bound)
#         [buff.add(tr) for tr in trace]
#         tr_sample = [buff.sample() for _ in range(50)]
#         q_actor.learn(tr_sample)
