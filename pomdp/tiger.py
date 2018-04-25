import random

# make parent directory available
import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from base_env import BaseEnv
from utils import *


class TigerEnv(BaseEnv):
    # based on http://cs.brown.edu/research/ai/pomdp/examples/tiger.aaai.POMDP

    def __init__(self):
        super(TigerEnv, self).__init__('tiger.aaai')

    def done(self):
        return 1 in self.actions or 2 in self.actions

    def reset(self):
        # start in state 10 because randomly places in a different state
        # and need to randomly start in 0-9 with a random obs
        self.state = 0
        first_obs = 1
        self.actions = []
        self.observations = []
        return None


if __name__ == "__main__":
    env = TigerEnv()
    r_actor = RandomActor(env.possible_actions)
    buff = Buffer(10000)

    for i in range(1):
        trace = play_game(env, r_actor, 1000)
        print(trace)
        for tr in trace:
          buff.add(tr)

    tr_sample = buff.sample_k(3)