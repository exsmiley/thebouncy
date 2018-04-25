import random

# make parent directory available
import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from base_env import BaseEnv
from utils import *


class CheeseEnv(BaseEnv):
    # based on http://cs.brown.edu/research/ai/pomdp/examples/cheese.95.POMDP

    def __init__(self):
        super(CheeseEnv, self).__init__('specs/cheese.pomdp')
        # actions: 0 = N, 1 = S, 2 = E, 3 = W
        
    def won(self):
        return self.state == 9

    def reset(self):
        # start in state 10 because randomly places in a different state
        # and need to randomly start in 0-9 with a random obs
        pre_state = 10
        self.observations = []
        self.actions = []
        self.state = self._get_next_state(pre_state, 0)
        first_obs = self._get_obs(pre_state, 0)
        self.observations.append(first_obs)
        return first_obs


if __name__ == "__main__":
    env = CheeseEnv()
    r_actor = RandomActor(env.possible_actions)
    buff = Buffer(10000)

    for i in range(1):
        trace = play_game(env, r_actor, 1000)
        print(trace)
        for tr in trace:
          buff.add(tr)

    tr_sample = buff.sample_k(3)
    print(tr_sample)




