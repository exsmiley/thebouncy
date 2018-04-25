import random
from pomdp.base_env import BaseEnv
from utils import *


class CheeseEnv(BaseEnv):
    # based on http://cs.brown.edu/research/ai/pomdp/examples/cheese.95.POMDP

    def __init__(self):
        # actions: 0 = N, 1 = S, 2 = E, 3 = W
        self.possible_actions = list(range(4))
        self.num_states = 11
        self.num_observations = 7
        
        # transition format {state: {action: resulting state}}
        self.transitions = {
            1: {0: 1, 1: 6, 2: 2, 3: 1},
            2: {0: 2, 1: 2, 2: 3, 3: 1},
            3: {0: 3, 1: 7, 2: 4, 3: 2},
            4: {0: 4, 1: 4, 2: 5, 3: 3},
            5: {0: 5, 1: 8, 2: 5, 3: 4},
            6: {0: 1, 1: 9, 2: 6, 3: 6},
            7: {0: 3, 1: 11, 2: 7, 3: 7},
            8: {0: 5, 1: 10, 2: 8, 3: 8},
            9: {0: 6, 1: 9, 2: 9, 3: 9},
            10: {0: 8, 1: 10, 2: 10, 3: 10},
            11: {0: {i: 0.1 for i in range(1, 11)}, 1: {i: 0.1 for i in range(1, 11)}, 2: {i: 0.1 for i in range(1, 11)}, 3: {i: 0.1 for i in range(1, 11)}},
        }

        # observation matrix format {state: {action: resulting state}}
        self.observations_matrix = {
            1: {0: 1, 1: 1, 2: 1, 3: 1},
            2: {0: 2, 1: 2, 2: 2, 3: 2},
            3: {0: 3, 1: 3, 2: 3, 3: 3},
            4: {0: 2, 1: 2, 2: 2, 3: 2},
            5: {0: 4, 1: 4, 2: 4, 3: 4},
            6: {0: 5, 1: 5, 2: 5, 3: 5},
            7: {0: 5, 1: 5, 2: 5, 3: 5},
            8: {0: 5, 1: 5, 2: 5, 3: 5},
            9: {0: 6, 1: 6, 2: 6, 3: 6},
            10: {0: 6, 1: 6, 2: 6, 3: 6},
            11: {0: 7, 1: 7, 2: 7, 3: 7},
        }

        self.rewards = {
            i: 0 for i in range(1, 12)
        }
        self.rewards[10] = 1

    def won(self):
        return self.state == 10

    def reset(self):
        # start in state 11 because randomly places in a different state
        self.state = 11
        self.observations = []
        self.actions = []
        self.state = self._select_from_matrix(self.transitions, 0)
        first_obs = self._select_from_matrix(self.observations_matrix, 0)
        self.observations.append(first_obs)
        return first_obs

    def step(self, action):
        next_state = self._select_from_matrix(self.transitions, action)
        observation = self._select_from_matrix(self.observations_matrix, action)
        reward = self._calc_reward()

        won = self.won()
        self.state = next_state

        self.actions.append(action)
        self.observations.append(observation)

        return observation, reward, won


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




