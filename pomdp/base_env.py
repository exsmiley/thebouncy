import random
import os
from parser import POMDPParser


class BaseEnv(object):
    # super class for all pomdp classes

    def __init__(self, filename):
        print('Using {}...'.format(self.__class__.__name__))
        filename = os.path.join(os.path.dirname(__file__), 'specs/{}.pomdp'.format(filename))
        parsed = POMDPParser(filename)
        self.transitions = parsed.T
        self.observations_matrix = parsed.Z
        self.rewards = parsed.R
        self.possible_actions = range(len(parsed.actions))
        self.num_states = len(parsed.states)
        self.num_observations = len(parsed.observations)

        self.actions = []
        self.observations = []

    def _select_from_matrix(self, mat, state, action):
        probs = mat[(action, state)]
        keys = list(range(len(probs)))

        return random.choices(keys, weights=probs)[0]

    def _get_next_state(self, state, action):
        return self._select_from_matrix(self.transitions, state, action)

    def _get_obs(self, state, action):
        return self._select_from_matrix(self.observations_matrix, state, action)

    def _calc_reward(self, start_state, action, next_state, obs):
        if (action, start_state, next_state, obs) in self.rewards:
            return self.rewards[(action, start_state, next_state, obs)]
        else:
            return 0

    def step(self, action):
        next_state = self._get_next_state(self.state, action)
        observation = self._get_obs(self.state, action)
        reward = self._calc_reward(self.state, action, next_state, observation)

        self.state = next_state
        self.actions.append(action)
        self.observations.append(observation)
        done = self.done()

        return observation, reward, done

    def done(self):
        # need a done function
        pass

    def reset(self):
        # need to define this
        pass