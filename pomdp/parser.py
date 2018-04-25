# Adapted from https://github.com/mbforbes/py-pomdp/blob/master/pomdp.py

import xml.etree.ElementTree as ET

# 3rd party
import numpy as np

class POMDPParser:
    def __init__(self, filename):
        """
        Parses .pomdp file and loads info into this object's fields.
        Attributes:
            discount
            values
            states
            actions
            observations
            T
            Z
            R
        """
        f = open(filename, 'r')
        self.contents = [
            x.strip() for x in f.readlines()
            if (not (x.startswith("#") or x.isspace()))
        ]

        # set up transition function T, observation function Z, and
        # reward R
        self.T = {}
        self.Z = {}
        self.R = {}
        self.start = []

        # go through line by line
        i = 0
        while i < len(self.contents):
            line = self.contents[i]
            if line.startswith('discount'):
                i = self.__get_discount(i)
            elif line.startswith('values'):
                i = self.__get_value(i)
            elif line.startswith('states'):
                i = self.__get_states(i)
            elif line.startswith('actions'):
                i = self.__get_actions(i)
            elif line.startswith('observations'):
                i = self.__get_observations(i)
            elif line.startswith('T'):
                i = self.__get_transition(i)
            elif line.startswith('O'):
                i = self.__get_observation(i)
            elif line.startswith('R'):
                i = self.__get_reward(i)
            elif line.startswith('start'):
                i = self.__get_start(i)
            else:
                raise Exception("Unrecognized line: " + line)

        # cleanup
        f.close()

        # transform T (action, start, next): prob -> (action, start) -> [prob, prob, ...]
        self.T2 = {}
        for (action, start, ns), prob in self.T.items():
            if (action, start) in self.T2:
                self.T2[(action, start)][ns] = prob
            else:
                self.T2[(action, start)] = {ns: prob}

        for k, v in self.T2.items():
            probs = [0 for i in range(len(v))]
            for i, prob in v.items():
                probs[i] = prob
            self.T2[k] = probs
        self.T = self.T2

        # transform Z (action, start, obs): prob -> (action, start) -> [prob, prob, ...]
        self.Z2 = {}
        for (action, start, obs), prob in self.Z.items():
            if (action, start) in self.Z2:
                self.Z2[(action, start)][obs] = prob
            else:
                self.Z2[(action, start)] = {obs: prob}

        for k, v in self.Z2.items():
            probs = [0 for i in range(len(v))]
            for i, prob in v.items():
                probs[i] = prob
            self.Z2[k] = probs
        self.Z = self.Z2

    def __get_start(self, i):
        self.start = self.contents[i+1].split()

        return i+2

    def __get_discount(self, i):
        line = self.contents[i]
        self.discount = float(line.split()[1])
        return i + 1

    def __get_value(self, i):
        # Currently just supports "values: reward". I.e. currently
        # meaningless.
        line = self.contents[i]
        self.values = line.split()[1]
        return i + 1

    def __get_states(self, i):
        # TODO support states as number
        line = self.contents[i]
        self.states = line.split()[1:]
        if len(self.states) == 1:
            try:
                num_states = int(self.states[0])
                self.states = list(range(num_states))
            except:
                pass
        return i + 1

    def __get_actions(self, i):
        line = self.contents[i]
        self.actions = line.split()[1:]
        return i + 1

    def __get_observations(self, i):
        line = self.contents[i]
        self.observations = line.split()[1:]
        if len(self.observations) == 1:
            try:
                num_obs = int(self.observations[0])
                self.observations = list(range(num_obs))
            except:
                pass
        return i + 1

    def __get_transition(self, i):
        line = self.contents[i]
        pieces = [x for x in line.split() if (x.find(':') == -1)]
        action = self.actions.index(pieces[0])

        if len(pieces) == 4:
            # case 1: T: <action> : <start-state> : <next-state> %f
            start_state = self.states.index(pieces[1])
            next_state = self.states.index(pieces[2])
            prob = float(pieces[3])
            self.T[(action, start_state, next_state)] = prob
            return i + 1
        elif len(pieces) == 3:
            # case 2: T: <action> : <start-state> : <next-state>
            # %f
            start_state = self.states.index(pieces[1])
            next_state = self.states.index(pieces[2])
            next_line = self.contents[i+1]
            prob = float(next_line)
            self.T[(action, start_state, next_state)] = prob
            return i + 2
        elif len(pieces) == 2:
            # case 3: T: <action> : <start-state>
            # %f %f ... %f
            start_state = self.states.index(pieces[1])
            next_line = self.contents[i+1]
            probs = next_line.split()
            assert len(probs) == len(self.states)
            for j in range(len(probs)):
                prob = float(probs[j])
                self.T[(action, start_state, j)] = prob
            return i + 2
        elif len(pieces) == 1:
            next_line = self.contents[i+1]
            if next_line == "identity":
                # case 4: T: <action>
                # identity
                for j in range(len(self.states)):
                    for k in range(len(self.states)):
                        prob = 1.0 if j == k else 0.0
                        self.T[(action, j, k)] = prob
                return i + 2
            elif next_line == "uniform":
                # case 5: T: <action>
                # uniform
                prob = 1.0 / float(len(self.states))
                for j in range(len(self.states)):
                    for k in range(len(self.states)):
                        self.T[(action, j, k)] = prob
                return i + 2
            else:
                # case 6: T: <action>
                # %f %f ... %f
                # %f %f ... %f
                # ...
                # %f %f ... %f
                for j in range(len(self.states)):
                    probs = next_line.split()
                    assert len(probs) == len(self.states)
                    for k in range(len(probs)):
                        prob = float(probs[k])
                        self.T[(action, j, k)] = prob
                    next_line = self.contents[i+2+j]
                return i+1+len(self.states)
        else:
            raise Exception("Cannot parse line " + line)

    def __get_observation(self, i):
        line = self.contents[i]
        pieces = [x for x in line.split() if (x.find(':') == -1)]
        action = self.actions.index(pieces[0])

        if len(pieces) == 4:
            # case 1: O: <action> : <next-state> : <obs> %f
            next_state = self.states.index(pieces[1])
            obs = self.observations.index(pieces[2])
            prob = float(pieces[3])
            self.Z[(action, next_state, obs)] = prob
            return i + 1
        elif len(pieces) == 3:
            # case 2: O: <action> : <next-state> : <obs>
            # %f
            next_state = self.states.index(pieces[1])
            obs = self.observations.index(pieces[2])
            next_line = self.contents[i+1]
            prob = float(next_line)
            self.Z[(action, next_state, obs)] = prob
            return i + 2
        elif len(pieces) == 2:
            # case 3: O: <action> : <next-state>
            # %f %f ... %f
            next_state = self.states.index(pieces[1])
            next_line = self.contents[i+1]
            probs = next_line.split()
            assert len(probs) == len(self.observations)
            for j in range(len(probs)):
                prob = float(probs[j])
                self.Z[(action, next_state, j)] = prob
            return i + 2
        elif len(pieces) == 1:
            next_line = self.contents[i+1]
            if next_line == "identity":
                # case 4: O: <action>
                # identity
                for j in range(len(self.states)):
                    for k in range(len(self.observations)):
                        prob = 1.0 if j == k else 0.0
                        self.Z[(action, j, k)] = prob
                return i + 2
            elif next_line == "uniform":
                # case 5: O: <action>
                # uniform
                prob = 1.0 / float(len(self.observations))
                for j in range(len(self.states)):
                    for k in range(len(self.observations)):
                        self.Z[(action, j, k)] = prob
                return i + 2
            else:
                # case 6: O: <action>
                # %f %f ... %f
                # %f %f ... %f
                # ...
                # %f %f ... %f
                for j in range(len(self.states)):
                    probs = next_line.split()
                    assert len(probs) == len(self.observations)
                    for k in range(len(probs)):
                        prob = float(probs[k])
                        self.Z[(action, j, k)] = prob
                    next_line = self.contents[i+2+j]
                return i + 1 + len(self.states)
        else:
            raise Exception("Cannot parse line: " + line)

    def __get_reward(self, i):
        """
        Wild card * are allowed when specifying a single reward
        probability. They are not allowed when specifying a vector or
        matrix of probabilities.
        """
        line = self.contents[i]
        pieces = [x for x in line.split() if (x.find(':') == -1)]

        if pieces[0] != '*':
            action = self.actions.index(pieces[0])
            return self.__get_reward_helper(i, action, pieces)
        else:
            next_i = None
            for action in range(len(self.actions)):
                next_i = self.__get_reward_helper(i, action, pieces)
            return next_i

    def __get_reward_helper(self, i, action, pieces):
        if len(pieces) == 5 or len(pieces) == 4:
            # case 1:
            # R: <action> : <start-state> : <next-state> : <obs> %f
            # any of <start-state>, <next-state>, and <obs> can be *
            # %f can be on the next line (case where len(pieces) == 4)
            start_state_raw = pieces[1]
            next_state_raw = pieces[2]
            obs_raw = pieces[3]
            prob = float(pieces[4]) if len(pieces) == 5 \
                else float(self.contents[i+1])
            self.__reward_ss(
                action, start_state_raw, next_state_raw, obs_raw, prob)
            return i + 1 if len(pieces) == 5 else i + 2
        elif len(pieces == 3):
            # case 2: R: <action> : <start-state> : <next-state>
            # %f %f ... %f
            start_state = self.states.index(pieces[1])
            next_state = self.states.index(pieces[2])
            next_line = self.contents[i+1]
            probs = next_line.split()
            assert len(probs) == len(self.observations)
            for j in range(len(probs)):
                prob = float(probs[j])
                self.R[(action, start_state, next_state, j)] = prob
            return i + 2
        elif len(pieces == 2):
            # case 3: R: <action> : <start-state>
            # %f %f ... %f
            # %f %f ... %f
            # ...
            # %f %f ... %f
            start_state = self.states.index(pieces[1])
            next_line = self.contents[i+1]
            for j in range(len(self.states)):
                probs = next_line.split()
                assert len(probs) == len(self.observations)
                for k in range(len(probs)):
                    prob = float(probs[k])
                    self.R[(action, start_state, j, k)] = prob
                next_line = self.contents[i+2+j]
            return i + 1 + len(self.states)
        else:
            raise Exception("Cannot parse line: " + line)

        

    def __reward_ss(self, a, start_state_raw, next_state_raw, obs_raw, prob):
        """
        reward_ss means we're at the start state of the unrolling of the
        reward expression. start_state_raw could be * or the name of the
        real start state.
        """
        if start_state_raw == '*':
            for i in range(len(self.states)):
                self.__reward_ns(a, i, next_state_raw, obs_raw, prob)
        else:
            start_state = self.states.index(start_state_raw)
            self.__reward_ns(a, start_state, next_state_raw, obs_raw, prob)

    def __reward_ns(self, a, start_state, next_state_raw, obs_raw, prob):
        """
        reward_ns means we're at the next state of the unrolling of the
        reward expression. start_state is the number of the real start
        state, and next_state_raw could be * or the name of the real
        next state.
        """
        if next_state_raw == '*':
            for i in range(len(self.states)):
                self.__reward_ob(a, start_state, i, obs_raw, prob)
        else:
            try:
                next_state = self.states.index(next_state_raw)
            except:
                next_state = self.states.index(int(next_state_raw))
            self.__reward_ob(a, start_state, next_state, obs_raw, prob)

    def __reward_ob(self, a, start_state, next_state, obs_raw, prob):
        """
        reward_ob means we're at the observation of the unrolling of the
        reward expression. start_state is the number of the real start
        state, next_state is the number of the real next state, and
        obs_raw could be * or the name of the real observation.
        """
        if obs_raw == '*':
            for i in range(len(self.observations)):
                self.R[(a, start_state, next_state, i)] = prob
        else:
            try:
                obs = self.observations.index(obs_raw)
            except:
                obs = self.observations.index(int(obs_raw))
            self.R[(a, start_state, next_state, obs)] = prob

    def update_belief(self, prev_belief, action_num, observation_num):
        """
        Note that a POMDPEnvironment doesn't hold beliefs, so this takes
        and returns a belief vector.
        prev_belief     numpy array
        action_num      int
        observation_num int
        return          numpy array
        """
        b_new_nonnormalized = []
        for s_prime in range(len(self.states)):
            p_o_prime = self.Z[(action_num, s_prime, observation_num)]
            summation = 0.0
            for s in range(len(self.states)):
                p_s_prime = self.T[(action_num, s, s_prime)]
                b_s = float(prev_belief[s])
                summation = summation + p_s_prime * b_s
            b_new_nonnormalized.append(p_o_prime * summation)

        # normalize
        b_new = []
        total = sum(b_new_nonnormalized)
        for b_s in b_new_nonnormalized:
            b_new.append([b_s/total])
        return np.array(b_new)

    def print_summary(self):
        print("discount:", self.discount)
        print("values:", self.values)
        print("states:", self.states)
        print("actions:", self.actions)
        print("observations:", self.observations)
        print("")
        print("T:", self.T)
        print("")
        print("Z:", self.Z)
        print("")
        print("R:", self.R)
