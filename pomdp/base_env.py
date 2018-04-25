import random


class BaseEnv(object):
    # based on http://cs.brown.edu/research/ai/pomdp/examples/cheese.95.POMDP

    def __init__(self):
        # need custom matrices for children envs:
        #      transitions, observations, rewards
        pass

    def _select_from_matrix(self, mat, action):
        result = mat[self.state][action]

        if type(result) != int:
            keys = []
            probs = []
            for k, prob in result.items():
                if prob > 0:
                    keys.append(k)
                    probs.append(prob)

            result = random.choices(keys, weights=probs)[0]
        return result

    def _calc_reward(self):
        result = self.rewards[self.state]

        if type(result) != int:
            keys = []
            probs = []
            for k, prob in result.items():
                if prob > 0:
                    keys.append(k)
                    probs.append(prob)

            result = random.choices(keys, weights=probs)[0]
        return result

    def won(self):
        # need a won function
        pass

    def reset(self):
        # need to define this
        pass

    def step(self, action):
        # need to define this
        pass