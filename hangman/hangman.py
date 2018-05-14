import random
import numpy as np


# first load words
words = []
with open('words.txt') as f:
    for word in f:
        # skip contractions
        if '\'' in word or len(word.strip()) != 10: #or len(word.strip()) < 3 or len(word.strip()) > 10:
            continue
        words.append(word.strip())

def word_to_int(word):
    # TODO find built-in to do this
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    int_word = []

    for c in word:
        int_word.append(letters.index(c))

    return int_word

def int_to_word(ints):
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    return [letters[i] for i in ints]


MAX_WORD_LENGTH = max([len(w) for w in words])


class HangmanEnv(object):

    def __init__(self):
        # TODO decide if death condition for hangman
        # max length word is of length 28
        self.reset()

    def reset(self):
        self.word_raw = random.choice(words)
        self.word = word_to_int(self.word_raw)
        self.num_letters = len(self.word)
        self.guessed_letters = set()
        self.num_actions = 26
        self.placeholder = [-1 for i in range(len(self.word))]
        state = self.placeholder
        return state

    def step(self, action):
        found_character = False
        reward = 0

        if action not in self.guessed_letters:
            self.guessed_letters.add(action)
            for i, c in enumerate(self.word):
                if c == action:
                    self.placeholder[i] = c
                    found_character = True
                    reward += 1

        # reward = 1 if found_character else 0
        if reward == 0:
            reward = -0.1
        done = -1 not in self.placeholder
        state = self.placeholder

        return state, reward, done

    def win(self):
        return -1 not in self.placeholder

    def forbid(self):
        return self.guessed_letters


class StateXform:
  def __init__(self):
    self.length = MAX_WORD_LENGTH*27

  def state_to_np(self, state):
    vec = []
    for i in range(MAX_WORD_LENGTH):
        char_vec = [0 for i in range(27)]
        if i < len(state):
            char = state[i]
            char_vec[char+1] = 1
        vec += char_vec
    return np.array(vec)

# class StateXformTruth:
#   def __init__(self):
#     self.length = BRAIN_INPUT_LENGTH + OUTPUT_LENGTH

#   def state_to_np(self, state):
#     board_mask, board_truth, _, _ = state
#     ret =  np.concatenate((board_mask, board_truth))
#     return ret

# class OracleXform:
#     def __init__(self, oracle):
#         self.length = OUTPUT_LENGTH + OUTPUT_LENGTH
#         self.oracle = oracle

#     def state_to_np(self, state):
#         board_mask, _, _, actions = state
#         actions_vec = [1 if i in actions else 0 for i in range(OUTPUT_LENGTH)]
#         oracle_prediction = self.oracle.predict(state)
#         ret =  np.concatenate((actions_vec, oracle_prediction))

#         return ret

# class FutureXform:
#     def __init__(self):
#         self.length = OUTPUT_LENGTH

#     def state_to_np(self, state):
#         _, _, mask_and_truth, _ = state

#         return mask_and_truth

class ActionXform:
  def __init__(self):
    self.possible_actions = list(range(26))
    self.length = 26
  def idx_to_action(self, idx):
    return self.possible_actions[idx]
  def action_to_idx(self, a):
    return a
  def action_to_1hot(self, a):
    ret = np.zeros(self.length)
    ret[a] = 1.0
    return ret


if __name__ == '__main__':
    # words2 = [len(w) for w in words]
    # print(np.argmax(words2))
    # print(words[np.argmax(words2)])

    # word_dict = {}

    # for w in words:
    #     if len(w) in word_dict:
    #         word_dict[len(w)] += 1
    #     else:
    #         word_dict[len(w)] = 1
    # print(len(words))
    # print(sorted(word_dict.items()))
    vec = [14, 0, 8, 13, 18, 20, 4, 11, 19, 7, 5, 17, 24, 22, 21, 15, 9, 12, 16, 2, 23]
    print(int_to_word(vec))