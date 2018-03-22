from zoombinis import *
from brain import *
import numpy as np


class EntropyPlayer(object):

    def __init__(self, prob_value=1.):
        self.brain = ZBrain(chkpt='models/brain')
        self.prob_value = prob_value


    def play_game(self, game):
        indices_remaining = [i for i in range(NUM_ZOOMBINIS)]
        tried = {} # maps indices to previously tried boolean values

        while game.can_move():
            best_entropy = -1
            best_entropy_i = -1

            entropies = []

            best_prob = -1
            best_prob_i = -1


            for i in indices_remaining:
                state = game.get_state_vector(i)
                # state = np.array(state).reshape(-1, INPUT_LENGTH)

                entropy = self.brain.get_entropy(state)

                if entropy > best_entropy:
                    best_entropy = entropy
                    best_entropy_i = i

                entropies.append(entropy)

                probabilities = self.brain.get_probabilities(state)
                prob = max(probabilities)

                if prob > best_prob:
                    best_prob = prob
                    best_prob_i = i

            # make "best move"
            if best_prob > self.prob_value:
                index = best_prob_i
            else:
                index = best_entropy_i

            state = game.get_state_vector(i)
            # state = np.array(state).reshape(-1, INPUT_LENGTH)
            probabilities = self.brain.get_probabilities(state)
            send_top = True if np.argmax(probabilities) == 0 else False

            if index in tried:
                send_top = not tried[index]

            # make sure no duplicate actions
            tried[index] = send_top

            feedback = game.send_zoombini(index, send_top)
            # print(state)
            # print(best_entropy, best_entropy_i, probabilities, index, indices_remaining, best_prob, best_prob_i)
            # print(feedback)
            # print(best_entropy, best_entropy-1.*sum(entropies)/len(entropies), np.var(entropies))

            if feedback:
                indices_remaining.remove(index)

        # print 'Won game:', game.has_won()
        # print 'Score:', 16-len(indices_remaining)
        return game.has_won()

if __name__ == '__main__':
    import tqdm
    player = EntropyPlayer()

    wins = 0
    num_games = 1000
    for i in tqdm.tqdm(range(num_games)):
        if player.play_game(Game()):
            wins += 1
    print('win rate:', 1.*wins/num_games)



