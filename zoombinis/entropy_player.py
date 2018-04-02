from zoombinis import *
from brain import *
import numpy as np


class EntropyPlayer(object):

    def __init__(self, prob_value=1.0):
        self.brain = Brain(chkpt='models/brain')
        self.prob_value = prob_value

        # brain that starts with random to just compare with random
        # self.brain = Brain()


    def play_game(self, game):
        indices_remaining = [i for i in range(NUM_ZOOMBINIS)]
        tried = {} # maps indices to previously tried boolean values

        while game.can_move():
            best_entropy = -1
            best_entropy_i = -1
            best_prob = -1
            best_prob_i = -1


            state = game.get_brain_state()
            entropies = self.brain.get_entropies(state)
            probs = self.brain.get_probabilities(state)

            for i in indices_remaining:
                
                entropy = entropies[i]

                if entropy > best_entropy:
                    best_entropy = entropy
                    best_entropy_i = i

                entropies.append(entropy)

                my_probs = probs[i]
                prob = max(my_probs)

                if prob > best_prob:
                    best_prob = prob
                    best_prob_i = i

            # make "best move"
            if best_prob > self.prob_value:
                index = best_prob_i
            else:
                index = best_entropy_i

            probabilities = probs[index]
            send_top = np.argmax(probabilities)

            if index in tried:
                send_top = not tried[index]

            # make sure no duplicate actions
            tried[index] = send_top

            feedback = game.send_zoombini(index, send_top)

            if feedback:
                indices_remaining.remove(index)

        # print('Won game:', game.has_won())
        # print('Score:', 16-len(indices_remaining))
        return game.has_won(), NUM_ZOOMBINIS-len(indices_remaining)

if __name__ == '__main__':
    import tqdm
    player = EntropyPlayer()

    wins = 0
    scores = []
    num_games = 100
    for i in tqdm.tqdm(range(num_games)):
        won, score = player.play_game(Game())
        if won:
            wins += 1
        scores.append(score)
    print('win rate:', wins/num_games)
    print('avg score:', sum(scores)/len(scores))



