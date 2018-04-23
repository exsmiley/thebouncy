from zoombinis import *
from brain import *
import numpy as np
import random


class MaxEntropyPlayer(object):

    def __init__(self, prob_value=1.0):
        self.brain = Brain(chkpt='models/brain')
        self.prob_value = prob_value

        # brain that starts with random to just compare with random
        # self.brain = Brain()

    def brain_test(self, game, brains):
        scores_dict = {brain: [] for brain in brains.keys()}
        moves = 0
        truth = game.get_brain_truth()
        indices_remaining = [i for i in range(NUM_BRIDGES*NUM_ZOOMBINIS)]
        tried = {} # maps indices to previously tried boolean values

        while game.can_move():
            moves += 1
            best_entropy = -1
            best_entropy_i = -1
            best_prob = -1
            best_prob_i = -1

            state = game.get_brain_state()
            entropies = self.brain.get_entropies(state)
            probs = self.brain.get_probabilities(state)

            for brain, scores in scores_dict.items():
                brain = brains[brain]
                probs = brain.get_probabilities(state)
                score = 0

                for i in range(0, len(truth), NUM_BRIDGES):
                    truths = truth[i:i+NUM_BRIDGES]
                    preds = probs[i:i+NUM_BRIDGES]
                    if np.argmax(truths) == np.argmax(preds):
                        score += 1
                scores.append(score)


            for i in indices_remaining:
                
                entropy = entropies[i]
                prob = probs[i]

                if entropy > best_entropy:
                    best_entropy_i = i
                    best_entropy = entropy

                if prob > best_prob:
                    best_prob_i = i
                    best_prob = prob


            if best_prob > self.prob_value:
                index = best_prob_i
            else:
                index = best_entropy_i

            indices_remaining.remove(index)

            zoombini = index//NUM_BRIDGES
            bridge = index % NUM_BRIDGES

            feedback = game.send_zoombini(zoombini, bridge)

        return scores_dict

    def play_game_trainer(self):
        game = Game()
        truth = game.get_brain_truth()
        indices_remaining = [i for i in range(NUM_BRIDGES*NUM_ZOOMBINIS)]

        states = []
        feedbacks = []

        while game.can_move():
            best_entropy = -1
            best_entropy_i = -1

            state = game.get_brain_state()
            entropies = self.brain.get_entropies(state)

            for i in indices_remaining:
                entropy = entropies[i]

                if entropy > best_entropy:
                    best_entropy_i = i
                    best_entropy = entropy

            index = best_entropy_i
            indices_remaining.remove(index)

            zoombini = index//NUM_BRIDGES
            bridge = index % NUM_BRIDGES

            game.send_zoombini(zoombini, bridge)
            

            brain_state = game.get_brain_state()

            states.append(brain_state)
            feedbacks.append(truth)

        return states, feedbacks

    def play_game_trainer_mask(self):
        game = Game()
        indices_remaining = [i for i in range(NUM_BRIDGES*NUM_ZOOMBINIS)]

        states = []

        while game.can_move():
            best_entropy = -1
            best_entropy_i = -1

            state = game.get_brain_state()
            entropies = self.brain.get_entropies(state)

            for i in indices_remaining:
                entropy = entropies[i]

                if entropy > best_entropy:
                    best_entropy_i = i
                    best_entropy = entropy

            index = best_entropy_i
            indices_remaining.remove(index)

            zoombini = index//NUM_BRIDGES
            bridge = index % NUM_BRIDGES

            game.send_zoombini(zoombini, bridge)

            brain_state = game.get_brain_state()

            states.append(brain_state)

        return states, game


    def play(self, game):
        actual_score = 0
        running_scores = []
        scores = []
        moves = 0
        truth = game.get_brain_truth()
        indices_remaining = [i for i in range(NUM_BRIDGES*NUM_ZOOMBINIS)]
        tried = {} # maps indices to previously tried boolean values

        while game.can_move():
            moves += 1
            best_entropy = -1
            best_entropy_i = -1
            best_prob = -1
            best_prob_i = -1

            state = game.get_brain_state()
            entropies = self.brain.get_entropies(state)
            probs = self.brain.get_probabilities(state)
            score = 0

            for i in range(0, len(truth), NUM_BRIDGES):
                truths = truth[i:i+NUM_BRIDGES]
                preds = probs[i:i+NUM_BRIDGES]
                if np.argmax(truths) == np.argmax(preds):
                    score += 1
            scores.append(score)


            for i in indices_remaining:
                
                entropy = entropies[i]
                prob = probs[i]

                if entropy > best_entropy:
                    best_entropy_i = i
                    best_entropy = entropy

                if prob > best_prob:
                    best_prob_i = i
                    best_prob = prob


            if best_prob > self.prob_value:
                index = best_prob_i
            else:
                index = best_entropy_i

            # index = best_entropy_i
            indices_remaining.remove(index)

            zoombini = index//NUM_BRIDGES
            bridge = index % NUM_BRIDGES

            feedback = game.send_zoombini(zoombini, bridge)

            # print('Sending {} to {} and got {}\n'.format(zoombini, bridge, feedback))

            if feedback:
                actual_score += 1
            running_scores.append(actual_score)

        return game.has_won(), scores, actual_score, running_scores

class MaxProbabilityPlayer(MaxEntropyPlayer):

    def __init__(self):
        super(MaxProbabilityPlayer, self).__init__(prob_value=0.5)


class RandomPlayer():

    def __init__(self, prob_value=1.0):
        self.brain = Brain(chkpt='models/brain')
        self.prob_value = prob_value

    def play(self, game):
        possible_moves = set([i for i in range(NUM_ZOOMBINIS*2)])
        next_move = None
        scores = []
        running_scores = []
        moves = 0
        actual_score = 0
        truth = game.get_brain_truth()

        while game.can_move():
            moves += 1
            for invalid in game.get_invalid_moves():
                if invalid in possible_moves:
                    possible_moves.remove(invalid)

            state = game.get_brain_state()
            probs = self.brain.get_probabilities(state)
            score = 0

            for i in range(0, len(truth), NUM_BRIDGES):
                truths = truth[i:i+NUM_BRIDGES]
                preds = probs[i:i+NUM_BRIDGES]
                if np.argmax(truths) == np.argmax(preds):
                    score += 1

            scores.append(score)


            action = random.choice(list(possible_moves))

            zoombini = action//NUM_BRIDGES
            bridge = action % NUM_BRIDGES

            result = game.send_zoombini(zoombini, bridge)

            if result:
                actual_score += 1

            running_scores.append(actual_score)
        return game.has_won(), scores, actual_score, running_scores


class RandomFlipFlop():

    def __init__(self, prob_value=1.0):
        self.brain = Brain(chkpt='models/brain')
        self.prob_value = prob_value

    def play(self, game):

        possible_moves = set([i for i in range(NUM_ZOOMBINIS*2)])
        next_move = None
        scores = []
        running_scores = []
        moves = 0
        actual_score = 0
        truth = game.get_brain_truth()

        while game.can_move():
            moves += 1
            for invalid in game.get_invalid_moves():
                if invalid in possible_moves:
                    possible_moves.remove(invalid)

            if next_move:
                action = next_move
                next_move = None
            else:
                action = random.choice(list(possible_moves))

            zoombini = action//NUM_BRIDGES
            bridge = action % NUM_BRIDGES

            state = game.get_brain_state()
            probs = self.brain.get_probabilities(state)
            score = 0

            for i in range(0, len(truth), NUM_BRIDGES):
                truths = truth[i:i+NUM_BRIDGES]
                preds = probs[i:i+NUM_BRIDGES]
                if np.argmax(truths) == np.argmax(preds):
                    score += 1
            scores.append(score)


            result = game.send_zoombini(zoombini, bridge)

            if not result and action % 2 == 0:
                next_move = action + 1
            elif not result:
                next_move = action - 1
            else:
                actual_score += 1
            running_scores.append(actual_score)

        return game.has_won(), scores, actual_score, running_scores



class WinningBridgePlayer():

    def __init__(self):
        self.brain = Brain(chkpt='models/brain')

    def play(self, game):
        zoombinis = set([i for i in range(NUM_ZOOMBINIS)])
        next_move = None
        scores = []
        running_scores = []
        moves = 0
        actual_score = 0
        truth = game.get_brain_truth()

        old_zoombini = None
        next_move = None
        top_count = 1
        bottom_count = 1

        while game.can_move():
            moves += 1

            state = game.get_brain_state()
            probs = self.brain.get_probabilities(state)
            score = 0

            for i in range(0, len(truth), NUM_BRIDGES):
                truths = truth[i:i+NUM_BRIDGES]
                preds = probs[i:i+NUM_BRIDGES]
                if np.argmax(truths) == np.argmax(preds):
                    score += 1

            scores.append(score)

            if next_move is not None:
                zoombini = old_zoombini
                action = next_move
                old_zoombini = None
                next_move = None
            else:
                together_count = top_count + bottom_count
                zoombini = random.choice(list(zoombinis))
                action = np.random.choice([0, 1], p=[top_count/together_count, bottom_count/together_count])
                zoombinis.remove(zoombini)

            result = game.send_zoombini(zoombini, action)

            if result:
                actual_score += 1
                if action == 0:
                    top_count += 1
                else:
                    bottom_count += 1
            else:
                old_zoombini = zoombini
                next_move = (action + 1) % 2

            running_scores.append(actual_score)
        return game.has_won(), scores, actual_score, running_scores

class WinningBridgePlayerNoHack():

    def __init__(self):
        self.brain = Brain(chkpt='models/brain')

    def play(self, game):
        zoombinis = set([i for i in range(NUM_ZOOMBINIS)])
        moved_zoombinis = {}
        next_move = None
        scores = []
        running_scores = []
        moves = 0
        actual_score = 0
        truth = game.get_brain_truth()

        old_zoombini = None
        next_move = None
        top_count = 1
        bottom_count = 1

        while game.can_move():
            moves += 1

            state = game.get_brain_state()
            probs = self.brain.get_probabilities(state)
            score = 0

            for i in range(0, len(truth), NUM_BRIDGES):
                truths = truth[i:i+NUM_BRIDGES]
                preds = probs[i:i+NUM_BRIDGES]
                if np.argmax(truths) == np.argmax(preds):
                    score += 1

            scores.append(score)

            together_count = top_count + bottom_count
            zoombini = random.choice(list(zoombinis))

            if zoombini in moved_zoombinis:
                action = moved_zoombinis[zoombini]
            else:
                action = np.random.choice([0, 1], p=[top_count/together_count, bottom_count/together_count])

            result = game.send_zoombini(zoombini, action)

            if result:
                actual_score += 1
                zoombinis.remove(zoombini)
                if action == 0:
                    top_count += 1
                else:
                    bottom_count += 1
            else:
                moved_zoombinis[zoombini] = (action + 1) % 2

            running_scores.append(actual_score)
        return game.has_won(), scores, actual_score, running_scores


if __name__ == '__main__':
    import tqdm
    player = WinningBridgePlayer()

    wins = 0
    scores = []
    num_games = 100
    for i in tqdm.tqdm(range(num_games)):
        won, es, score, running_scores = player.play(Game())
        if won:
            wins += 1
        scores.append(score)
    print('win rate:', wins/num_games)
    # print(scores)
    print('avg score:', sum(scores)/len(scores))



