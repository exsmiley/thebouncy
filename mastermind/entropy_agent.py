from brain import BrainModel, OPTIONS_LENGTH
from encoder_nn import EncoderModel
from player import Player
from solver import MastermindSolver
from mastermind import *


class SmartEntropyAgent(Player):

    def __init__(self):
        super(SmartEntropyAgent, self).__init__()
        self.encoder = EncoderModel() # don't need a checkpoint since not encoded
        self.brain = BrainModel(chkpt='models/brain')
        self.solver = MastermindSolver()
        self.all_possible_moves = list(generate_all_targets(self.num_pegs, self.num_options))

    def make_guess(self):
        if len(self.attempts) > 3:
            is_unique, soln = self.solver.is_unique_solution()
            if is_unique:
                # print "solver solved...", soln
                return soln

        top_entropy = 0
        top_move = None
        state = self.encoder.create_current_state(self.attempts)
        for move in self.all_possible_moves:
            action_vec = np.array(guess_to_vector(move)).reshape(-1, OPTIONS_LENGTH)
            entropy = self.brain.get_entropy(state, action_vec)
            if entropy > top_entropy:
                top_move = move
                top_entropy = entropy

        # print top_entropy, top_move
        return top_move

    def add_feedback(self, guess, feedback):
        # overwrite super since store slightly differently
        self.used.add(tuple(guess))
        self.attempts.append((guess, feedback))
        self.solver.add_feedback((guess, feedback))

    def reset(self):
        super(SmartEntropyAgent, self).reset()
        self.solver.reset()

if __name__ == '__main__':
    from player import PlayerRunner

    game = Mastermind()
    print 'Target:', game.target
    runner = PlayerRunner(SmartEntropyAgent())

    won, actions = runner.play(game, loss_threshold=10)
    if won:
        print 'Won in {} moves!'.format(len(actions))
    else:
        print 'Lost :('

