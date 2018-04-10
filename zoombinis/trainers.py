from brain import *
from actor import *


class BrainTrainer2(BrainTrainer):

    def __init__(self):
        super(BrainTrainer2, self).__init__()
        self.policy = Policy()
        self.policy.load()
        self.env = GameEnv()

    def play_game(self):
        state = self.env.reset()
        sent_indices = set()
        states = []
        feedbacks = []

        while self.env.game.can_move():
            cant_use = self.env.get_invalid_moves()
            action = self.policy.select_action2(state, cant_use)

            state, reward, done = self.env.step(action)

            brain_state = self.env.game.get_brain_state()

            states.append(brain_state)
            feedbacks.append(self.env.game.get_brain_truth())

        return states, feedbacks

if __name__ == '__main__':
    trainer = BrainTrainer2()
    trainer.run()