from brain import *
from actor import *
from baselines import *
import matplotlib.pyplot as plt


class ActorTrainer(BrainTrainer):

    def __init__(self):
        super(ActorTrainer, self).__init__()
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


class EntropyTrainer(BrainTrainer):

    def __init__(self):
        super(EntropyTrainer, self).__init__()
        self.chkpt_name = 'models/brain_entropy'
        self.player = MaxEntropyPlayer()
        self.brain = Brain()
        self.player.brain = self.brain

    def play_game(self):
        return self.player.play_game_trainer()


class PipelineTrainer(BrainTrainer):

    def __init__(self):
        super(PipelineTrainer, self).__init__()
        self.chkpt_name = 'models/brain_pipelined'
        self.policy = Policy(pipeline=True)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)
        self.env = GameEnv()
        self.i_episode = 0
        self.running_reward = 0
        self.running_reward_list = []

    def play_game(self):
        self.i_episode += 1
        self.env.reset()
        truth = self.env.game.get_brain_truth()
        total_reward = 0
        made_actions = []
        states = []
        feedbacks = []
        for t in range(100):  # Don't infinite loop while learning
            # invalid_moves = self.env.get_invalid_moves()

            state = np.array(self.model.get_probabilities_total(self.env.game.get_brain_state(), self.env.game.known))

            action = self.policy.select_action(state)#, invalid_moves+made_actions)
            made_actions.append(action)

            state, reward, done = self.env.step(action)

            total_reward += reward

            if reward <= 0:
                reward = -1

            if done and not self.env.game.has_won():
                reward = -100

            self.policy.rewards.append(reward)
            states.append(self.env.game.get_brain_state())
            feedbacks.append(truth)
            if done:
                # made_actions = []
                break
        if self.i_episode == 0:
            self.running_reward = running_reward
        else:
            self.running_reward = self.running_reward * 0.99 + total_reward * 0.01

        self.running_reward_list.append(self.running_reward)
        finish_episode(self.policy, self.optimizer)
        if self.i_episode % LOG_INTERVAL == 0:
            print('Episode {}\tLast length/reward: {:5d}/{}\tAverage reward: {:.2f}'.format(
                self.i_episode, t+1, total_reward, self.running_reward))
        return states, feedbacks

    def run(self):
        super(PipelineTrainer, self).run()
        self.policy.save('models/actor_pipelined2')
        plt.plot(self.running_reward_list)
        plt.show()

if __name__ == '__main__':
    trainer = PipelineTrainer()
    trainer.run()