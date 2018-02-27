import tensorflow as tf
import os
import numpy as np
import random
import time
import tqdm
from threading import Thread
from copy import copy
from baselines import MaxEntropyPlayer
from encoder_nn import EncoderModel
from mastermind import ENCODER_VECTOR_LENGTH, Mastermind, NUM_PEGS, NUM_OPTIONS, EMBEDDED_LENGTH, random_guess, guess_to_vector, ALL_GUESSES


OPTIONS_LENGTH = NUM_PEGS*NUM_OPTIONS
PERMITTED_ACTIONS = 10

def action_from_vector(choices_vector):
    choices = []

    for vec in choices_vector:
        vec = vec[0]
        # need to stabilize because might be slightly off of one
        vec = vec/np.sum(vec)

        # choose probabilistically
        # choices.append(np.random.choice(NUM_OPTIONS, p=vec))
        choices.append(np.argmax(vec))
    return choices

# supress warnings...
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

class DaggerModel(object):

    def __init__(self, chkpt=None):
        # load from chkpt if exists
        self.create_graph()
        if chkpt:
            self.load(chkpt)

    def create_graph(self):
        self.session = tf.Session()

        # up to PERMITTED_ACTIONS embedded inputs
        self.input_layer = tf.placeholder(tf.float32, [None, EMBEDDED_LENGTH*PERMITTED_ACTIONS])
        self.teacher_layer = tf.placeholder(tf.float32, [None, OPTIONS_LENGTH])


        layer1 = tf.layers.dense(self.input_layer, 1000, activation=tf.nn.relu)
        
        # actor layers
        self.pre_out_layer = tf.layers.dense(layer1, OPTIONS_LENGTH)
        self.split_out_layers = tf.split(self.pre_out_layer, NUM_PEGS, 1)
        self.out_layers = [tf.nn.softmax(layer) for layer in self.split_out_layers]
        self.actor_layer = tf.concat(self.out_layers, 1)

        # loss
        self.split_teacher = tf.split(self.teacher_layer, NUM_PEGS, 1)

        self.loss = tf.reduce_sum([tf.reduce_mean(
            tf.squared_difference(self.split_teacher[i], self.out_layers[i]))
            for i in xrange(len(self.out_layers))
        ])
        self.optimizer = tf.train.AdamOptimizer(1e-5)
        self.min_loss = self.optimizer.minimize(self.loss)

        initializer = tf.global_variables_initializer()
        self.session.run(initializer)

        self.saver = tf.train.Saver()

    def get_choice_vectors(self, embedding):
        return self.session.run([self.out_layers], feed_dict={
            self.input_layer: embedding
        })[0]

    def get_action_vector(self, embedding):
        return self.session.run([self.actor_layer], feed_dict={
            self.input_layer: embedding
        })

    def get_action(self, embedding):
        choice = self.get_choice_vectors(embedding)
        # print choice
        return action_from_vector(choice)

    def train(self, state, opt_actions):
        loss, _ = self.session.run([self.loss, self.min_loss], feed_dict={
            self.input_layer: state,
            self.teacher_layer: opt_actions,
        })
        print('Loss: {}'.format(loss))

    def load(self, name='models/dagger'):
        self.saver.restore(self.session, name)

    def save(self, name='models/dagger'):
        self.saver.save(self.session, name)
        print 'Saved model to {}!'.format(name)


class Environment(object):

    def __init__(self):
        self.game_class = Mastermind
        chkpt = 'models/encoder_model'
        self.encoder = EncoderModel(chkpt=chkpt)

    def run_episode(self, agent):

        game = self.game_class()
        # print 'TARGET:', game.target
        teacher = MaxEntropyPlayer()

        num_actions = 0
        won = False

        current_state = np.zeros((1, PERMITTED_ACTIONS*EMBEDDED_LENGTH))
        states = [current_state]
        opt_actions = []

        while not won:
            action = agent.get_action(current_state)
            feedback = game.guess(action)

            # handle teacher
            opt_action = teacher.make_guess()
            teacher.add_feedback(action, feedback)
            # print 'opt action:', opt_action, 'chosen action:', action, 'feedback:', feedback

            won = game.is_winning_feedback(feedback)
            opt_action_vec = np.array(guess_to_vector(opt_action))
            opt_actions.append(opt_action_vec)

            if len(states) > PERMITTED_ACTIONS or won:
                break
            # print states.shape, actions.shape, rewards.shape

            additional_state = self.encoder.get_embeddings(action, feedback).reshape(EMBEDDED_LENGTH, )
            current_state[:,(len(states)-1)*EMBEDDED_LENGTH:len(states)*EMBEDDED_LENGTH] = additional_state
            states.append(current_state)


        if len(states) != len(opt_actions):
            states.pop()

        states = np.vstack(states)
        actions = np.vstack(opt_actions)

        # print states.shape, actions.shape, rewards.shape

        return states, opt_actions, won


class DaggerRunner(object):

    def __init__(self, chkpt=None, new_chkpt='models/dagger'):
        self.env = Environment()
        self.model = DaggerModel(chkpt=chkpt)
        self.new_chkpt = new_chkpt
        self.dataset_states = None
        self.dataset_actions = None

    def run(self, num_episode=1000):
        best_eval = 0.5 # start with a baseline
        for i in xrange(num_episode):
            print 'Game:', i+1
            states, actions, won = self.env.run_episode(self.model)
            if self.dataset_states is None:
                self.dataset_states = states
                self.dataset_actions = actions
            else:
                self.dataset_states = np.vstack([self.dataset_states, states])
                self.dataset_actions = np.vstack([self.dataset_actions, actions])

            self.model.train(self.dataset_states, self.dataset_actions)
            # print '{} Number of actions: {}'.format(won, len(actions))

            # eval_value = check(None, model=self.model, encoder=self.env.encoder)
            # print 'Success Rate:', eval_value

            # if eval_value > best_eval:
            #     self.model.save(name='models/dagger{}'.format(eval_value))
            #     best_eval = eval_value

        self.model.save(name=self.new_chkpt)


def check(chkpt, model=None, encoder=None, show_results=False):
    if model is None:
        tf.reset_default_graph()
        encoder = EncoderModel(chkpt='models/encoder_model')
        agent = DaggerModel(chkpt=chkpt)
    else:
        encoder = encoder
        agent = model

    num_moves = []
    num_moves_success = []
    successes = 0
    num_trials = 1
    num_games = len(ALL_GUESSES)*num_trials

    for trial in tqdm.tqdm(xrange(num_trials)):
        for i, target in enumerate(ALL_GUESSES):
            # play a random game
            game = Mastermind(target=target)
            won = False
            current_state = np.zeros((1, PERMITTED_ACTIONS*EMBEDDED_LENGTH))
            num_actions = 0

            while not won:
                action = agent.get_action(current_state)
                num_actions += 1
                feedback = game.guess(action)
                won = game.is_winning_feedback(feedback)
                additional_state = encoder.get_embeddings(action, feedback).reshape(EMBEDDED_LENGTH, )
                current_state[:,(num_actions-1)*EMBEDDED_LENGTH:num_actions*EMBEDDED_LENGTH] = additional_state
                if num_actions >= PERMITTED_ACTIONS:
                    break

            num_moves.append(num_actions)
            if won:
                successes += 1
                num_moves_success.append(num_actions)

    if show_results:
        print 'Success Rate: {}'.format(1.*successes/num_games)
        print 'Avg Moves: {}'.format(1.*sum(num_moves)/num_games)
        print 'Avg Moves when win: {}'.format(1.*sum(num_moves_success)/len(num_moves_success))

    return 1.*successes/num_games




def investigate_moves(chkpt):
    tf.reset_default_graph()
    encoder = EncoderModel(chkpt='models/encoder_model')
    agent = DaggerModel(chkpt=chkpt)

    game = Mastermind()
    print 'target:', game.target

    won = False
    current_state = np.zeros((1, PERMITTED_ACTIONS*EMBEDDED_LENGTH))
    num_actions = 0

    while not won:
        print agent.get_choice_vectors(current_state)
        action = agent.get_action(current_state)
        num_actions += 1
        feedback = game.guess(action)
        print action, feedback
        won = game.is_winning_feedback(feedback)
        additional_state = encoder.get_embeddings(action, feedback).reshape(EMBEDDED_LENGTH, )
        current_state[:,(num_actions-1)*EMBEDDED_LENGTH:num_actions*EMBEDDED_LENGTH] = additional_state
        if num_actions >= 10:
            break


def learn_anything():
    tf.reset_default_graph()
    encoder = EncoderModel(chkpt='models/encoder_model')
    agent = DaggerModel(chkpt=None)

    start_state = np.zeros((1, PERMITTED_ACTIONS*EMBEDDED_LENGTH))
    actions = np.vstack([guess_to_vector([0, 0, 1, 1])])

    for i in tqdm.tqdm(xrange(100)):
        agent.train(start_state, actions)

    print agent.get_choice_vectors(start_state)


if __name__ == '__main__':
    chkpt = 'models/dagger2'
    new_chkpt = 'models/daggerarg2'
    runner = DaggerRunner(chkpt=new_chkpt, new_chkpt=new_chkpt)
    runner.run()
    check(new_chkpt, show_results=True)

    investigate_moves(new_chkpt)
    # print guess_to_vector([0, 0, 1, 1])
