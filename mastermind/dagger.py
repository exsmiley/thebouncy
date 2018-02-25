import tensorflow as tf
import os
import numpy as np
import random
import time
from threading import Thread
from copy import copy
from baselines import MaxEntropyPlayer
from encoder_nn import EncoderModel
from mastermind import ENCODER_VECTOR_LENGTH, Mastermind, NUM_PEGS, NUM_OPTIONS, EMBEDDED_LENGTH, random_guess, guess_to_vector


OPTIONS_LENGTH = NUM_PEGS*NUM_OPTIONS

def action_from_vector(choices_vector):
    choices = []

    for vec in choices_vector:
        vec = vec[0]
        # need to stabilize because might be slightly off of one
        vec = vec/np.sum(vec)

        # choose probabilistically
        choices.append(np.argmax(vec))
        # choices.append(np.argmax(vec))
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

        # up to 10 embedded inputs
        self.input_layer = tf.placeholder(tf.float32, [None, EMBEDDED_LENGTH*10])
        self.teacher_layer = tf.placeholder(tf.float32, [None, OPTIONS_LENGTH])

        layer1 = tf.layers.dense(self.input_layer, 1000, activation=tf.nn.relu)
        
        # actor layers
        self.pre_out_layer = tf.layers.dense(layer1, OPTIONS_LENGTH)
        self.split_out_layers = tf.split(self.pre_out_layer, NUM_PEGS, 1)
        self.out_layers = [tf.nn.softmax(layer) for layer in self.split_out_layers]
        self.actor_layer = tf.concat(self.out_layers, 1)

        # loss
        self.loss = tf.reduce_mean(tf.squared_difference(self.teacher_layer, self.actor_layer))
        self.optimizer = tf.train.AdamOptimizer(1e-3)
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
        print 'TARGET:', game.target
        teacher = MaxEntropyPlayer()

        num_actions = 0
        won = False

        current_state = np.zeros((1, 10*EMBEDDED_LENGTH))
        states = [current_state]
        opt_actions = []

        num_term = 10

        while not won:
            action = agent.get_action(current_state)
            feedback = game.guess(action)

            # handle teacher
            opt_action = teacher.make_guess()
            teacher.add_feedback(action, feedback)
            print 'opt action:', opt_action, 'chosen action:', action, 'feedback:', feedback

            won = game.is_winning_feedback(feedback)
            opt_action_vec = np.array(guess_to_vector(opt_action))
            opt_actions.append(opt_action_vec)

            if len(states) > num_term or won:
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

    def __init__(self, chkpt=None):
        self.env = Environment()
        self.model = DaggerModel(chkpt=chkpt)

    def run(self, num_episode=1000):
        for i in xrange(num_episode):
            print 'Game:', i+1
            states, actions, won = self.env.run_episode(self.model)
            self.model.train(states, actions)
            print '{} Number of actions: {}'.format(won, len(actions))

        self.model.save()



if __name__ == '__main__':
    chkpt = 'models/dagger'
    runner = DaggerRunner(chkpt=chkpt)
    runner.run()

