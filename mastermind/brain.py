import tensorflow as tf
import os
import numpy as np
import random
import time
import tqdm
import pickle
from threading import Thread
from copy import copy
from baselines import MaxEntropyPlayer, SwaszekPlayer
from encoder_nn import EncoderModel
from mastermind import ENCODER_VECTOR_LENGTH, Mastermind, NUM_PEGS, NUM_OPTIONS, EMBEDDED_LENGTH, random_guess, guess_to_vector, ALL_GUESSES, random_guess, feedback_to_vector
from solver import MastermindSolver
from dagger import pretty_print_numpy

# supress warnings...
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

OPTIONS_LENGTH = NUM_PEGS*NUM_OPTIONS
PERMITTED_ACTIONS = 10
NUM_FEEDBACK = 2
FEEDBACK_LENGTH = NUM_FEEDBACK*(NUM_PEGS+1)


class BrainModel(object):

    def __init__(self, chkpt=None):
        self.gamma = 2

        # load from chkpt if exists
        self.create_graph()

        if chkpt:
            self.load(chkpt)

    def create_graph(self):
        self.session = tf.Session()

        # up to PERMITTED_ACTIONS embedded inputs
        self.input_layer = tf.placeholder(tf.float32, [None, EMBEDDED_LENGTH*PERMITTED_ACTIONS])
        self.action_input = tf.placeholder(tf.float32, [None, OPTIONS_LENGTH])

        # training info
        self.teacher_layer = tf.placeholder(tf.float32, [None, FEEDBACK_LENGTH])
        self.weights = tf.placeholder(tf.float32, [None, 1])

        layer1 = tf.layers.dense(tf.concat([self.input_layer, self.action_input], 1), 1000, activation=tf.nn.relu)
        
        # feedback layers
        self.pre_out_layer = tf.layers.dense(layer1, FEEDBACK_LENGTH)
        self.split_pre_out_layers = tf.split(self.pre_out_layer, NUM_FEEDBACK, 1)
        self.out_layers = [tf.nn.softmax(layer) for layer in self.split_pre_out_layers]
        self.feedback_layer = tf.concat(self.out_layers, 1)

        # loss
        self.split_teacher = tf.split(self.teacher_layer, NUM_FEEDBACK, 1)

        self.loss = tf.reduce_sum([tf.reduce_mean(
            self.weights * tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.split_teacher[i], logits=self.split_pre_out_layers[i]))
            for i in xrange(len(self.out_layers))
        ])
        self.optimizer = tf.train.AdamOptimizer(1e-3)
        self.min_loss = self.optimizer.minimize(self.loss)

        initializer = tf.global_variables_initializer()
        self.session.run(initializer)

        self.saver = tf.train.Saver()

    def get_feedback_layer(self, state, action):
        # action is one hot encoded
        return self.session.run([self.out_layers], feed_dict={
            self.input_layer: state,
            self.action_input: action
        })[0]

    def get_action_vector(self, embedding):
        return self.session.run([self.actor_layer], feed_dict={
            self.input_layer: embedding
        })

    def train(self, state, action, feedback, weights):
        loss, _ = self.session.run([self.loss, self.min_loss], feed_dict={
            self.input_layer: state,
            self.action_input: action,
            self.teacher_layer: feedback,
            self.weights: weights
        })
        print('Loss: {}'.format(loss))

    def load(self, name='models/brain'):
        self.saver.restore(self.session, name)

    def save(self, name='models/brain'):
        self.saver.save(self.session, name)
        print 'Saved model to {}!'.format(name)


class BrainTrainer(object):

    def __init__(self, chkpt=None, buffer_size=5000):
        self.encoder = EncoderModel(chkpt='models/encoder_model')

        # experience buffer
        self.buffer_size = buffer_size
        self.buffer_state = []
        self.buffer_av = []
        self.buffer_feedback = []
        self.buffer_weights = []

        if chkpt is None:
            self.brain = BrainModel()
        else:
            self.brain = BrainModel(chkpt=chkpt)


    def _run_episode(self):
        game = Mastermind()

        guesses = [random_guess() for i in xrange(random.randint(0,5))]
        action_feedback = [(guess, game.guess(guess)) for guess in guesses]
        state = self.encoder.create_current_state(action_feedback)

        next_action = random_guess()
        action_vector = np.array(guess_to_vector(next_action))#.reshape(-1, OPTIONS_LENGTH)

        feedback = np.array(feedback_to_vector(game.guess(next_action)))#.reshape(-1, FEEDBACK_LENGTH)

        # add to buffer and update
        self.add_to_buffer(state, action_vector, feedback)

    def add_to_buffer(self, state, action_vector, feedback):
        self.buffer_state.append(state)
        self.buffer_av.append(action_vector)
        self.buffer_feedback.append(feedback)
        self.buffer_weights.append(1)

        # if too many in buffer, remove from front
        if len(self.buffer_weights) > self.buffer_size:
            self.buffer_state.pop(0)
            self.buffer_av.pop(0)
            self.buffer_feedback.pop(0)
            self.buffer_weights.pop(0)

    def train_brain(self):
        state = np.array(self.buffer_state).reshape(-1, EMBEDDED_LENGTH*PERMITTED_ACTIONS)
        action_vector = np.array(self.buffer_av).reshape(-1, OPTIONS_LENGTH)
        feedback = np.array(self.buffer_feedback).reshape(-1, FEEDBACK_LENGTH)
        weights = np.array(self.buffer_weights).reshape(-1, 1)

        self.brain.train(state, action_vector, feedback, weights)

        # now to update weights, give higher weight to incorrect, lower correct
        results = self.brain.get_feedback_layer(state, action_vector)
        for i in xrange(len(self.buffer_weights)):
            if np.argmax(results[0][i]) == np.argmax(self.buffer_feedback[i]):
                self.buffer_weights[i] /= 2
            else:
                self.buffer_weights[i] += 0.1

            self.buffer_weights[i] = max(self.buffer_weights[i], 0.1)

        # turn buffer average to 1
        avg = 1.*sum(self.buffer_weights)/len(self.buffer_weights)
        factor = 1./avg
        for i in xrange(len(self.buffer_weights)):
            self.buffer_weights[i] *= factor

    def run(self, num_times=100000):
        for i in xrange(num_times):
            print 'Episode {}...'.format(i+1)
            self._run_episode()

            if (i+1) % 100 == 0:
                self.train_brain()
        self.brain.save()

    def test(self):
        game = Mastermind()
        print 'Target:', game.target

        guesses = [random_guess() for i in xrange(random.randint(0,0))]
        action_feedback = [(guess, game.guess(guess)) for guess in guesses]
        state = self.encoder.create_current_state(action_feedback)

        print 'Previous actions:', action_feedback

        next_action = random_guess()
        print 'Action:', next_action
        print 'feedback:', game.guess(next_action)
        action_vector = np.array(guess_to_vector(next_action)).reshape(-1, OPTIONS_LENGTH)

        feedback = np.array(feedback_to_vector(game.guess(next_action))).reshape(-1, FEEDBACK_LENGTH)
        print 'Proposed feedback:', pretty_print_numpy(self.brain.get_feedback_layer(state, action_vector))



if __name__ == '__main__':
    # BrainTrainer().run()
    BrainTrainer(chkpt='models/brain').test()
