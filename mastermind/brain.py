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
from mastermind import ENCODER_VECTOR_LENGTH, Mastermind, NUM_PEGS, NUM_OPTIONS, EMBEDDED_LENGTH, random_guess, guess_to_vector, ALL_GUESSES, random_guess, feedback_to_vector, feedback_index
from solver import MastermindSolver

# supress warnings...
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

OPTIONS_LENGTH = NUM_PEGS*NUM_OPTIONS + NUM_PEGS
PERMITTED_ACTIONS = 10
NUM_FEEDBACK = 2
FEEDBACK_LENGTH = NUM_PEGS*(NUM_PEGS+1)+2


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

        layer1 = tf.layers.dense(tf.concat([self.action_input, self.input_layer], 1), 1000, activation=tf.nn.relu)
        
        # feedback layers
        self.pre_out_layer = tf.layers.dense(layer1, FEEDBACK_LENGTH)
        self.feedback_layer = tf.nn.softmax(self.pre_out_layer)

        # loss
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.teacher_layer, logits=self.pre_out_layer))
        self.optimizer = tf.train.AdamOptimizer(1e-3)
        self.min_loss = self.optimizer.minimize(self.loss)

        initializer = tf.global_variables_initializer()
        self.session.run(initializer)

        self.saver = tf.train.Saver()

    def focal_loss(self, logits, labels, weights):
        eps = 1e-8
        return -weights*(
            ((1-logits)**self.gamma)*tf.log(logits+eps)*(labels) +
            ((logits)**self.gamma)*tf.log(1-logits+eps)*(1-labels)
        )

    def get_feedback_layer(self, state, action):
        # action is one hot encoded
        probs = self.session.run([self.feedback_layer], feed_dict={
            self.input_layer: state,
            self.action_input: action
        })[0]
        return probs

    def get_entropy(self, state, action):
        probs = [
            p*np.log2(p)
            for p in self.get_feedback_layer(state, action)[0]
        ]
        return -sum(probs)

    def train(self, state, action, feedback):
        loss, _ = self.session.run([self.loss, self.min_loss], feed_dict={
            self.input_layer: state,
            self.action_input: action,
            self.teacher_layer: feedback
        })
        print('Loss: {}'.format(loss))

    def load(self, name='models/brain'):
        self.saver.restore(self.session, name)

    def save(self, name='models/brain'):
        self.saver.save(self.session, name)
        print 'Saved model to {}!'.format(name)


class BrainTrainer(object):

    def __init__(self, chkpt=None, buffer_size=20000):
        tf.reset_default_graph()
        self.encoder = EncoderModel(chkpt='models/encoder_model')

        # experience buffer
        self.buffer_size = buffer_size
        self.buffer_state = []
        self.buffer_av = []
        self.buffer_feedback = []
        # self.buffer_weights = []

        if chkpt is None:
            self.brain = BrainModel()
        else:
            self.brain = BrainModel(chkpt=chkpt)


    def _run_episode(self):
        game = Mastermind()

        guesses = []
        action_feedback = []

        for i in xrange(11):
            state = self.encoder.create_current_state(action_feedback)

            next_action = random_guess()
            action_vector = np.array(guess_to_vector(next_action))#.reshape(-1, OPTIONS_LENGTH)
            feedback = game.guess(next_action)
            feedback_vec = np.array(feedback_to_vector(feedback))#.reshape(-1, FEEDBACK_LENGTH)

            self.add_to_buffer(state, action_vector, feedback_vec)

            guesses.append(next_action)
            action_feedback.append((next_action, feedback))

    def add_to_buffer(self, state, action_vector, feedback):
        self.buffer_state.append(state)
        self.buffer_av.append(action_vector)
        self.buffer_feedback.append(feedback)
        # self.buffer_weights.append(weight)

        # if too many in buffer, remove from front
        if len(self.buffer_state) > self.buffer_size:
            self.buffer_state.pop(0)
            self.buffer_av.pop(0)
            self.buffer_feedback.pop(0)
            # self.buffer_weights.pop(0)

    def train_brain(self):
        state = np.array(self.buffer_state).reshape(-1, EMBEDDED_LENGTH*PERMITTED_ACTIONS)
        action_vector = np.array(self.buffer_av).reshape(-1, OPTIONS_LENGTH)
        feedback = np.array(self.buffer_feedback).reshape(-1, FEEDBACK_LENGTH)
        # weights = np.array(self.buffer_weights).reshape(-1, 2)

        self.brain.train(state, action_vector, feedback)


    def run(self, num_times=100000):
        for i in xrange(num_times):
            print 'Episode {}...'.format(i+1)
            self._run_episode()

            self.train_brain()

            if (i+1) % 10000 == 0:
                self.brain.save('models/brain{}'.format(i))

        self.brain.save()

    def test(self):
        game = Mastermind()
        print 'Target:', game.target

        guesses = [random_guess() for i in xrange(random.randint(0,0))]
        action_feedback = [(guess, game.guess(guess)) for guess in guesses]
        state = self.encoder.create_current_state(action_feedback)

        # print 'Previous actions:', action_feedback

        for i in xrange(10):
            next_action = random_guess()
            feedback = game.guess(next_action)
            print 'Action:', next_action
            print 'Feedback:', feedback
            action_feedback.append((next_action, feedback))
            action_vector = np.array(guess_to_vector(next_action)).reshape(-1, OPTIONS_LENGTH)

            feedback = np.array(feedback_to_vector(game.guess(next_action))).reshape(-1, FEEDBACK_LENGTH)
            print 'Proposed feedback:', pretty_print_numpy(self.brain.get_feedback_layer(state, action_vector))
            print 'Entropy:', self.brain.get_entropy(state, action_vector)
            state = self.encoder.create_current_state(action_feedback)
            print


def find_feedback_distribution(num_tries=25000000):
    counts = {}

    for i in tqdm.tqdm(xrange(num_tries)):
        game = Mastermind()
        guess = random_guess()
        feedback = game.guess(guess)

        if feedback in counts:
            counts[feedback] += 1
        else:
            counts[feedback] = 1

    avg_counts = {}
    for k, v in counts.iteritems():
        avg_counts[k] = 1.*v/num_tries

    print avg_counts

def pretty_print_numpy(a):
    a = a[0]
    feedback_dict = {0: 'unk', 1: (0, 0), 2: (1, 0), 3: (2, 0), 4: (3, 0), 5: (4, 0), 6: (1, 1), 7: (2, 1), 8: (3, 1), 9: (4, 1), 10: (1, 2), 11: (2, 2), 12: (3, 2), 13: (4, 2), 14: (1, 3), 15: (2, 3), 16: (3, 3), 17: (4, 3), 18: (1, 4), 19: (2, 4), 20: (3, 4), 21: (4, 4)}
    return [(feedback_dict[i], round(x, 3)) for i, x in enumerate(a)]

def get_weight(feedback):
    # get the weight for a specific feedback
    # dists = {(3, 2): 0.03085612, (0, 0): 0.07243912, (3, 3): 0.01544488, (3, 0): 0.0488908, (3, 1): 0.06817216, (4, 4): 0.00077264, (2, 1): 0.17494632, (2, 0): 0.171483, (2, 2): 0.08087732, (4, 2): 0.00384536, (1, 0): 0.1866952, (4, 1): 0.00343616, (1, 1): 0.13929196, (4, 0): 0.00284896}
    exist = [0.072, 0.326, 0.427, 0.163, 0.011]
    correct = [0.482, 0.386, 0.116, 0.015, 0.001]
    return [1/exist[feedback[0]], 1/correct[feedback[1]]]
    # return 1/dists[feedback]

if __name__ == '__main__':
    # BrainTrainer().run()
    BrainTrainer(chkpt='models/brain').test()
