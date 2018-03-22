import tensorflow as tf
import os
import numpy as np
import random
import time
import tqdm
import pickle
from copy import copy
from zoombinis import *

# tunable hyperparameters
NUM_RUNS = 10000
LEARNING_RATE = 1e-3
to_train = True

# supress warnings...
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'


class ZBrain(object):

    def __init__(self, chkpt=None):
        # load from chkpt if exists
        self.create_graph()

        if chkpt:
            self.load(chkpt)

    def create_graph(self):
        self.session = tf.Session()

        self.input_layer = tf.placeholder(tf.float32, [None, INPUT_LENGTH])

        # training info
        self.teacher_layer = tf.placeholder(tf.float32, [None, FEEDBACK_LENGTH])

        layer1 = tf.layers.dense(self.input_layer, 2500, activation=tf.nn.relu)
        
        # feedback layers
        self.pre_out_layer = tf.layers.dense(layer1, FEEDBACK_LENGTH)
        self.feedback_layer = tf.nn.softmax(self.pre_out_layer)

        # loss
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.teacher_layer, logits=self.pre_out_layer))
        self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        self.min_loss = self.optimizer.minimize(self.loss)

        initializer = tf.global_variables_initializer()
        self.session.run(initializer)

        self.saver = tf.train.Saver()

    def get_feedback_layer(self, state):
        # action is one hot encoded
        probs = self.session.run([self.feedback_layer], feed_dict={
            self.input_layer: state,
        })[0]
        return probs

    def get_entropy(self, state):
        probs = [
            p*np.log2(p)
            for p in self.get_feedback_layer(state)[0]
            if p > 1e-8
        ]
        return -sum(probs)

    def train(self, state, feedback):
        loss, _ = self.session.run([self.loss, self.min_loss], feed_dict={
            self.input_layer: state,
            self.teacher_layer: feedback
        })
        print('Loss: {}'.format(loss))

    def load(self, name='models/brain'):
        self.saver.restore(self.session, name)

    def save(self, name='models/brain'):
        self.saver.save(self.session, name)
        print 'Saved model to {}!'.format(name)


class ZTrainer(object):

    def __init__(self, chkpt=None):
        self.model = ZBrain(chkpt=chkpt)
        self.state_buffer = []
        self.feedback_buffer = []

    def run(self):
        for i in xrange(NUM_RUNS):
            states, feedbacks = self.play_game()
            self.state_buffer.extend(states)
            self.feedback_buffer.extend(feedbacks)

            if (i+1) % 10 == 0:
                print 'Episode {}...'.format(i+1)
                self.train()

        self.model.save()

    def train(self):
        all_indices = [i for i in xrange(len(self.state_buffer))]
        indices = set(random.sample(all_indices, 20))

        states = [state for i, state in enumerate(self.state_buffer) if i in indices]
        feedbacks = [fb for i, fb in enumerate(self.feedback_buffer) if i in indices]

        states = np.array(states).reshape(-1, INPUT_LENGTH)
        feedbacks = np.array(feedbacks).reshape(-1, FEEDBACK_LENGTH)
        self.model.train(states, feedbacks)

        # reset buffers
        self.state_buffer = []
        self.feedback_buffer = []


    def play_game(self):
        game = Game()
        sent_indices = set()
        states = []
        feedbacks = []

        while game.can_move():
            index = random.randint(0, NUM_ZOOMBINIS-1)

            while index in sent_indices:
                index = random.randint(0, NUM_ZOOMBINIS-1)

            state = game.get_state_vector(index)

            send_top = True if random.randint(0, 1) else False
            feedback = game.send_zoombini(index, send_top)

            if (feedback and send_top) or (not feedback and not send_top):
                feedback_vec = [1, 0]
            else:
                feedback_vec = [0, 1]

            if feedback:
                sent_indices.add(index)

            states.append(state)
            feedbacks.append(feedback_vec)

        

        return states, feedbacks

    def test(self):
        game = Game()
        print game
        sent_indices = set()

        while game.can_move():
            index = random.randint(0, NUM_ZOOMBINIS-1)

            while index in sent_indices:
                index = random.randint(0, NUM_ZOOMBINIS-1)

            state = game.get_state_vector(index)
            state = np.array(state).reshape(-1, INPUT_LENGTH)

            probabilities = self.model.get_feedback_layer(state)
            entropy = self.model.get_entropy(state)
            print '\nGoing to send index', index, game.zoombinis[index]
            print 'Probabilities: {}\nEntropy: {}'.format(probabilities, entropy)

            send_top = True if random.randint(0, 1) else False
            feedback = game.send_zoombini(index, send_top)
            print feedback, send_top
            if (feedback and send_top) or (not feedback and not send_top):
                feedback_vec = [1, 0]
            else:
                feedback_vec = [0, 1]
            print 'Feedback:', feedback_vec

            if feedback:
                sent_indices.add(index)



if __name__ == '__main__':
    if to_train:
        trainer = ZTrainer()
        trainer.run()
        trainer.test()
    else:
        trainer = ZTrainer(chkpt='models/brain')
        trainer.test()
