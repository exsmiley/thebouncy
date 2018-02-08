import tensorflow as tf
import os
import numpy as np
import random
from mastermind import ENCODER_VECTOR_LENGTH, Mastermind, random_guess, unencoded_vector


# supress warnings...
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


def generate_sample_encoder_input(num=10):
    samples = []

    for _ in xrange(num):
        game = Mastermind()
        guess = random_guess()
        feedback = game.guess(guess)
        samples.append(unencoded_vector(guess, feedback))

    return np.array(samples)


class EncoderModel(object):

    def __init__(self, chkpt=None):
        # load from chkpt if exists
        self.create_graph()
        if chkpt:
            self.load(chkpt)

    def create_graph(self):
        self.session = tf.Session()

        self.input_layer = tf.placeholder(tf.float32, [None, ENCODER_VECTOR_LENGTH])
        self.labels = tf.placeholder(tf.float32, [None, ENCODER_VECTOR_LENGTH])

        self.embedded_layer = tf.layers.dense(self.input_layer, 10, activation=tf.nn.relu)

        self.out_layer = tf.layers.dense(self.embedded_layer, ENCODER_VECTOR_LENGTH)

        self.loss = tf.reduce_mean(tf.squared_difference(self.labels, self.out_layer))
        
        self.optimizer = tf.train.AdamOptimizer(1e-3)
        self.train = self.optimizer.minimize(self.loss)

        initializer = tf.global_variables_initializer()
        self.session.run(initializer)

        self.saver = tf.train.Saver()

    def learn(self, num_epochs=10000):
        for _ in xrange(num_epochs):
            samples = generate_sample_encoder_input()
            self._learn_samples(samples, samples)

    def _learn_samples(self, x, y):
        loss, train = self.session.run([self.loss, self.train], feed_dict={
            self.input_layer: x,
            self.labels: y
        })
            # losses.append(loss)
        print 'loss:', loss

    def get_embeddings(self, guess, feedback):
        x = np.array(unencoded_vector(guess, feedback)).reshape(-1, ENCODER_VECTOR_LENGTH)
        # x = np.array(x).reshape(-1, 2)
        return self.session.run([self.embedded_layer], feed_dict={
            self.input_layer: x,
        })[0][0]

    def load(self, name='models/encoder_model'):
        self.saver.restore(self.session, name)

    def save(self, name='models/encoder_model'):
        self.saver.save(self.session, name)
        print 'Saved model to {}!'.format(name)



if __name__ == '__main__':
    chkpt = 'models/encoder_model'
    model = EncoderModel(chkpt=chkpt)

    # model.learn()
    # model.save()
    print model.get_embeddings([3, 4, 5, 4], (3, 0))



