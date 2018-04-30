import tensorflow as tf
import os
import numpy as np
import random
import time
from threading import Thread
from copy import copy
from encoder_nn import EncoderModel
from mastermind import ENCODER_VECTOR_LENGTH, Mastermind, NUM_PEGS, NUM_OPTIONS, EMBEDDED_LENGTH, random_guess, guess_to_vector


OPTIONS_LENGTH = NUM_PEGS*NUM_OPTIONS


# supress warnings...
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


def action_from_vector(choices_vector):
    choices = []

    for vec in choices_vector:
        vec = vec[0]
        # need to stabilize because might be slightly off of one
        vec = vec/np.sum(vec)

        # choose probabilistically
        choices.append(np.random.choice(NUM_OPTIONS, p=vec))
        # choices.append(np.argmax(vec))
    return choices


# Neural Network Hyperparameters
RUN_TIME = 30
THREADS = 8
OPTIMIZERS = 2
THREAD_DELAY = 0.001

GAMMA = 0.99

N_STEP_RETURN = 8
GAMMA_N = GAMMA ** N_STEP_RETURN

EPS_START = 0.99
EPS_STOP  = .01
EPS_STEPS = 7500000

MIN_BATCH = 32
LEARNING_RATE = 1e-6


LOSS_V = .5         # v loss coefficient
LOSS_ENTROPY = .01  # entropy coefficient

# global variable used for epsilon calculation
total_steps = 0

def format_rewards(rewards):
    rewards_with_future = []
    for i, reward in enumerate(rewards):
        rewards_with_future.append(sum([val2*(GAMMA**j) for j, val2 in enumerate(rewards[i:])]))
    return rewards_with_future


class ActorModel(object):
    # inspired by https://jaromiru.com/2017/03/26/lets-make-an-a3c-implementation/

    def __init__(self, chkpt=None):
        # load from chkpt if exists
        self.create_graph()
        if chkpt:
            self.load(chkpt)

    def create_graph(self):
        self.session = tf.Session()

        self.input_layer = tf.placeholder(tf.float32, [None, EMBEDDED_LENGTH])
        # self.labels = tf.placeholder(tf.float32, [None, ENCODER_VECTOR_LENGTH])

        layer1 = tf.layers.dense(self.input_layer, 100*NUM_PEGS, activation=tf.nn.relu)
        layer2 = tf.layers.dense(layer1, 100*NUM_PEGS, activation=tf.nn.relu)
        layer3 = tf.layers.dense(layer2, 100*NUM_PEGS, activation=tf.nn.relu)
        layer4 = tf.layers.dense(layer3, 75*NUM_PEGS, activation=tf.nn.relu)
        self.last_layer = tf.layers.dense(layer4, 50*NUM_PEGS, activation=tf.nn.relu)
        
        # actor layers
        self.pre_out_layer = tf.layers.dense(self.last_layer, OPTIONS_LENGTH)
        self.split_out_layers = tf.split(self.pre_out_layer, NUM_PEGS, 1)
        self.out_layers = [tf.nn.softmax(layer) for layer in self.split_out_layers]
        self.actor_layer = tf.concat(self.out_layers, 1)

        # critic layer
        self.value_layer = tf.layers.dense(self.last_layer, 1)

        # loss stuff
        # s_t in piyomaru is just our input_layer
        self.a_t = tf.placeholder(tf.float32, shape=(None, OPTIONS_LENGTH))
        self.r_t = tf.placeholder(tf.float32, shape=(None, 1)) # not immediate, but discounted n step reward

        log_prob = tf.log( tf.reduce_sum(self.actor_layer * self.a_t, axis=1, keep_dims=True) + 1e-10)
        advantage = (self.r_t - self.value_layer)

        self.loss_policy = - log_prob * tf.stop_gradient(advantage)                                  # maximize policy (without value error)
        self.loss_value  = LOSS_V * tf.square(advantage)                                             # minimize value error
        entropy = LOSS_ENTROPY * tf.reduce_sum(self.actor_layer * tf.log(self.actor_layer + 1e-10), axis=1, keep_dims=True)   # maximize entropy (regularization)

        self.loss_total = tf.reduce_mean(self.loss_policy + self.loss_value + entropy)

        optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, decay=.99)
        self.minimize = optimizer.minimize(self.loss_total)

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
        return action_from_vector(self.get_choice_vectors(embedding))

    def get_val_and_action(self, embedding):
        val, action = self.session.run([self.value_layer, self.out_layers], feed_dict={
            self.input_layer: embedding
        })
        val = val[0][0]
        action = action
        return val, action

    def train(self, states, actions, rewards):
        _, minimize, policy, value = self.session.run([self.loss_total, self.minimize, self.loss_policy, self.loss_value], feed_dict={
            self.input_layer: states,
            self.a_t: actions,
            self.r_t: rewards
        })
        print('Loss Policy: {}\nLoss Value: {}'.format(np.mean(policy), np.mean(value)))

    def load(self, name='models/actor_model'):
        self.saver.restore(self.session, name)

    def save(self, name='models/actor_model'):
        self.saver.save(self.session, name)
        print 'Saved model to {}!'.format(name)


class Agent(object):

    def __init__(self, actor_nn):
        self.actor_nn = copy(actor_nn) # use copy of nn for stability

    def get_action(self, state):
        global total_steps
        if(total_steps >= EPS_STEPS):
            epsilon =  EPS_STOP
        else:
            epsilon = EPS_START + total_steps * (EPS_STOP - EPS_START) / EPS_STEPS

        total_steps += 1

        if random.random() < epsilon:
            return random_guess()

        else:
            return self.actor_nn.get_action(state)



class Environment(object):

    def __init__(self):
        self.game_class = Mastermind
        chkpt = 'models/encoder_model'
        self.encoder = EncoderModel(chkpt=chkpt)

    def run_episode(self, agent):

        game = self.game_class()

        num_actions = 0
        won = False

        current_state = np.zeros((1, EMBEDDED_LENGTH))
        states = [current_state]
        actions = []
        rewards = []

        num_term = 100

        while len(states) <= num_term and not won:
            action = agent.get_action(current_state)
            feedback = game.guess(action)
            won = game.is_winning_feedback(feedback)
            action_vec = np.array(guess_to_vector(action))
            actions.append(action_vec)
            if won:
                rewards.append(10.)
                break
            else:
                rewards.append(-1./num_term)
            # print states.shape, actions.shape, rewards.shape

            additional_state = self.encoder.get_embeddings(action, feedback)
            current_state += additional_state
            states.append(current_state)

        if len(states) != len(actions):
            states.pop()

        rewards = format_rewards(rewards)

        states = np.vstack(states)
        actions = np.vstack(actions)
        rewards = np.vstack(rewards)

        print len(actions)

        # print states.shape, actions.shape, rewards.shape

        return states, actions, rewards


class A3C(object):

    def __init__(self, actor_chkpt=None, total_games=20):
        self.env = Environment()
        self.actor_nn = ActorModel(chkpt=actor_chkpt)
        
        self.num_games = 0
        self.total_games = total_games
        self.num_threads = 0

    def do_episode(self):
        self.num_games += 1
        print 'Doing game {}...'.format(self.num_games)
        self.num_threads += 1
        states, actions, rewards = self.env.run_episode(Agent(self.actor_nn))
        self.actor_nn.train(states, actions, rewards)
        self.num_threads -= 1

    def run(self):
        while self.num_games < self.total_games:
            if self.num_threads < THREADS:
                thread = Thread(target=self.do_episode)
                thread.start()
                time.sleep(random.random()*0.3)
            else:
                time.sleep(random.random()*1.5)
            if(self.num_games % 1000 == 0):
                self.actor_nn.save('models/actor_nn/{}'.format(self.num_games))
        print 'Finished training...'
        self.actor_nn.save('models/actor_nn/done')


def check_nn():
    game = Mastermind()
    print game.target
    chkpt = 'models/encoder_model'
    encoder = EncoderModel(chkpt=chkpt)
    actor_nn = ActorModel(chkpt='models/actor_nn/5000')
    current_state = np.zeros((1, EMBEDDED_LENGTH))
    for i in xrange(5):
        val, action_vec = actor_nn.get_val_and_action(current_state)
        action = action_from_vector(action_vec)
        feedback = game.guess(action_from_vector(action_vec))
        current_state += encoder.get_embeddings(action, feedback)
        print val, action_vec, action



def train_actor():
    TOTAL_GAMES = 100000
    model = A3C(total_games=TOTAL_GAMES)
    model.run()


if __name__ == '__main__':
    check_nn()
    # train_actor()
    

