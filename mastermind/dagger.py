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
from mastermind import ENCODER_VECTOR_LENGTH, Mastermind, NUM_PEGS, NUM_OPTIONS, EMBEDDED_LENGTH, random_guess, guess_to_vector, ALL_GUESSES, random_guess

# supress warnings...
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

OPTIONS_LENGTH = NUM_PEGS*NUM_OPTIONS
PERMITTED_ACTIONS = 10

def action_from_vector(choices_vector, previous_actions, num_tries=0):
    choices = []
    num_tries += 1

    # choose legit randomly if can't do anything
    if num_tries >= 10:
        guess = random_guess()
        while str(guess) in previous_actions:
            guess = random_guess()
        return random_guess()


    for vec in choices_vector:
        vec = vec[0]
        # need to stabilize because might be slightly off of one
        vec = vec/np.sum(vec)

        # choose probabilistically
        choices.append(np.random.choice(NUM_OPTIONS, p=vec))
        # choices.append(np.argmax(vec))

    # don't repeat actions
    if str(choices) in previous_actions:
        return action_from_vector(choices_vector, previous_actions, num_tries=num_tries)

    return choices


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
        self.split_pre_out_layers = tf.split(self.pre_out_layer, NUM_PEGS, 1)
        self.out_layers = [tf.nn.softmax(layer) for layer in self.split_pre_out_layers]
        self.actor_layer = tf.concat(self.out_layers, 1)

        # loss
        self.split_teacher = tf.split(self.teacher_layer, NUM_PEGS, 1)

        self.loss = tf.reduce_sum([tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.split_teacher[i], logits=self.split_pre_out_layers[i]))
            for i in xrange(len(self.out_layers))
        ])
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

    def get_action(self, embedding, previous_actions):
        choice = self.get_choice_vectors(embedding)
        # print choice
        return action_from_vector(choice, previous_actions)

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
        # teacher = SwaszekPlayer() # use for faster output

        num_actions = 0
        won = False

        action_feedback = []
        states = []
        opt_actions = []

        while not won:
            current_state = self.encoder.create_current_state(action_feedback)
            previous_actions = set([str(a) for (a, f) in action_feedback])

            action = agent.get_action(current_state, previous_actions)
            feedback = game.guess(action)

            # handle teacher
            opt_action = teacher.make_guess()
            teacher.add_feedback(action, feedback)

            won = game.is_winning_feedback(feedback)
            opt_action_vec = np.array(guess_to_vector(opt_action))

            opt_actions.append(opt_action_vec)
            states.append(current_state)
            action_feedback.append((action, feedback))

            if len(states) > PERMITTED_ACTIONS or won:
                break

        return states, opt_actions, won


class DaggerRunner(object):

    def __init__(self, chkpt=None, new_chkpt='models/dagger', load_old=True):
        self.env = Environment()
        self.model = DaggerModel(chkpt=chkpt)
        self.new_chkpt = new_chkpt
        self.dataset_states = []
        self.dataset_actions = []

        # loads old data from pickle so that we can resume where we left off
        if load_old:
            try:
                self.dataset_states = pickle.load( open( "pickle_data/states.p", "rb" ) )
                self.dataset_actions = pickle.load( open( "pickle_data/actions.p", "rb" ) )
            except:
                pass

    def run(self, num_episode=1000):
        best_eval = 0.5 # start with a baseline
        for i in xrange(num_episode):
            print 'Game:', i+1
            states, actions, won = self.env.run_episode(self.model)
            self.dataset_states.append(states)
            self.dataset_actions.append(actions)

            # TODO turn states/actions into "memory buffer" instead of full data set

            # train the model with aggregated states and actions
            states_agg = np.vstack(self.dataset_states).reshape(-1, EMBEDDED_LENGTH*PERMITTED_ACTIONS)
            actions_agg = np.vstack(self.dataset_actions).reshape(-1, OPTIONS_LENGTH)

            self.model.train(states_agg, actions_agg)

            # save a checkpoint of data every once in a while in case want to break early
            if (i+1) % 10000 == 0:
                pickle.dump(self.dataset_states, open( "pickle_data/states.p", "wb" ) )
                pickle.dump(self.dataset_actions, open( "pickle_data/actions.p", "wb" ) )
                self.model.save(name=self.new_chkpt)

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

    for trial in xrange(num_trials):
        print 'Trial {}/{}...'.format(trial+1, num_trials)
        for i, target in tqdm.tqdm(enumerate(ALL_GUESSES), total=len(ALL_GUESSES)):
            # play a random game
            game = Mastermind(target=target)
            won = False
            action_feedback = []

            while not won:
                current_state = encoder.create_current_state(action_feedback)
                previous_actions = set([str(a) for (a, f) in action_feedback])

                action = agent.get_action(current_state, previous_actions)
                feedback = game.guess(action)
                action_feedback.append((action, feedback))
                won = game.is_winning_feedback(feedback)

                if len(action_feedback) >= PERMITTED_ACTIONS:
                    break

            num_moves.append(len(action_feedback))
            if won:
                successes += 1
                num_moves_success.append(len(action_feedback))

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
    action_feedback = []

    while not won:
        current_state = encoder.create_current_state(action_feedback)
        print agent.get_choice_vectors(current_state)
        previous_actions = set([str(a) for (a, f) in action_feedback])

        action = agent.get_action(current_state, previous_actions)
        feedback = game.guess(action)
        print action, feedback
        action_feedback.append((action, feedback))
        won = game.is_winning_feedback(feedback)

        if len(action_feedback) >= 10:
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
    chkpt = 'models/dagger'
    new_chkpt = 'models/dagger_6peg'
    runner = DaggerRunner(chkpt=None, new_chkpt=new_chkpt)
    runner.run()
    check(new_chkpt, show_results=True)

    investigate_moves(new_chkpt)
    # print guess_to_vector([0, 0, 1, 1])
