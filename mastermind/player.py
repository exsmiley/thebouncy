from mastermind import NUM_PEGS, NUM_OPTIONS

class Player(object):
    
    def __init__(self):
        self.attempts = []
        self.used = set()
        self.num_pegs = NUM_PEGS
        self.num_options = NUM_OPTIONS

    def make_guess(self):
        '''implement in child players'''
        pass

    def add_feedback(self, guess, feedback):
        self.used.add(tuple(guess))
        exist, match = feedback
        attempt = (guess, exist, match)
        self.attempts.append(attempt)

    def reset(self):
        self.attempts = []
        self.used = set()


class PlayerRunner(object):

    def __init__(self, player):
        self.player = player
        self.player_type = player.__class__.__name__

    def play(self, game, loss_threshold=30):
        '''plays the game until the end or 30 moves are made'''
        self.player.reset()
        won_game = False
        while not won_game:
            guess = self.player.make_guess()
            feedback = game.guess(guess)
            self.player.add_feedback(guess, feedback)
            if feedback[1] == game.num_pegs:
                won_game = True
            if len(self.player.attempts) >= loss_threshold:
                break
        return won_game, self.player.attempts
