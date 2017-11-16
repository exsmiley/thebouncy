import random
from player import Player
from game import validate_attempt


class FiveGuessPlayer(Player):
    '''uses the Knuth 5 Guess algorithm (https://en.wikipedia.org/wiki/Mastermind_(board_game))'''

    def __init__(self, num_pegs=4, num_options=10):
        super(FiveGuessPlayer, self).__init__(num_pegs=num_pegs, num_options=num_options)
        self._setup()

    def _setup(self):
        # construct all possible answers
        all_possible = []
        i = 0
        while i < self.num_pegs:
            temp = []
            for opt in xrange(self.num_options):
                if len(all_possible) == 0:
                    temp.append([opt])
                for pos in all_possible:
                    temp.append(pos + [opt])
            all_possible = temp
            i += 1
        self.all_possible = all_possible
        self.remaining_answers = self.all_possible

    def make_guess(self):
        if len(self.attempts) == 0:
            return [1, 1, 2, 2]
        elif len(self.remaining_answers) == 1:
            return self.remaining_answers[0]
        else:
            return self.run_minimax()

    def add_feedback(self, guess, feedback):
        super(FiveGuessPlayer, self).add_feedback(guess, feedback)
        
        # only keep answers that give the same feedback
        still_remaining = []
        for target in self.remaining_answers:
            if validate_attempt(target, guess) == feedback:
                still_remaining.append(target)
        self.remaining_answers = still_remaining

    def run_minimax(self):
        possible_scores = [(i,j) for i in xrange(self.num_pegs) for j in xrange(self.num_pegs) if i >= j]
        best_count = -1
        best_guess = None
        for i, guess in enumerate(self.all_possible):
            eliminated_counts = []
            for score in possible_scores:
                eliminated_count = 0
                for answer in self.remaining_answers:
                    if validate_attempt(guess, answer) != score:
                        eliminated_count += 1
                eliminated_counts.append(eliminated_count)
            eliminated_count = min(eliminated_counts)

            # if i % (len(self.all_possible)/10) == 0:
            #     print '{}/{}'.format(i, len(self.all_possible)), best_guess, best_count
            
            if eliminated_count > best_count and tuple(guess) not in self.used:
                best_count = eliminated_count
                best_guess = guess
        return best_guess

    def reset(self):
        super(FiveGuessPlayer, self).reset()
        self._setup()


class Swaszek(Player):
    '''uses Swaszek's strategy'''

    def __init__(self, num_pegs=4, num_options=10):
        super(Swaszek, self).__init__(num_pegs=num_pegs, num_options=num_options)
        self._setup()

    def _setup(self):
        # construct all possible answers
        all_possible = []
        i = 0
        while i < self.num_pegs:
            temp = []
            for opt in xrange(self.num_options):
                if len(all_possible) == 0:
                    temp.append([opt])
                for pos in all_possible:
                    temp.append(pos + [opt])
            all_possible = temp
            i += 1
        self.all_possible = all_possible
        self.remaining_answers = self.all_possible

    def make_guess(self):
        print self.remaining_answers
        return random.choice(self.remaining_answers)

    def add_feedback(self, guess, feedback):
        super(Swaszek, self).add_feedback(guess, feedback)

        # only keep answers that give the same feedback
        still_remaining = []
        print [4, 0, 7, 5] in self.remaining_answers
        for target in self.remaining_answers:
            if validate_attempt(target, guess) == feedback:
                still_remaining.append(target)
        self.remaining_answers = still_remaining
        print 'made {} guesses with {}/{} remaining'.format(len(self.attempts), len(self.remaining_answers), len(self.all_possible))

    def reset(self):
        super(Swaszek, self).reset()
        self._setup()

if __name__ == '__main__':
    s = Swaszek(num_options=10)
    feedbacks = [([4, 5, 6, 7], 2, 1), ([4, 0, 6, 7], 2, 2), ([4, 0, 5, 8], 3, 2), ([4, 0, 7, 5], 3, 3)]
    for f in feedbacks:
        s.add_feedback(f[0], (f[1], f[2]))
    print s.make_guess()
    print s.make_guess()

