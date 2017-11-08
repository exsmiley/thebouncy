import random


def random_numbers(num_pegs=4, num_options=10):
    options = range(num_options)
    return [random.choice(options) for i in xrange(num_pegs)]


class Mastermind(object):

    def __init__(self, target=None, num_options=10, num_pegs=4):
        '''target is the goal array to match'''
        if target is None:
            target = random_numbers(num_pegs=num_pegs, num_options=num_options)
        self.target = target
        self.counts = self._get_counts(target)
        assert len(target) == num_pegs
        self.num_pegs = len(target)
        self.num_options = num_options

    def _get_counts(self, arr):
        '''get counts for each of the numbers'''
        counts = {}
        for num in set(arr):
            counts[num] = 0
            for num2 in arr:
                if num == num2:
                    counts[num] += 1
        return counts

    def guess(self, attempt):
        '''attempt is an array trying to match the target'''
        num_exist = 0
        num_match = 0

        # first compare counts to get num_exist
        counts = self._get_counts(attempt)

        for num, count in counts.iteritems():
            if num in self.counts:
                my_count = self.counts[num]
                num_exist += min(count, my_count)

        # compare arrays to get num_match
        for i in xrange(len(attempt)):
            if attempt[i] == self.target[i]:
                num_match += 1

        return num_exist, num_match


if __name__ == '__main__':
    game = Mastermind([4, 2, 3, 4])
    print game.guess([4, 5, 6, 7]), (1, 0)
    print game.guess([4, 3, 3, 1]), (2, 2)

    target = [2, 6, 2, 5]
    game = Mastermind(target)
    print game.guess(target)
