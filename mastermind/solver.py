from z3 import Solver, Int, Bool, Or, Xor, And, If, Not, sat, is_true, is_false, unsat, unknown, Sum
from mastermind import NUM_PEGS, NUM_OPTIONS


class MastermindSolver(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.solver = Solver()
        self.variables = []

        for i in xrange(NUM_PEGS):
            v = Int('peg_%d' % i)
            self.variables.append(v)
            self.solver.add(And(0 <= v, v < NUM_OPTIONS))

    def add_feedback(self, action_feedback):
        query, (num_present, num_match) = action_feedback

        # check present
        nums = []
        for k in xrange(NUM_OPTIONS):
            ifs = []
            count_query = 0
            for i in xrange(len(query)):
                ifs.append(If(self.variables[i] == k, 1, 0))
                if query[i] == k:
                    count_query += 1
            if_sum = Sum(ifs)
            nums.append(If(if_sum < count_query, if_sum, count_query))
        present_constraint = Sum(nums) == num_present
        self.solver.add(present_constraint)

        # check match
        match_constraint = Sum([If(self.variables[i] == query[i], 1, 0) for i in xrange(NUM_PEGS)]) == num_match
        self.solver.add(match_constraint)

    def solve(self):
        if self.solver.check() == sat:
            model = self.solver.model()
            nums = []
            for v in self.variables:
                nums.append(int(model[v].as_long()))
            return nums
        else:
            print 'UNSAT'

    def is_unique_solution(self):
        soln = self.solve()
        self.solver.push()

        self.solver.add(Not(And([
            self.variables[i] == soln[i]
            for i in xrange(NUM_PEGS)
        ])))

        if self.solve():
            self.solver.pop()
            return False, None
        else:
            self.solver.pop()
            return True, soln



def solve(num_pegs=NUM_PEGS, num_options=NUM_OPTIONS, feedbacks=None):
    """Solves the problem given the list of feedbacks"""
    s = MastermindSolver()
    feedbacks = feedbacks or []

    for feedback in feedbacks:
        s.add_feedback(feedback)

    return s.solve()


def test1():
    answers = ['2002', '0202', '2020']
    feedbacks = [([0, 0, 2, 2], 4, 2)]
    assert ''.join(map(str, solve(feedbacks=feedbacks))) in answers


def test2():
    answers = ['5302', '7502']
    feedbacks = [([0, 0, 2, 2], 2, 1), ([1, 0, 3, 7], 2, 0), ([3, 7, 1, 2], 2, 1), ([7, 3, 0, 2], 3, 3), ([5, 5, 5, 5], 1, 1)]
    assert ''.join(map(str, solve(feedbacks=feedbacks, num_options=8))) in answers


def test3():
    answers = ['4005', '4015', '4025', '4035', '4045']
    feedbacks = [([4, 5, 6, 7], 2, 1), ([4, 0, 6, 7], 2, 2), ([4, 0, 5, 8], 3, 2), ([4, 0, 9, 5], 3, 3)]
    assert ''.join(map(str, solve(feedbacks=feedbacks))) in answers


if __name__ == '__main__':
    test1()
    test2()
    test3()


