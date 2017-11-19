from z3 import Solver, Int, Bool, Or, Xor, And, If, Not, sat, is_true, is_false, unsat, unknown, Sum


def solve(num_pegs=4, num_options=10, feedbacks=None):
    """Solves the problem given the list of feedbacks"""
    feedbacks = feedbacks or []

    variables = []
    constraints = []

    for i in xrange(num_pegs):
        v = Int('peg_%d' % i)
        variables.append(v)
        constraints.append(And(0 <= v, v < num_options))

    # feedback is (query (4 nums), num_present, num_match)
    for query, num_present, num_match in feedbacks:
        # check present
        nums = []
        for k in xrange(num_options):
            ifs = []
            count_query = 0
            for i in xrange(len(query)):
                ifs.append(If(variables[i] == k, 1, 0))
                if query[i] == k:
                    count_query += 1
            if_sum = Sum(ifs)
            nums.append(If(if_sum < count_query, if_sum, count_query))
        present_constraint = Sum(nums) == num_present
        constraints.append(present_constraint)

        # check match
        match_constraint = Sum([If(variables[i] == query[i], 1, 0) for i in xrange(num_pegs)]) == num_match
        constraints.append(match_constraint)

    s = Solver()
    s.add(constraints)
    if s.check() == sat:
        model = s.model()
        nums = []
        for v in variables:
            nums.append(int(model[v].as_long()))
        return nums
    else:
        print 'UNSAT'


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


