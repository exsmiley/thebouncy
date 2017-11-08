from z3 import Solver, Int, Bool, Or, Xor, And, If, Not, sat, is_true, is_false, unsat, unknown
from itertools import permutations, combinations


def gen_subsets(length, num_indices=4):
    S = range(num_indices)
    return list(combinations(S, length))


def solve(num_pegs=4, num_options=10, feedbacks=None):
    """Solves the problem given the list of feedback"""
    feedbacks = feedbacks or []

    # generate all possible combinations of indices
    index_subsets = [gen_subsets(i, num_indices=num_pegs) for i in xrange(num_pegs+1)]

    variables = []
    constraints = []

    for i in xrange(num_pegs):
        v = Int('peg_%d' % i)
        variables.append(v)
        constraints.append(And(0 <= v, v < num_options))

    # feedback is (query (4 nums), num_exist, num_match)
    for query, num_exist, num_match in feedbacks:
        if num_exist == 0:
            for num in query:
                for v in variables:
                    constraints.append(v != num)
        else:
            # existence groups
            groups = list(permutations(query))
            subsets = list(combinations(range(num_pegs), num_exist))

            var_groups = set()

            for g in groups:
                for s in subsets:
                    s = set(s)
                    var_group = tuple([
                        variables[i] == g[i] if i in s
                        else variables[i] != g[i]
                        for i in xrange(num_pegs)
                    ])
                    var_groups.add(var_group)
            
            var_groups = map(And, var_groups)

            constraints.append(Or(var_groups))
    
        # Constraints on positions of groups
        if num_match > 0:
            groups = index_subsets[num_match]
            xor_constraint = None
            for group in groups:
                group = set(group)
                var_group = [
                    variables[i] == query[i] if i in group
                    else variables[i] != query[i]
                    for i in xrange(num_pegs)
                ]
                if xor_constraint is None:
                    xor_constraint = And(var_group)
                else:
                    xor_constraint = Xor(xor_constraint, And(var_group))
            constraints.append(xor_constraint)
        else:
            var_group = [
                    variables[i] != query[i]
                    for i in xrange(num_pegs)
                ]
            constraints.append(And(var_group))


    s = Solver()
    s.add(constraints)
    if s.check() == sat:
        model = s.model()
        num = ''
        for v in variables:
            num += str(model[v])
        return num
    else:
        print 'UNSAT'


def test1():
    answers = ['2002', '0202', '2020']
    feedbacks = [([0, 0, 2, 2], 4, 2)]
    assert solve(feedbacks=feedbacks) in answers

def test2():
    answer = '5302'
    feedbacks = [([0, 0, 2, 2], 2, 1), ([1, 0, 3, 7], 2, 0), ([3, 7, 1, 2], 2, 1), ([7, 3, 0, 2], 3, 3), ([5, 5, 5, 5], 1, 1)]
    assert answer == solve(feedbacks=feedbacks, num_options=8)


if __name__ == '__main__':
    test1()
    test2()

