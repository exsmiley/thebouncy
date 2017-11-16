from z3 import Solver, Int, Bool, Or, Xor, And, If, Not, sat, is_true, is_false, unsat, unknown, Sum
from itertools import permutations, combinations


def gen_subsets(length, num_indices=4):
    S = range(num_indices)
    return list(combinations(S, length))


def solve1(num_pegs=4, num_options=10, feedbacks=None):
    """Solves the problem given the list of feedbacks"""
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
        nums = []
        for v in variables:
            nums.append(int(model[v].as_long()))
        return nums
    else:
        print 'UNSAT'


def solve(num_pegs=4, num_options=10, feedbacks=None):
    """Solves the problem given the list of feedbacks"""
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
            # present groups
            # ifs = []
            # for k in xrange(num_options):
            #     for i in xrange(len(query)):
            #         ifs.append(If(variables[i] == k, 1, 0))
            # present_constraint = Sum(ifs) == num_present
            # constraints.append(present_constraint)
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
    assert ''.join(map(str, solve1(feedbacks=feedbacks))) in answers


def test2():
    answers = ['5302', '7502']
    feedbacks = [([0, 0, 2, 2], 2, 1), ([1, 0, 3, 7], 2, 0), ([3, 7, 1, 2], 2, 1), ([7, 3, 0, 2], 3, 3), ([5, 5, 5, 5], 1, 1)]
    assert ''.join(map(str, solve(feedbacks=feedbacks, num_options=8))) in answers


def test3():
    answers = ['4005', '4015', '4025', '4035', '4045']
    feedbacks = [([4, 5, 6, 7], 2, 1), ([4, 0, 6, 7], 2, 2), ([4, 0, 5, 8], 3, 2), ([4, 0, 9, 5], 3, 3)]
    print ''.join(map(str, solve1(feedbacks=feedbacks))), answers


if __name__ == '__main__':
    test1()
    test2()
    test3()
    # variables = []
    # constraints = []

    # for i in xrange(4):
    #     v = Int('peg_%d' % i)
    #     variables.append(v)

#     Game 0: [3, 0, 5, 0]

# made 1 guesses with 276/1296 remaining
# [5, 0, 0, 3] (4, 1)
# made 2 guesses with 1/1296 remaining
# [3, 0, 5, 0] (4, 4)

    # derps = [([1, 5, 2, 5], 1, 3), ([5, 0, 0, 3], 4, 1)]

    # for derp, p, m in derps:
    #     ifs = []
    #     for i in xrange(4):
    #         ifs.append(If(variables[i] == derp[i], 1, 0))
    #     match_constraint = Sum(ifs) == m
    #     print match_constraint
    #     constraints.append(match_constraint)

    # s = Solver()
    # s.add(constraints)
    # if s.check() == sat:
    #     print s.model()



