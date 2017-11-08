from z3 import Solver, Int, Bool, Or, Xor, And, If, Not, sat, is_true, is_false, unsat, unknown


def gen_subsets(length, num_indices=4, index_subsets=None):
    """Generates all subsets of indices of length length from num_indices"""
    nums = range(num_indices)
    if length == 1:
        subsets = [[num] for num in range(num_indices)]
        if index_subsets:
            index_subsets.append(subsets)
        return subsets
    smaller_subsets = gen_subsets(length-1, num_indices=num_indices, index_subsets=index_subsets)
    subsets = []
    for num in nums:
        for subset in smaller_subsets:
            s2 = set(subset)
            s2.add(num)
            if len(s2) == length and s2 not in subsets:
                subsets.append(s2)
    subsets = [list(s) for s in subsets]
    if index_subsets:
        index_subsets.append(subsets)
    return subsets


def solve(num_pegs=4, num_options=10, feedbacks=None):
    """Solves the problem given the list of feedback"""
    feedbacks = feedbacks or []

    # generate all possible permutations of subsets
    index_subsets =[[]]
    gen_subsets(num_pegs, index_subsets=index_subsets)

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
            ors = []
            for val in set(query):
                count = 0
                for v2 in query:
                    if val == v2:
                        count += 1

                groups = index_subsets[count]
                var_groups = []
                for group in groups:
                    var_group = [
                        variables[i] == val
                        for i in group
                    ]
                    var_groups.append(var_group)

                xors = And(var_groups[0])
                for v in var_groups[1:]:
                    xors = Xor(xors, And(v))
                ors.append(xors)
            constraints.append(Or(ors))
    
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
    answer = '1316'
    feedbacks = [([1, 3, 2, 4], 2, 2), ([1, 3, 1, 5], 3, 3)]
    assert answer == solve(feedbacks=feedbacks)

def test2():
    answer = '5302'
    feedbacks = [([0, 0, 2, 2], 4, 0)]#, ([1, 0, 3, 7], 2, 0), ([3, 7, 1, 2], 2, 1), ([7, 3, 0, 2], 3, 3), ([5, 3, 0, 2], 4, 4)]
    print solve(feedbacks=feedbacks, num_options=8)


if __name__ == '__main__':
    test2()


