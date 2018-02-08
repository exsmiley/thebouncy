
import numpy as np
g = []

i = 0
while True:
    fname = 'small3_6/lu{}'.format(i)
    try:
        with open(fname) as f:
            count = 0.
            total_guesses = 0
            total_correct = 0.
            for line in f:
                count += 1
                line = line.split('[')
                ans = ''.join(line[0].split('(')[1][:-3].split(', '))
                guesses = set(map(lambda x: x[1:].replace(', ', ''), line[1][:-3].split('), ')))
                # if ans in guesses:
                total_guesses += len(guesses)
                total_correct += 1

            g.append(total_guesses/total_correct)
    except:
        break
    i += 1

g = np.array(g)
print g, len(g)
print np.argmin(g), min(g)
print np.mean(g)
