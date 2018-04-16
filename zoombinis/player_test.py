from zoombinis import *
from baselines import *
from actor import *
import tqdm


num_games = 1000

players = [MaxProbabilityPlayer(), MaxEntropyPlayer(), ActorPlayer(), ActorShapedPlayer(), ActorPipelinePlayer(), RandomFlipFlop(), RandomPlayer()]
wins = [0 for i in range(len(players))]
scores = [[] for i in range(len(players))]
actual_scores = [[] for i in range(len(players))]

for i in tqdm.tqdm(range(num_games)):
    game = Game()
    for j, player in enumerate(players):
        game.reset()
        # print(game)
        won, score, actual_score = player.play(game)
        if won:
            wins[j] += 1
        scores[j].append(score)
        actual_scores[j].append(actual_score)
        # print('total mistakes:', game.mistakes)

all_score_sums = []

for i in range(len(players)):
    print(players[i].__class__.__name__)
    print('win rate:', wins[i]/num_games)
    # print(actual_scores[i])
    print('avg score:', sum(actual_scores[i])/num_games)
    score_sums = []
    for j in range(NUM_ZOOMBINIS+MAX_MISTAKES):
        score_sum = 0
        num = 0
        for score in scores[i]:
            if j < len(score):
                score_sum += score[j]
                num += 1
        if num > 0:
            # print('avg entropy score at {} steps: {}'.format(j+1, score_sum/num))
            score_sums.append(score_sum/num)
        else:
            score_sums.append(score_sums[-1])
    print()
    all_score_sums.append(score_sums)

import matplotlib.pyplot as plt


x = list(range(NUM_ZOOMBINIS+MAX_MISTAKES))
for i, y in enumerate(all_score_sums):
    plt.plot(x, y)

plt.legend([players[i].__class__.__name__ for i in range(len(players))])
plt.show()