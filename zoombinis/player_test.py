from zoombinis import *
from baselines import *
from actor import *
import tqdm


num_games = 1000

players = [ActorPipelinePlayer(), MaxProbabilityPlayer(), ActorPlayer(), WinningBridgePlayer(), WinningBridgePlayerNoHack()]
# players = [MaxProbabilityPlayer(), MaxEntropyPlayer(), ActorPlayer(), ActorShapedPlayer(), ActorPipelinePlayer(), ActorPipelinePlayer2(), RandomFlipFlop(), RandomPlayer()]
wins = [0 for i in range(len(players))]
entropy_scores = [[] for i in range(len(players))]
scores = [[] for i in range(len(players))]
running_scores = [[] for i in range(len(players))]

for i in tqdm.tqdm(range(num_games)):
    game = Game()
    for j, player in enumerate(players):
        game.reset()
        won, entropy_score, score, running_score = player.play(game)
        if won:
            wins[j] += 1
        entropy_scores[j].append(entropy_score)
        scores[j].append(score)
        running_scores[j].append(running_score)

all_score_sums = []
running_scores2 = []
win_rates = []

for i in range(len(players)):
    print(players[i].__class__.__name__)
    print('win rate:', wins[i]/num_games)
    win_rates.append(wins[i]/num_games)
    # print(scores[i])
    print('avg score:', sum(scores[i])/num_games)
    score_sums = []
    running_sums = []
    for j in range(NUM_ZOOMBINIS+MAX_MISTAKES):
        score_sum = 0
        running_sum = 0
        num = 0
        for score in entropy_scores[i]:
            if j < len(score):
                score_sum += score[j]
                num += 1
        for score in running_scores[i]:
            if j < len(score):
                running_sum += score[j]
            else:
                running_sum += score[-1]

        if num > 0:
            # print('avg entropy score at {} steps: {}'.format(j+1, score_sum/num))
            score_sums.append(score_sum/num)
        else:
            pass
        running_sums.append(running_sum/len(entropy_scores[i]))
    print()
    all_score_sums.append(score_sums)
    running_scores2.append(running_sums)

import matplotlib.pyplot as plt


for i, y in enumerate(all_score_sums):
    plt.plot(list(range(len(y))), y)

plt.title('Entropy Scores')
plt.legend([players[i].__class__.__name__ for i in range(len(players))])
plt.show()

plt.gcf().subplots_adjust(left=0.3)
plt.barh([players[i].__class__.__name__ for i in range(len(players))], win_rates)
plt.title('Win Rates')
plt.show()

for i, y in enumerate(running_scores2):
    plt.plot(list(range(len(y))), y)

plt.title('Total Scores')
plt.legend([players[i].__class__.__name__ for i in range(len(players))])
plt.show()

plt.gcf().subplots_adjust(left=0.3)
plt.title('Distribution of Scores')
plt.boxplot(scores, vert=False, labels=[players[i].__class__.__name__ for i in range(len(players))])

plt.show()