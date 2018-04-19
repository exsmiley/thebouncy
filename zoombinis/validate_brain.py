from zoombinis import *
from baselines import *
from actor import *
import tqdm
import matplotlib.pyplot as plt


num_games = 1000

players = [MaxProbabilityPlayer(), MaxEntropyPlayer(), ActorPlayer(), ActorPipelinePlayer2()]
brain_names = ['brain', 'brain_pipelined', 'brain_entropy']
brains = {name: Brain('models/'+name) for name in brain_names}


entropy_scores = {brain: [[] for i in range(NUM_ZOOMBINIS+MAX_MISTAKES)] for brain in brain_names}


for i in tqdm.tqdm(range(num_games)):
    game = Game()
    for j, player in enumerate(players):
        game.reset()
        entropy_scores_int = player.brain_test(game, brains)
        
        for brain, scores in entropy_scores_int.items():
            for i in range(min(len(scores), NUM_ZOOMBINIS+MAX_MISTAKES)):
                entropy_scores[brain][i].append(scores[i])


all_scores = []
brain_names = []

for brain, scores in entropy_scores.items():
    for i in range(len(scores)):
        if len(scores[i]) == 0:
            scores[i] = scores[i-1]
        else:
            scores[i] = sum(scores[i])/len(scores[i])
    all_scores.append(scores)
    brain_names.append(brain)


for i, y in enumerate(all_scores):
    plt.plot(list(range(len(y))), y)

plt.title('Entropy Scores')
plt.legend(brain_names)
plt.show()
