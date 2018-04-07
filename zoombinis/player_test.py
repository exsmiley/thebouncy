from zoombinis import *
from entropy_player import EntropyPlayer
from actor import ActorPlayer
import tqdm


num_games = 1000

players = [EntropyPlayer(), ActorPlayer()]
wins = [0 for i in range(len(players))]
scores = [[] for i in range(len(players))]

for i in tqdm.tqdm(range(num_games)):
    game = Game()
    for j, player in enumerate(players):
        game.reset()
        won, score = player.play(game)
        if won:
            wins[j] += 1
        scores[j].append(score)

for i in range(len(players)):
    print(players[i].__class__.__name__)
    print('win rate:', wins[i]/num_games)
    print('avg score:', sum(scores[i])/len(scores[i]))
    print()