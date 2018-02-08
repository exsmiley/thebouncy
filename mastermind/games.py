import time
import tqdm
from player import PlayerRunner
from random_player import *
from baselines import *
from mastermind import Mastermind, generate_all_targets, NUM_PEGS, NUM_OPTIONS
import sys


def play_games(players, num_games=5, do_all=False):
    '''assumes no duplicate player types'''
    runners = map(PlayerRunner, players)

    # tuple of (num_wins, total_num_attempts, total_time)
    results = {runner.player_type: (0., 0., 0.) for runner in runners}

    if do_all:
        num_games = NUM_OPTIONS ** NUM_PEGS
        possible_games = generate_all_targets(NUM_PEGS, NUM_OPTIONS)

    for i in tqdm.tqdm(xrange(num_games)):
        if do_all:
            target = list(possible_games.next())
            game = Mastermind(target=target)
        else:
            # play a random game
            game = Mastermind()
        print 'Game {}: {}'.format(i, game.target)
        sys.stderr.write('Doing Game {}: {}\n\n'.format(i, game.target))
        for runner in runners:
            start = time.time()
            won, attempts = runner.play(game)
            (num_wins, total_num_attempts, total_time_taken) = results[runner.player_type]
            time_taken = (time.time()-start)
            total_time_taken += time_taken
            if won:
                num_wins += 1
                total_num_attempts += len(attempts)
                print '{} won in {}s with {} moves'.format(runner.player_type, time_taken, len(attempts))
            else:
                print '{} lost in {}s after {} moves'.format(runner.player_type, time_taken, len(attempts))
            results[runner.player_type] = (num_wins, total_num_attempts, total_time_taken)
        print

    sys.stderr.write('Calculating stats... ')
    print 'Final Stats:'
    for name, (num_wins, total_num_attempts, total_time_taken) in results.iteritems():
        print name
        print 'win percent: {}%'.format(num_wins/num_games*100)
        if num_wins > 0:    
            print 'avg num attempts to win:', total_num_attempts/num_wins
        else:
            print 'never won'
        print 'avg time per game: {}s'.format(total_time_taken/num_games)
        print
    sys.stderr.write('Done!\n')


if __name__ == '__main__':

    play_games([
            FiveGuessPlayer(),
            MaxEntropyPlayer(),
            MaxPartsPlayer(),
            SwaszekPlayer()
        ],
        do_all=False, num_games=1
    )
    # players = [SolverPlayer(), SwaszekPlayer()]
    # play_games(players, do_all=False, num_games=1000)

