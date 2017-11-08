import time
from player import PlayerRunner
from random_player import *
from game import Mastermind


def play_games(players, num_games=5):
    '''assumes no duplicate player types'''
    runners = map(PlayerRunner, players)

    # tuple of (num_wins, total_num_attempts)
    results = {runner.player_type: (0., 0., 0.) for runner in runners}

    for _ in xrange(num_games):
        # play a random game
        game = Mastermind()
        for runner in runners:
            start = time.time()
            won, attempts = runner.play(game)
            (num_wins, total_num_attempts, time_taken) = results[runner.player_type]
            if won:
                num_wins += 1
                total_num_attempts += len(attempts)
            time_taken += (time.time()-start)
            results[runner.player_type] = (num_wins, total_num_attempts, time_taken)

    for name, (num_wins, total_num_attempts, time_taken) in results.iteritems():
        print name
        print 'win percent: {}%'.format(num_wins/num_games*100)
        if num_wins > 0:    
            print 'avg num attempts to win:', total_num_attempts/num_wins
        else:
            print 'never won'
        print 'avg time per game: {}s'.format(time_taken/num_games)
        print

play_games([RandomPlayer(), RandomPlayerSolver()], num_games=2)