import matplotlib.pyplot as plt


def read_data(fname):
    players = {}
    with open(fname) as f:
        for line in f:
            if line[:5] == 'Final':
                break
            elif len(line) > 1 and line[:4] != 'Game':
                line = line.split()
                player = line[0]
                t = float(line[3][:-1])
                moves = int(line[5])
                if player in players:
                    players[player][0].append(t)
                    players[player][1].append(moves)
                else:
                    players[player] = [[t], [moves]]
    return players


def get_plot(fname, plot_time=True, plot_moves=True):
    data = read_data(fname)
    player_names = data.keys()
    print player_names
    time_data = [data[name][0] for name in player_names]
    move_data = [data[name][1] for name in player_names]

    plt.boxplot(time_data)
    plt.show()

if __name__ == '__main__':
    get_plot('all_4_10')

