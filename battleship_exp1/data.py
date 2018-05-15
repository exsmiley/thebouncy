from battleship import *
import tqdm
import pickle

if __name__ == "__main__":
    oracle_train_games = []
    train_games = []
    test_games = []
    for i in tqdm.tqdm(range(1001)):
        oracle_train_games.append(GameEnv())
    for i in tqdm.tqdm(range(10001)):
        train_games.append(GameEnv())
    for j in tqdm.tqdm(range(1000)):
        test_games.append(GameEnv())

    pickle.dump( (oracle_train_games, train_games, test_games) , open( "games.p", "wb" ) )
