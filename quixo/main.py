import os
import pickle
import random
import zipfile

from RLPlayer import RLPlayer, Environment, QMethods, train
from game import Game, Move, Player

# seed_value = 42
# os.environ['PYTHONHASHSEED'] = str(seed_value)
# random.seed(seed_value)
# np.random.seed(seed_value)


class RandomPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        # select a random position and move
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        return from_pos, move

player1 = RLPlayer()
player1.agent = QMethods(epsilon=None)
player2 = RandomPlayer()

with zipfile.ZipFile('qtable0.zip', 'r') as zip_ref:  # extract a pre-trained q table for RL Player
    zip_ref.extractall()

with open('qtable0.pkl', 'rb') as f:  # load the pre-trained q-table for the RL player
    player1.agent.q_table = pickle.load(f)


def play_rounds(player1, player2, num_rounds=100, player1_symbol='X'):
    # plays a number of rounds between players tracking wins
    win_counter = 0
    player1.player_symbol = player1_symbol  # sets the symbol for players

    for round_num in range(num_rounds):
        game = Game()
        env = Environment()
        env.game = game
        player1.env = env

        # determine winner of each round and update count
        if player1_symbol == 'X':
            winner = game.play(player1, player2)
        else:
            winner = game.play(player2, player1)
        if winner == 0:
            win_counter += 1  # if player 1 wins
        print(f'{round_num}) winner_id: {winner}')

    print(f"Win percentage: {win_counter/num_rounds}")
    try:
        print(f"Agent usefulness: {player1.agent.usefulness / player1.agent.total}")
    except ZeroDivisionError:  # should never be called
        print("No total actions recorded for agent.")


# Player 1 plays as 'X'
play_rounds(player1, player2, 100, 'X')

# Player 1 plays as 'O'
play_rounds(player1, player2, 100, 'O')
