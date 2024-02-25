import pickle
import random
from collections import defaultdict
from copy import deepcopy
from itertools import product

import numpy as np
from tqdm.auto import tqdm

from MinMax import MinMax
from game import Game, Player, Move

# cartesian product of the tuple and the range, first [] is for first and last row
# second [] is for the external position of the other rows, the concantenates the two lists
POSITION = [pos for pos in product((0, 4), range(5))] + [pos for pos in product((1, 2, 3), (0, 4))]
MOVE = [Move.BOTTOM, Move.TOP, Move.LEFT, Move.RIGHT]
ACTION = [a for a in product(POSITION, MOVE)]  # tuple of position and move
COUNTER = 0


class RLPlayer(Player):
    def __init__(self) -> None:
        super().__init__()
        self.agent = None
        self.env = None
        self.minmax = MinMax()
        self.player_symbol = 'X'  # default

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        player_id = 0 if self.player_symbol == 'X' else 1  # determines player id
        state = tuple(self.env.game.get_board().flatten().tolist())  # flatten board to a tuple compatible with state
        self.env.current_player = self.player_symbol  # set current player
        available_moves = self.env.available_moves()
        imminent_win, action = one_move_to_win(self.env, player_id)  # is one move away from winning?
        if not imminent_win:  # if there is no action to win in 1 move
            action = self.agent.choose_action(state, available_moves, play_as=self.player_symbol, playing=True)  # RL
            if action is None:  # if no actions are found, pick with minmax
                action = self.minmax.make_move(game, self.player_symbol)
        return action


# manages the game state and interactions
class Environment:
    def __init__(self):
        self.game = Game()
        self.current_player = 'X'

    def print_board(self):
        self.game.print()

    def available_moves(self, state=None):
        # determine available moves
        player_id = 0 if self.current_player == 'X' else 1
        # list with only the valid moves, a is a tuple: ((row, col), move)
        moves = [act for act in ACTION if
                    self.game.valid(from_pos=(act[0][0], act[0][1]), slide=act[1], player_id=player_id)]
        random.shuffle(moves)
        return moves

    def make_move(self, action):
        # execute move
        player_id = 0 if self.current_player == 'X' else 1
        from_pos = (action[0][0], action[0][1])  # action is a tuple: ((row, col), move)
        slide = action[1]  # a is a tuple: ((row, col), move)
        self.game.move(from_pos=from_pos, slide=slide, player_id=player_id)  # call Game's public method
        self.current_player = 'O' if self.current_player == 'X' else 'X'  # switch player

    def check_winner(self):
        # check if there is a winner in the current state with game's method
        winner_id = self.game.check_winner()
        if winner_id == 0:
            return 'X'
        elif winner_id == 1:
            return 'O'
        else:
            return None

    def game_over(self):
        # determines if game's over
        global COUNTER
        COUNTER += 1

        # resets counter and eng game after 100 moves with no winner
        if COUNTER == 100:
            COUNTER = 0
            return True
        # or end game if there is a winner
        return self.check_winner() is not None

    def reset(self):
        '''Resets board'''
        self.game._board = np.ones((5, 5), dtype=np.uint8) * -1
        self.current_player = 'X'


class QMethods:
    def __init__(self, epsilon, alpha=0.5, gamma=0.1):
        self.epsilon = epsilon  # exploration rate
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.q_table = defaultdict(float)  # action - value pairs, defaultdict better in case of empty values
        self.env = None
        self.usefulness = 0
        self.total = 0  # total actions

    def get_q_value(self, state, action):
        # retrieve q-value for a given state-action pair, default 0 if not found
        return self.q_table.get((state, action), 0.0)

    def update_q_value(self, state, action, reward, next_state, player):
        # updates q-value for a given state-action
        old_value = self.get_q_value(state, action)  # defaults to 0
        # computes rewards for all possible actions from the next state
        future_rewards = [self.get_q_value(next_state, a) for a in self.env.available_moves(next_state)]
        # maximize for X, highest possible reward, minimize for O (choose the action for the worst outcome for the opponent)
        best_future_reward = max(future_rewards) if player == 'X' else min(future_rewards, default=0)
        # given formula
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * best_future_reward)
        self.q_table[(state, action)] = new_value

    def choose_action(self, state, available_moves, play_as=None, playing=False):
        if not playing:  # training mode
            if random.uniform(0, 1) < self.epsilon.get():  # either explore or exploit based on epsilon
                return random.choice(available_moves)  # explore
            else:  # exploit
                if play_as == 'X':
                    return max(available_moves, key=lambda a: self.get_q_value(state, a))
                else:  # playing as O you choose least beneficial actions for X
                    return min(available_moves, key=lambda a: self.get_q_value(state, a))
        else:  # playing mode, no random exploration
            if play_as == 'X':
                a = max(available_moves, key=lambda a: self.get_q_value(state, a))
                self.total += 1
                if self.get_q_value(state, a) > 0:
                    self.usefulness += 1
                else:
                    return None  # no useful actions found
                return a
            else:
                a = min(available_moves, key=lambda a: self.get_q_value(state, a))
                self.total += 1
                if self.get_q_value(state, a) < 0:  # negative q value is good for O
                    self.usefulness += 1
                else:
                    return None
                return a


def save_q_table(agent, filename="qtable1.pkl"):
    # save the trained Q-table to a file
    with open(filename, 'wb') as f:
        pickle.dump(agent.q_table, f)
    print(f"Q-table saved to {filename}.")


def train(games):
    # trains the RL agent over a certain number of games
    epsilon = EpsilonScheduler(low=0.1, high=1, num_round=games)  # adjust exploration rate over time
    agent = QMethods(epsilon=epsilon)
    env = Environment()
    agent.env = env  # sets environment for the agent

    for _ in tqdm(range(games)):
        # training loop, resets environment at each episode
        env.reset()
        state = tuple(env.game.get_board().flatten().tolist())  # flatten board to a tuple

        while not env.game_over():  # either n moves or someone wins
            available_moves = env.available_moves()
            player_sym = env.current_player  # get current player
            player_id = 0 if env.current_player == 'X' else 1
            imminent_win, action = one_move_to_win(env, player_id)  # check if I can win with 1 move
            if not imminent_win:  # otherwise choose an action
                action = agent.choose_action(state, available_moves, play_as=player_sym, playing=False)
            # to improve efficiency
            augmented_states, augmented_actions = augment_board(deepcopy(env.game.get_board()),
                                                                deepcopy(action))
            env.make_move(action)  # execute action
            next_state = tuple(env.game.get_board().flatten().tolist())  # new state after the move

            # determines reward for the action if player wins
            if env.check_winner() == 'X':
                reward = 100
            elif env.check_winner() == 'O':
                reward = -100
            else:  # reward for non-winning actions
                reward = intermediate_reward(env, player_id)

            # update q-values
            for state, action in zip(augmented_states, augmented_actions):
                agent.update_q_value(state, action, reward, next_state, player=player_sym)

            state = next_state  # transition for next iteration

        agent.epsilon.update()  # update with method

    save_q_table(agent, "qtable1.pkl")

    return agent


class EpsilonScheduler():
    def __init__(self, low, high, num_round):
        self.low = low  # min eps
        self.high = high  # max eps
        self.num_round = num_round
        self.step = (high - low) / num_round  # step size to decrease eps each round
        self.counter = 0  # to track rounds

    def get(self):
        # calculate current eps value based on the counter and step, if lower than min return threshold
        return_val = self.high - self.counter * self.step
        return return_val if return_val > self.low else self.low

    def update(self):
        self.counter += 1  # update counter


def intermediate_reward(env, player_id):
    # calculates reward based on pieces in rows/col/diag
    board = env.game.get_board()
    reward = 0

    # add to reward based on pieces in each column
    for col in range(5):
        r = np.sum(board[:, col] == player_id) / 5
        reward += r

    # add to reward based on pieces in each row
    for row in range(5):
        r = np.sum(board[row, :] == player_id) / 5
        reward += r

    reward += np.sum(np.diag(board) == player_id) / 5  # check  diag
    reward += np.sum(np.diag(np.flip(board, axis=1)) == player_id) / 5  # and opposite diag

    # normalize reward and invert for player 0
    return reward / 12 if player_id == 0 else - reward / 12


def augment_board(board, action):
    # generate augmented states and actions by rotating and flipping the board
    augmented_states = []
    augmented_actions = []

    for i in range(0, 4):  # generate rotations and flips
        rotated_board = np.rot90(board, k=i, axes=(0, 1))  # rotate board of 90 degrees
        flipped_board = np.flip(rotated_board, axis=0)  # flip the board after rotating

        rotated_state = tuple(rotated_board.flatten().tolist())  # convert to tuple
        flipped_state = tuple(flipped_board.flatten().tolist())

        from_pos = (action[0][0], action[0][1])
        slide = action[1]
        rotated_pos, flipped_pos = rotate_and_flip(from_pos, i)  # compute new positions after rotation and flip
        rotated_slide = Move._value2member_map_[(slide.value + 1) % 4]  # adjust slide direction

        if slide == Move.LEFT or slide == Move.RIGHT:  # maintain direction if moving lef/right  for flipped actions
            flipped_slide = slide
        elif slide == Move.TOP:  # but invert if moving up/down
            flipped_slide = Move.BOTTOM
        else:
            flipped_slide = Move.TOP

        # create new action tuple for the rotated and flipped versions
        rotated_action = ((rotated_pos[0], rotated_pos[1]), rotated_slide)
        flipped_action = ((flipped_pos[0], flipped_pos[1]), flipped_slide)

        augmented_states.append(rotated_state)
        augmented_states.append(flipped_state)
        augmented_actions.append(rotated_action)
        augmented_actions.append(flipped_action)

    return augmented_states, augmented_actions


def rotate_and_flip(pos, i):
    # adjust a position based on rotation and flip (needed for consistency after modifications)
    x, y = pos[0] - 2, pos[1] - 2  # change position to have the center of the board as the origin
    rotated_xy = None
    flipped_xy = None

    if i == 0:  # no rotations = no change
        rotated_xy = (x, y)
        flipped_xy = (x, -y)
    elif i == 1:  # 90 degrees rotation
        rotated_xy = (-y, x)
        flipped_xy = (-y, -x)
    elif i == 2:  # 180 degrees rotation
        rotated_xy = (-x, -y)
        flipped_xy = (-x, y)
    elif i == 3:  # 270 degrees rotation
        rotated_xy = (y, -x)
        flipped_xy = (y, x)

    # translate back from centered coordinates
    rotated_pos = (rotated_xy[0] + 2, rotated_xy[1] + 2)
    flipped_pos = (flipped_xy[0] + 2, flipped_xy[1] + 2)

    return rotated_pos, flipped_pos


def one_move_to_win(env, player_id):
    board = env.game.get_board()

    for col in range(5):  # check col for win conditions
        if np.sum(board[:, col] == player_id) == 4:  # if there are 4 pieces
            row = np.argwhere(board[:, col] != player_id).ravel()[0]  # find empty spot, ravel flattens
            possible, action = horizontal_slide(row, col, board, player_id)  # check if horiz slide can win
            if possible:  # if you can return the action
                return possible, action

    for row in range(5):  # check row for win conditions
        if np.sum(board[row, :] == player_id) == 4:  # if there are 4 pieces
            col = np.argwhere(board[row, :] != player_id).ravel()[0]  # find empty spot
            possible, action = vertical_slide(row, col, board, player_id)  # check if vert slide can win
            if possible:  # if you can return the action
                return possible, action

    if np.sum(np.diag(board) == player_id) == 4:  # check diag for win conditions if there are 4 pieces
        diag_pos = np.argwhere(np.diag(board) != player_id).ravel()[0]  # find empty spot
        possible, action = horizontal_slide(diag_pos, diag_pos, board, player_id)  # check if horiz slide can win
        if possible:  # if you can return the action
            return possible, action
        possible, action = vertical_slide(diag_pos, diag_pos, board, player_id)  # check if vert slide can win
        if possible:  # if you can return the action
            return possible, action

    if np.sum(np.diag(np.flip(board, axis=1)) == player_id) == 4:  # same for opposite diagonal
        diag_pos = np.argwhere(np.diag(np.flip(board, axis=1)) != player_id).ravel()[0]
        possible, action = horizontal_slide(diag_pos, 4 - diag_pos, board, player_id)
        if possible:
            return possible, action
        possible, action = vertical_slide(diag_pos, 4 - diag_pos, board, player_id)
        if possible:
            return possible, action

    return False, None


def horizontal_slide(row, col, board, player_id):
    # check for winning move when missing piece is in col
    if col == 0:
        if row == 0 or row == 4:  # corners
            # find column where a move is possible
            selected = np.argwhere(np.logical_or(board[row, :] == player_id, board[row, :] == -1)).ravel()
            selected = selected[selected > col]  # column to the right of the current one

            if len(selected) > 0:  # if a valid move is found return it
                selected = selected[0]
                return True, ((selected, row), Move.LEFT)

        else:  # if not corners in first col, check last col if valid action
            if board[row][4] == player_id or board[row][4] == -1:
                return True, ((4, row), Move.LEFT)

    if col == 4:
        if row == 0 or row == 4:  # corners
            # find column where a move is possible
            selected = np.argwhere(np.logical_or(board[row, :] == player_id, board[row, :] == -1)).ravel()
            selected = selected[selected < col]  # column to the left of the current one

            if len(selected) > 0:  # if a valid move is found return it
                selected = selected[0]
                return True, ((selected, row), Move.RIGHT)

        else:  # if not corners in last col, check first col if valid action
            if board[row][0] == player_id or board[row][0] == -1:
                return True, ((0, row), Move.RIGHT)

    if col > 0:  # check potential moves between first and last
        # if the piece to the left is of the player and there is a valid move to the right
        if board[row, col - 1] == player_id and np.any(
                np.logical_or(board[row, col:] == player_id, board[row, col:] == -1)):
            # find column where a move is possible
            selected = np.argwhere(np.logical_or(board[row, :] == player_id, board[row, :] == -1)).ravel()
            selected = selected[selected >= col][0]  # first valid to the right

            if row != 0 and row != 4:
                if selected == 4:
                    return True, ((selected, row), Move.LEFT)
            else:
                return True, ((selected, row), Move.LEFT)

    if col < 4:  # check potential moves between first and last
        # if the piece to the right is of the player and there is a valid move to the left
        if board[row, col + 1] == player_id and np.any(
                np.logical_or(board[row, :col] == player_id, board[row, :col] == -1)):
            # find column where a move is possible
            selected = np.argwhere(np.logical_or(board[row, :] == player_id, board[row, :] == -1)).ravel()
            selected = selected[selected <= col][0]  # first valid to the left

            if row != 0 and row != 4:
                if selected == 0:
                    return True, ((selected, row), Move.RIGHT)
            else:
                return True, ((selected, row), Move.RIGHT)

    # if no horiz slide can lead to win return nothing
    return False, None


def vertical_slide(row, col, board, player_id):
    if row == 0:  # check wining move when first row
        if col == 0 or col == 4:  # corners
            # find row where a move is possible
            selected = np.argwhere(np.logical_or(board[:, col] == player_id, board[:, col] == -1)).ravel()
            selected = selected[selected > row]  # rows below the current

            if len(selected) > 0:  # if a move is found return it
                selected = selected[0]
                return True, ((col, selected), Move.TOP)

        else:  # for non corners check if bottom row is valid
            if board[4][col] == player_id or board[4][col] == -1:
                return True, ((col, 4), Move.TOP)

    if row == 4:  # check winning for last row
        if col == 0 or col == 4:  # corners
            # find row where a move is possible
            selected = np.argwhere(np.logical_or(board[:, col] == player_id, board[:, col] == -1)).ravel()
            selected = selected[selected < row]  # rows above the current

            if len(selected) > 0:  # if a move is found return it
                selected = selected[0]
                return True, ((col, selected), Move.BOTTOM)

        else:  # for non corners check if top row is valid
            if board[0][col] == player_id or board[0][col] == -1:
                return True, ((col, 0), Move.BOTTOM)

    if row > 0:  # check between first and last
        # if piece above is of the player and there is a valid move down
        if board[row - 1, col] == player_id and np.any(
                np.logical_or(board[row:, col] == player_id, board[row:, col] == -1)):
            # find row where a move is possible
            selected = np.argwhere(np.logical_or(board[:, col] == player_id, board[:, col] == -1)).ravel()
            selected = selected[selected >= row][0]

            if col != 0 and col != 4:
                if selected == 4:
                    return True, ((col, selected), Move.TOP)
            else:
                return True, ((col, selected), Move.TOP)

    if row < 4:
        # if piece below is of the player and there is a valid move up
        if board[row + 1, col] == player_id and np.any(
                np.logical_or(board[:row, col] == player_id, board[:row, col] == -1)):
            # find row where a move is possible
            selected = np.argwhere(np.logical_or(board[:, col] == player_id, board[:, col] == -1)).ravel()
            selected = selected[selected <= row][0]

            if col != 0 and col != 4:
                if selected == 0:
                    return True, ((col, selected), Move.BOTTOM)
            else:
                return True, ((col, selected), Move.BOTTOM)

    # if nothing found
    return False, None
