from copy import deepcopy
import numpy as np
from game import Game, Move, Player


class MinMax(Player):   # inherits player base class
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'Game', play_as) -> tuple[tuple[int, int], Move]:
        # uses minmax algorithm to make a move
        best_move, _ = self.minmax(game, depth=2, alpha=float('-inf'), beta=float('inf'),
                                   maximizing_player=(True if play_as == 'X' else False))
        return best_move

    def minmax(self, game: 'Game', depth: int, alpha: float, beta: float, maximizing_player: bool):
       # implements minmax algo with alpha-beta pruning to find best move
       # depth indicates how many moves ahead the algo should look
       # if depth is 0 or winner is found, evaluate board and return score

        if depth == 0 or game.check_winner() != -1:
            return None, self.evaluate_board(game)  # evaluate board to get a score

        legal_moves = self.get_legal_moves(game)    # get all possible moves
        best_move = None

        if maximizing_player:   # if maximizing player (x) then look for move with highest score
            max_eval = float('-inf')
            for move in legal_moves:
                new_game = deepcopy(game)
                new_game.move(move[0], move[1], game.get_current_player())   # simulate move
                _, eval_score = self.minmax(new_game, depth - 1, alpha, beta, False)    # recurse with decreasing depth
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                alpha = max(alpha, eval_score)  # update alpha
                if beta <= alpha:
                    break
            return best_move, max_eval
        else:   # turn of the minimizing player 0
            min_eval = float('inf')
            for move in legal_moves:
                new_game = deepcopy(game)
                new_game.move(move[0], move[1], game.get_current_player())   # simulate move
                _, eval_score = self.minmax(new_game, depth - 1, alpha, beta, True)     # recurse with decreasing depth
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return best_move, min_eval

    def get_legal_moves(self, game: 'Game') -> list[tuple[tuple[int, int], Move]]:
        # get all possible moves for the current player, iterating over every possible move and checking if it's valid
        legal_moves = []
        for row in range(5):
            for col in range(5):
                for slide in [Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT]:
                    state = deepcopy(game)
                    acceptable = state._Game__move((row, col), slide, game.get_current_player())    # check if valid move
                    if acceptable:  # if it is append to list
                        legal_moves.append(((row, col), slide))
        return legal_moves

    def evaluate_board(self, game: 'Game') -> float:
        current_player_id = game.get_current_player()
        opponent_player_id = 1 - current_player_id

        if game.check_winner() == current_player_id:
            return 100.0    # high score if the player has won, favourable state
        elif game.check_winner() == opponent_player_id:
            return -100.0   # negative score if opposite has won, unfavourable state
        else:
            # count pieces for each player and calculate difference, higher means better
            current_player_pieces = np.count_nonzero(game.get_board() == current_player_id)
            opponent_player_pieces = np.count_nonzero(game.get_board() == opponent_player_id)
            piece_difference = current_player_pieces - opponent_player_pieces

            # compute positional advantage based on the values assigned to each position
            position_score = 0.0
            for row in range(5):
                for col in range(5):
                    if game.get_board()[row, col] == current_player_id:
                        position_score += self.position_value(row, col)

            # compute score based on the potential to complete a line
            line_completion_score = self.line_completion_factor(game, current_player_id)

            # combine factors into a final score based on a weighted sum
            final_score = piece_difference + 0.1 * position_score + 0.2 * line_completion_score

            return final_score

    def line_completion_factor(self, game: 'Game', player_id: int) -> float:
        # to quantify the completion potential of the board
        line_score = 0.0

        # iterate over rows to evaluate completion potential
        for row in range(5):
            line_score += self.evaluate_line(game.get_board()[row, :], player_id)

        # iterate over cols to evaluate completion potential
        for col in range(5):
            line_score += self.evaluate_line(game.get_board()[:, col], player_id)

        # iterate over diag and secondary diag to evaluate completion potential
        line_score += self.evaluate_line(np.diag(game.get_board()), player_id)
        line_score += self.evaluate_line(np.diag(np.fliplr(game.get_board())), player_id)

        return line_score   # cumulative score of the line completion potential

    def evaluate_line(self, line: np.ndarray, player_id: int) -> float:
        # count the number of player's pieces in the line
        player_pieces = np.count_nonzero(line == player_id)

        # assign and return score based on player's pieces in the line
        if player_pieces == 4 and -1 in line:
            return 0.5  # higher score since line is easy to complete
        elif player_pieces == 3 and -1 in line:
            return 0.2  # moderate score for 3 pieces and 1 empty still
        else:
            return 0.0  # no significance

    def position_value(self, row: int, col: int) -> float:
        # each position is assigned a strategic value, center and adjacents are the highest
        position_values = [
            [1, 2, 3, 2, 1],
            [2, 4, 6, 4, 2],
            [3, 6, 8, 6, 3],
            [2, 4, 6, 4, 2],
            [1, 2, 3, 2, 1]
        ]
        return position_values[row][col]    # return the value of the position to prioritize important positions
