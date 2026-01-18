from typing import Tuple
import numpy as np

from tictactoe_game import TicTacToe, Player, GameResult


class MinimaxAgent:
    """Perfect-play Tic-Tac-Toe agent using the Minimax algorithm.
    Searches the full game tree (no pruning here) and chooses the move that
    maximizes its chance of winning (and at worst forces a draw).
    """

    def __init__(self, name: str = "MinimaxAgent"):
        self.name = name

    def select_action(self, game: TicTacToe) -> Tuple[int, int]:
        """
        Choose the best move for the current player in the given game.
        """
        board = game.board.copy()
        current_player_val = game.current_player.value  # 1 for X, -1 for O

        best_score = float("-inf")
        best_move = None

        for (r, c) in self._get_valid_moves(board):
            board[r, c] = current_player_val
            score = self._minimax(board, is_maximizing=False,
                                  root_player=current_player_val)
            board[r, c] = 0

            if score > best_score or best_move is None:
                best_score = score
                best_move = (r, c)

        if best_move is None:
            raise ValueError("MinimaxAgent: no valid move found")

        return best_move

    def _get_valid_moves(self, board: np.ndarray):
        return [(i, j) for i in range(3) for j in range(3) if board[i, j] == 0]

    def _check_terminal(self, board: np.ndarray):
        """
        Returns:
            1  if X has won
            -1 if O has won
            0  if draw
            None if game is still in progress
        """
        for i in range(3):
            s = board[i, :].sum()
            if s == 3:
                return 1
            if s == -3:
                return -1

        for j in range(3):
            s = board[:, j].sum()
            if s == 3:
                return 1
            if s == -3:
                return -1

        diag1 = board.trace()
        diag2 = np.fliplr(board).trace()
        if diag1 == 3 or diag2 == 3:
            return 1
        if diag1 == -3 or diag2 == -3:
            return -1

        if not self._get_valid_moves(board):
            return 0

        return None 

    def _minimax(self, board: np.ndarray, is_maximizing: bool,
                 root_player: int) -> float:

        terminal = self._check_terminal(board)
        if terminal is not None:
            # terminal is 1 (X win), -1 (O win) or 0 (draw)
            if terminal == 0:
                return 0.0
            # +1 if root player wins, -1 if root player loses
            return 1.0 if terminal == root_player else -1.0

        current_player = root_player if is_maximizing else -root_player

        if is_maximizing:
            best_val = float("-inf")
            for (r, c) in self._get_valid_moves(board):
                board[r, c] = current_player
                val = self._minimax(board, False, root_player)
                board[r, c] = 0
                best_val = max(best_val, val)
            return best_val
        else:
            best_val = float("inf")
            for (r, c) in self._get_valid_moves(board):
                board[r, c] = current_player
                val = self._minimax(board, True, root_player)
                board[r, c] = 0
                best_val = min(best_val, val)
            return best_val
