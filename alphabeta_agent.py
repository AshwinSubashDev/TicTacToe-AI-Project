from typing import Tuple
import numpy as np
from tictactoe_game import TicTacToe, Player, GameResult


class AlphaBetaAgent:
    def __init__(self, name: str = "AlphaBeta"):
        self.name = name

    def select_action(self, game: TicTacToe) -> Tuple[int, int]:
        board = game.board.copy()
        current_player_val = game.current_player.value  # 1 for X, -1 for O

        best_score = float("-inf")
        best_move = None

        for (r, c) in self._get_valid_moves(board):
            board[r, c] = current_player_val
            score = self._alphabeta(
                board,
                alpha=float("-inf"),
                beta=float("inf"),
                is_maximizing=False,
                root_player=current_player_val,
            )
            board[r, c] = 0

            if best_move is None or score > best_score:
                best_score = score
                best_move = (r, c)

        if best_move is None:
            raise ValueError("AlphaBetaAgent: no valid move found")

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

    def _alphabeta(
        self,
        board: np.ndarray,
        alpha: float,
        beta: float,
        is_maximizing: bool,
        root_player: int,
    ) -> float:
        """
        Minimax with Alpha-Beta pruning.
        root_player: 1 (X) or -1 (O) – the player we are optimizing for.
        """
        terminal = self._check_terminal(board)
        if terminal is not None:
            if terminal == 0:
                return 0.0
            return 1.0 if terminal == root_player else -1.0

        current_player = root_player if is_maximizing else -root_player

        if is_maximizing:
            best_val = float("-inf")
            for (r, c) in self._get_valid_moves(board):
                board[r, c] = current_player
                val = self._alphabeta(board, alpha, beta, False, root_player)
                board[r, c] = 0
                best_val = max(best_val, val)
                alpha = max(alpha, best_val)
                if beta <= alpha:
                    break 
            return best_val
        else:
            best_val = float("inf")
            for (r, c) in self._get_valid_moves(board):
                board[r, c] = current_player
                val = self._alphabeta(board, alpha, beta, True, root_player)
                board[r, c] = 0
                best_val = min(best_val, val)
                beta = min(beta, best_val)
                if beta <= alpha:
                    break
            return best_val
