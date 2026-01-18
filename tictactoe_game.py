import numpy as np
import random
from enum import Enum
from typing import List, Tuple, Optional


class Player(Enum):
    """ Enum representing the two players """
    X = 1
    O = -1
    EMPTY = 0

class GameResult(Enum):
    """ Enum representing the possible game outcomes """
    X_WIN = 1
    O_WIN = -1
    DRAW = 0
    IN_PROGRESS = None

class TicTacToe:
    """
    Tic-Tac-Toe game environment
    State Representation:
        - A 3x3 numpy array where:
            - 1 represents Player X's move
            - -1 represents Player O's move
            - 0 represents an empty cell
    """

    def __init__(self):
        """ Initialize empty 3x3 board and set current player to Player X """
        self.board = np.zeros((3,3), dtype = int)
        self.current_player = Player.X
        self.move_history = []

    def reset(self) -> np.ndarray:
        """ Resets the game to the initial state """
        self.board = np.zeros((3,3), dtype = int)
        self.current_player = Player.X
        self.move_history = []
        return self.get_state()
    
    def get_state(self) -> np.ndarray:
        """ Returns the current state of the board """
        return self.board.copy()
    
    def get_valid_actions(self) -> List[Tuple[int, int]]:
        """ Returns a list of valid actions (empty cells)
            Returns:
               - List of (row, col) tuples representing valid moves
        """
        valid_actions = []
        for i in range(3):
            for j in range(3):
                if self.board[i, j] == Player.EMPTY.value:
                    valid_actions.append((i, j))
        return valid_actions
    
    def is_valid_action(self, action: Tuple[int, int]) -> bool:
        """ Checks if the given action is valid """
        row, col = action
        if row < 0 or row > 2 or col < 0 or col > 2:
            return False
        return self.board[row, col] == Player.EMPTY.value
    
    def make_move(self, action: Tuple[int, int]) -> Tuple[np.ndarray, GameResult]:
        """
        Make a move on the board
        Args:
            action: (row, col) tuple
        Returns:
            Tuple of (new_state, game_result)
        Raises:
            ValueError: If the action is invalid
        """
        if not self.is_valid_action(action):
            raise ValueError(f"Invalid move: {action}")
        
        row, col = action
        self.board[row, col] = self.current_player.value
        self.move_history.append((action, self.current_player)) # Appending in the format - ((1,2), X)

        # Checking game result
        result = self.check_winner()

        # Switch player if game continues
        if result == GameResult.IN_PROGRESS:
            self.current_player = Player.O if self.current_player == Player.X else Player.X

        return self.get_state(), result
    
    def check_winner(self) -> GameResult:
        """ 
        Check if there's a winner or if the game is drawn
        Returns:
            GameResult enum indicating the game outcome
        """
        # Check rows:
        for i in range(3):
            if abs(self.board[i, :].sum()) == 3:
                return GameResult.X_WIN if self.board[i, 0] == 1 else GameResult.O_WIN
            
        # Check columns:
        for j in range(3):
            if abs(self.board[:, j].sum()) == 3:
                return GameResult.X_WIN if self.board[0, j] == 1 else GameResult.O_WIN
        
        # Check diagonals:
        if abs(self.board.trace()) == 3:
            return GameResult.X_WIN if self.board[0, 0] == 1 else GameResult.O_WIN
        
        if abs(np.fliplr(self.board).trace()) == 3:
            return GameResult.X_WIN if self.board[0, 2] == 1 else GameResult.O_WIN
        
        # Check for draw (no empty cells left)
        if not self.get_valid_actions():
            return GameResult.DRAW

        # Otherwise, Game is still in progress
        return GameResult.IN_PROGRESS
    
    def render(self) -> str:
        """
        Renders the board as a string for visualization
        Returns:
            String representation of the board
        """
        symbols = {1: 'X', -1: 'O', 0: '.'}
        lines = []
        lines.append("  0 1 2")  # Two spaces before column numbers
        for i in range(3):
            row_str = f"{i} "  # Row number with space after
            for j in range(3):
                row_str += symbols[self.board[i, j]] + " "
            lines.append(row_str)
        return "\n".join(lines)
    
class RandomAgent:
    """
    Random Baseline Agent
    This agent selects moves uniformly at random from the set of all valid actions.
    It serves as the baseline (benchmark) for comparison with more sophisticated agents.
    """
    def __init__(self, name: str = "RandomAgent"):
        self.name = name

    def select_action(self, game: TicTacToe) -> Tuple[int, int]:
        """ 
        Selects a random valid action from the game
        Args:
            game: Current TicTacToe game instance
        Returns:
        (row, col) tuple representing the selected move
        """
        valid_actions = game.get_valid_actions()
        if not valid_actions:
            raise ValueError("No valid moves available")
        return random.choice(valid_actions)


class RuleBasedAgent:
    """
    Rule-Based Heuristic Agent
    This agent selects moves using a fixed set of strategic heuristics rather than
    search. It attempts to win when possible, block opponent wins, take the center,
    prefer corners, and otherwise choose a remaining valid move.
    """
    def __init__(self, name: str = "RuleBasedAgent"):
        self.name = name

    def select_action(self, game: TicTacToe) -> Tuple[int, int]:
        player = game.current_player.value
        opponent = -player
        board = game.board
        valid_actions = game.get_valid_actions()

        # Helper to test hypothetical move outcomes
        def move_wins(move, player_to_test):
            row, col = move
            board[row, col] = player_to_test  # Acts as if we made the move
            result = game.check_winner()
            board[row, col] = 0  # Undo as we altered the actual board
            return ((result == GameResult.X_WIN and player_to_test == 1)  # True if X would win
                    or (result == GameResult.O_WIN and player_to_test == -1))  # True if O would win

        # 1. Wins immediately if possible
        for move in valid_actions:
            if move_wins(move, player):
                return move

        # 2. Blocks opponent immediate win 
        for move in valid_actions:
            if move_wins(move, opponent):
                return move

        # 3. Takes center
        if ((1, 1)) in valid_actions:
            return (1, 1)

        # 4. Takes a corner
        corners = [(0,0), (0,2), (2,0), (2,2)]
        open_corners = [corner for corner in corners if corner in valid_actions]
        if open_corners:
            return random.choice(open_corners)

        # 5. Takes any side
        sides = [(0,1), (1,0), (1,2), (2,1)]
        open_sides = [side for side in sides if side in valid_actions]
        if open_sides:
            return random.choice(open_sides)

        # 6. Last Resort: Picks any valid move 
        return random.choice(valid_actions)
