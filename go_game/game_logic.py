"""
Name: Ali Benjouad
Student number: 3052766
Group: none
"""

# Import necessary libraries: numpy for numerical operations, logging for logging information, and PyQt6.QtCore for GUI interaction.
import numpy as np
import logging
from PyQt6.QtCore import QObject, pyqtSignal

# Define the GameLogic class inheriting from QObject to utilize PyQt6's signal and slot mechanism
class GameLogic(QObject):
    # Define PyQt6 signals to communicate game events to the GUI or other components
    player_moved = pyqtSignal()
    game_ended = pyqtSignal(int, int, str)  # Signal for game end with scores and winner

    # Initialization method with adhering to the 7*7 gird specified
    def __init__(self, size=7):
        super().__init__()  # Call the QObject initializer
        self.size = size  # Set the board size
        # Initialize the board with all positions set to None indicating no stone placed
        self.board = np.full((size, size), None)
        # Initialize game variables such as the current player, previous states for KO rule...
        self.current_player = 'black'
        self.previous_states = []
        self.pass_count = 0
        self.black_captures = 0
        self.white_captures = 0
        self.game_over = False  # Initialize game_over attribute
        # Configure logging to INFO level
        logging.basicConfig(level=logging.INFO)
        print("Initialized new game board.")

    # Method to switch the current player between 'black' and 'white'
    def switch_player(self):
        self.current_player = 'white' if self.current_player == 'black' else 'black'
        print(f"Switched player. Current player is now {self.current_player}.")

    # Check if a position is within the bounds of the board
    def is_on_board(self, row, col):
        on_board = 0 <= row < self.size and 0 <= col < self.size
        logging.debug(f"Checking if position ({row}, {col}) is on board: {on_board}")
        return on_board

    # Retrieve adjacent positions for a given stone position
    def get_adjacent(self, row, col):
        adjacent = []
        # Define possible adjacent positions (up, down, left, right)
        for r, c in [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]:
            if self.is_on_board(r, c):  # Only add it if it's on the board
                adjacent.append((r, c))
        return adjacent

    # Find all positions that are part of the same group/chain of stones
    def get_group(self, row, col, board):
        color = board[row, col]
        if color is None:  # If the position is empty, return an empty group
            return []

        group = set()  # Using a set to avoid duplicates
        stack = [(row, col)]  # Start with the initial stone
        # i used here depth first search algorithm to find all connected stones of the same color
        while stack:
            r, c = stack.pop()
            if (r, c) not in group:
                group.add((r, c))
                for adj_r, adj_c in self.get_adjacent(r, c):
                    if board[adj_r, adj_c] == color:  # If adjacent stone is the same color, add it to the stack
                        stack.append((adj_r, adj_c))
        return group

    # Calculate liberties (empty adjacent positions) for a group of stones
    def get_liberties(self, group, board):
        liberties = set()  # Use a set to avoid duplicates
        for row, col in group:
            for adj_r, adj_c in self.get_adjacent(row, col):
                if board[adj_r, adj_c] is None:  # If adjacent position is empty, it's a liberty
                    liberties.add((adj_r, adj_c))
        return liberties

    # Determine if placing a stone at the given position would be a suicide
    def is_suicide(self, row, col):
        temp_board = self.board.copy()  # Make a temporary copy of the board
        temp_board[row, col] = self.current_player  # Place the stone on the temporary board
        group = self.get_group(row, col, temp_board)  # Get the group of the placed stone
        liberties = self.get_liberties(group, temp_board)  # Calculate liberties for the group
        
        if len(liberties) == 0:  # If no liberties, it might be suicide unless it captures other stones
            for adj in self.get_adjacent(row, col):
                adj_group = self.get_group(*adj, temp_board)  # Get the adjacent group
                adj_liberties = self.get_liberties(adj_group, temp_board)  # Calculate liberties for the adjacent group
                if temp_board[adj] != self.current_player and len(adj_liberties) == 0:
                    return False  # If capturing, not suicide
            return True  # If not capturing any stone, it's a suicide
        return False  # If there are liberties, it's not a suicide

    # Check for the KO rule, preventing repeat positions.
    def is_ko(self, row, col):
        temp_board = self.board.copy()  # Make a temporary copy of the board
        temp_board[row, col] = self.current_player  # Place the stone on the temporary board
        return str(temp_board) in self.previous_states  # Check if the new board configuration has occurred before

    # Attempt to place a stone at the given position
    def place_stone(self, row, col):
        print(f"Attempting to place stone for {self.current_player} at ({row}, {col}).")
        # Check for valid position, not suicide, and KO rule
        if not self.is_on_board(row, col) or self.board[row, col] is not None:
            print(f"Invalid move: outside board or position already occupied at ({row}, {col})")
            return False

        if self.is_suicide(row, col):
            print(f"Move at ({row}, {col}) would be suicide.")
            return False

        if self.is_ko(row, col):
            print(f"Move at ({row}, {col}) would violate the KO rule.")
            return False

        # Place the stone on the board
        print(f"Placing stone at ({row}, {col}).")
        self.board[row, col] = self.current_player
        self.capture_stones(row, col)  # Capture any opposing stones
        self.previous_states.append(str(self.board))  # Record the new board state
        self.player_moved.emit()  # Emit signal after a successful move

        # Check for remaining valid moves or end the game
        if not self.no_valid_moves_left():
            self.switch_player()  # If there are moves, switch player
        else:
            self.exit_game()  # If no moves, exit the game

        print(f"Stone placed at ({row}, {col}) by {self.current_player}.")
        return True

    # Remove captured stones from the board and update capture counts.
    def capture_stones(self, row, col):
        captured_stones = 0
        for adj in self.get_adjacent(row, col):
            adj_color = self.board[adj]
            if adj_color is not None and adj_color != self.current_player:
                group = self.get_group(*adj, self.board)
                if not self.get_liberties(group, self.board):
                    for r, c in group:
                        self.board[r, c] = None  # Remove the stone
                        captured_stones += 1
                        logging.info(f"Captured stone at ({r}, {c})")

        # Update capture counts
        if self.current_player == 'black':
            self.white_captures += captured_stones
        else:
            self.black_captures += captured_stones

    # Handle passing, ending the game if necessary
    def passed(self):
        self.pass_count += 1
        if self.pass_count >= 2:
            logging.info("Game ended: Both players passed consecutively")
            return True
        self.switch_player()  # Switch players if the game continues
        return False

    # Return the color of the stone at the given position
    def get_stone_color(self, row, col):
        return self.board[row, col]

    # Calculate and return the scores based on captures and potentially territory
    def score(self):
        # Calculate scores based on captures with specific point values
        black_score = 7 * self.white_captures
        white_score = 7.5 * self.black_captures

        # Initialize territory counts
        black_territory, white_territory = 0, 0
        # Calculate territory (additional rules might be applied for counting)
        for row in range(self.size):
            for col in range(self.size):
                stone = self.get_stone_color(row, col)
                if stone == 'black':
                    black_territory += 1
                elif stone == 'white':
                    white_territory += 1

        # Return the scores and territories
        return black_score, white_score, black_territory, white_territory

    # Determine territory control based on surrounding stones
    def check_territory_control(self, row, col):
        # Initialize a matrix to keep track of visited positions
        visited = np.full((self.size, self.size), False)
        stack = [(row, col)]  # Stack for depth first search 
        surrounding_stones = set()  # Set to store surrounding stone colors

        # DFS to find all bordering stones and determine territory control
        while stack:
            r, c = stack.pop()
            if not visited[r, c]:
                visited[r, c] = True
                if self.board[r, c] is None:  # If empty, continue DFS
                    for adj in self.get_adjacent(r, c):
                        stack.append(adj)
                else:  # If stone found, add its color to the surrounding stones
                    surrounding_stones.add(self.board[r, c])

        # Determine territory control based on surrounding stones
        if len(surrounding_stones) == 1:
            return surrounding_stones.pop()  # If only one color surrounds, it controls the territory
        return None

    # Check if there are any valid moves left on the board
    def no_valid_moves_left(self):
        for row in range(self.size):
            for col in range(self.size):
                if self.board[row, col] is None and not self.is_suicide(row, col) and not self.is_ko(row, col):
                    logging.info(f"Valid move found at ({row}, {col}).")
                    return False
        logging.info("No valid moves left on the board.")
        return True

    # Reset the game to its initial state
    def reset_game(self):
        self.board = np.full((self.size, self.size), None)  # Reset the board
        # Reset game variables to initial state
        self.current_player = 'black'
        self.previous_states = []
        self.game_over = False
        self.pass_count = 0
        self.black_captures = 0
        self.white_captures = 0
        self.player_moved.emit()  # Emit signal to indicate player moved/reset

    # Handle the end of the game, determining the winner and logging the result
    def exit_game(self):
        black_score, white_score, _, _ = self.score()  # Calculate final scores
        # Determine the winner based on scores
        self.winner = 'black' if black_score > white_score else 'white' if white_score > black_score else 'draw'
        self.game_over = True  # Set the game over flag
        # Log and emit game ended information
        logging.info(f"Game ended. Winner: {self.winner} - Black: {black_score}, White: {white_score}")
        self.game_ended.emit(black_score, white_score, self.winner)
