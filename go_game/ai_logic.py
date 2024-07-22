"""
Name: Ali Benjouad
Student number: 3052766
Group: none
"""

# Import the necessary libraries: random for making random choices, and logging for logging information.
import random
import logging
# Import the GameLogic class, this class will managethe game state and rules.
from game_logic import GameLogic

# Defininng a class named SimpleAIOpponent, representing a simple AI opponent.
class SimpleAIOpponent:
    # Initialization method with a parameter for game logic
    def __init__(self, game_logic):
        # Assign the provided game logic to an instance variable for later use
        self.game_logic = game_logic
        # Configure the logging level to INFO, which controls what level of messages are tracked
        logging.basicConfig(level=logging.INFO)

    # Define a method to find all legal moves based on the game's current state
    def find_legal_moves(self):
        # Initialize an empty list to hold legal moves
        legal_moves = []
        # Iterate through each row and column index based on the size of the game board
        for row in range(self.game_logic.size):
            for col in range(self.game_logic.size):
                # Check if the current cell is empty, not a suicide move, and not a ko move using the defined functions in game_logic.py
                if self.game_logic.board[row, col] is None and not self.game_logic.is_suicide(row, col) and not self.game_logic.is_ko(row, col):
                    # If the conditions are met, append the move (row, col) as a legal move.
                    legal_moves.append((row, col))
        # Return the list of all found legal moves.
        return legal_moves

    # method for the AI to make a move
    def make_move(self):
        # Retrieve the current legal moves by calling the find_legal_moves' function
        legal_moves = self.find_legal_moves()
        # Check if there are no legal moves available
        if not legal_moves:
            # Log the information that the AI has no legal moves
            logging.info("AI has no legal moves.")
            # Make the AI pass its turn
            self.game_logic.passed()
            # Return None to indicate no move was made
            return None

        # Choose a random move from the list of legal moves, I would have loved to add strategies and make it harder, but the AI opponent was mainly to speed up testing
        move = random.choice(legal_moves)
        # Attempt to place a stone at the chosen move using the 'place_stone' function from game_logic
        placed = self.game_logic.place_stone(*move)
        # If the stone was successfully placed
        if placed:
            # Log the successful placement with the move's coordinates
            logging.info(f"AI placed a stone at {move}")
            # Return the move as the result of this method
            return move
        else:
            # If the move was unsuccessful, log an error with the move's coordinates
            logging.error(f"AI failed to place a stone at {move} due to invalid move.")
            # Return None to indicate the move was unsuccessful
            return None
