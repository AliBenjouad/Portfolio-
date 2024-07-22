"""
Name: Ali Benjouad
Student number: 3052766
Group: none
"""

import sys
from PyQt6.QtWidgets import QApplication, QMessageBox
from view import GameView
from game_logic import GameLogic
from ai_logic import SimpleAIOpponent
import logging

def main():
    logging.basicConfig(level=logging.DEBUG)

    # Create the application
    app = QApplication(sys.argv)

    while True:  # Start of the new game loop
        # Initialize game logic
        game_logic = GameLogic()

        # Create the main game view and set up the interface
        game_view = GameView(game_logic)

        # Prompt user to play against AI
        play_with_ai = QMessageBox.question(
            None, "Play Against AI",
            "Do you want to play against the AI?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        ai_opponent = None
        if play_with_ai == QMessageBox.StandardButton.Yes:
            # Set up AI opponent
            ai_opponent = SimpleAIOpponent(game_logic)

        game_view.show()

        # Application main loop for the current game
        while not game_logic.game_over:
            app.processEvents()  # Process any pending events

            # If AI is enabled and it's the AI's turn to play
            if ai_opponent and game_logic.current_player == 'white':
                ai_move = ai_opponent.make_move()
                if ai_move:
                    game_view.update_board()  # Update the board in the UI

        # Prompt user to play a new game or exit
        play_again = QMessageBox.question(
            None, "Play Again",
            "Do you want to start a new game?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if play_again == QMessageBox.StandardButton.No:
            break  # Exit the new game loop if the user doesn't want to play again

        game_view.close()  # Close the current game window

    sys.exit(app.exec())

if __name__ == '__main__':
    main()
