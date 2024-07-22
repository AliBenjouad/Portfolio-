"""
Name: Ali Benjouad
Student number: 3052766
Group: none
"""

# Import required modules from PyQt6 for creating UI components, handling events, and drawing graphics.
from PyQt6.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout, QLabel, QWidget, QPushButton,
                             QGridLayout, QGraphicsDropShadowEffect, QStatusBar, QComboBox)
from PyQt6.QtGui import QAction, QIcon, QPixmap, QPainter, QImage, QPalette, QColor
from PyQt6.QtCore import Qt, QSize
import logging  # For logging messages
from game_logic import GameLogic  # Import the game logic class

# Set up logging to display information messages
logging.basicConfig(level=logging.INFO)

class GoBoard(QWidget):
    # Initialize the Go board
    def __init__(self, game_logic, place_stone, parent=None):
        super().__init__(parent)  # Initialize the QWidget parent class
        self.game_logic = game_logic  # Game logic instance to manage game state
        self.place_stone = place_stone  # Callback function to place a stone
        self.stone_size = 200 // (self.game_logic.size * 2)  # Calculate the stone size
        self.buttons = {}  # Dictionary to keep track of button widgets
        self.setFixedSize(560, 560)  # Set the fixed size of the board
        self.init_board()  # Initialize the board's UI components
        logging.info("GoBoard initialized.")  # Log the initialization

    # Create the board UI
    def init_board(self):
        self.stone_size = 30  # Set the stone size
        # Loop through each position on the board to create buttons
        for row in range(self.game_logic.size):
            for col in range(self.game_logic.size):
                button = QPushButton(self)  # Create a push button for each position
                button.setFixedSize(self.stone_size, self.stone_size)  # Set button size
                button.setIconSize(QSize(self.stone_size, self.stone_size))  # Set the icon size for stones
                # Set button style to be circular and transparent
                button.setStyleSheet(
                    "QPushButton {"
                    "border-radius: %dpx;"
                    "background-color: transparent;"
                    "border: none;"
                    "}" % (self.stone_size // 2)
                )
                # Connect button click to the place_stone method with row and column as arguments
                button.clicked.connect(lambda _, r=row, c=col: self.place_stone(r, c))
                # Calculate and move button to the correct position on the board
                x = (col + 0.5) * self.width() / self.game_logic.size - (self.stone_size // 2)
                y = (row + 0.5) * self.height() / self.game_logic.size - (self.stone_size // 2)
                button.move(x, y)
                self.buttons[(row, col)] = button  # Store button in dictionary with position as key

    # Handle the painting of the board
    def paintEvent(self, event):
        painter = QPainter(self)  # Initialize QPainter to draw on the widget
        board_size = self.game_logic.size  # Get board size from game logic
        rect = self.contentsRect()  # Get the content rectangle of the widget
        square_size = rect.width() // board_size  # Calculate the size of each square
        pixmap = QPixmap("wooden_texture.png")  # Load wooden texture for the board
        # Draw the wooden texture across the entire content rectangle
        painter.drawPixmap(rect, pixmap.scaled(rect.size(), Qt.AspectRatioMode.KeepAspectRatio))
        painter.setPen(QColor(0, 0, 0))  # Set pen color to black for the grid
        # Draw horizontal and vertical lines to create the grid
        for i in range(board_size):
            start_offset = square_size // 2
            # Draw vertical lines
            painter.drawLine(rect.left() + i * square_size + start_offset, rect.top() + start_offset,
                             rect.left() + i * square_size + start_offset, rect.bottom() - start_offset)
            # Draw horizontal lines
            painter.drawLine(rect.left() + start_offset, rect.top() + i * square_size + start_offset,
                             rect.right() - start_offset, rect.top() + i * square_size + start_offset)
        logging.info("Finished painting the game board.")  # Log the completion of painting

    # Handle mouse press events for stone placement
    def mousePressEvent(self, event):
        stone_radius = self.stone_size // 2  # Calculate the radius of the stone
        square_size = self.width() // self.game_logic.size  # Calculate the size of each square
        # Get the mouse click position and calculate the corresponding row and column
        x, y = event.pos().x(), event.pos().y()
        col = round((x - stone_radius) / square_size)
        row = round((y - stone_radius) / square_size)
        # Check if the click is within the valid range and if the left mouse button was clicked
        if 0 <= row < self.game_logic.size and 0 <= col < self.game_logic.size:
            if event.buttons() == Qt.MouseButton.LeftButton:
                self.place_stone(row, col)  # Call the place_stone method with the calculated row and column
        else:
            logging.info("Mouse pressed outside the valid intersection area.")  # Log if the click is outside valid area

# ----------------------------------------------------------------------------------------------------

class GameView(QMainWindow):
    # Initialize the GameView
    def __init__(self, game_logic, parent=None):
        super().__init__(parent)  # Initialize the QMainWindow parent class
        self.game_logic = game_logic  # Game logic instance to manage game state
        self.setWindowTitle("Go Game")  # Set the window title
        self.setFixedSize(600, 650)  # Set the fixed size of the window
        self.status_bar = QStatusBar()  # Initialize a status bar
        self.setStatusBar(self.status_bar)  # Set the status bar for the window
        self.black_score_label = QLabel('Black: 0')  # Score label for black player
        self.white_score_label = QLabel('White: 0')  # Score label for white player
        # Connect game logic signals to update methods
        self.game_logic.player_moved.connect(self.update_board)
        self.game_logic.player_moved.connect(self.update_scores)
        self.game_logic.game_ended.connect(self.display_end_game_results)
        self.initUI()  # Initialize the user interface
        logging.info("GameView initialized.")  # Log the initialization

    # Set up the UI elements of the game
    def initUI(self):
        main_layout = QVBoxLayout()  # Main layout for arranging widgets vertically

        # Game controls and scoring at the top
        game_control_layout = QHBoxLayout()  # Horizontal layout for game controls
        new_game_button = QPushButton('New Game')  # Button to start a new game
        new_game_button.clicked.connect(self.new_game)  # Connect button click to new_game method
        game_control_layout.addWidget(new_game_button)  # Add button to the game control layout

        exit_button = QPushButton('Exit')  # Button to exit the game
        exit_button.clicked.connect(self.exit_game)  # Connect button click to exit_game method
        game_control_layout.addWidget(exit_button)  # Add button to the game control layout

        # Set fixed sizes for score labels and add them to the game control layout
        self.black_score_label.setFixedSize(100, 30)
        self.white_score_label.setFixedSize(100, 30)
        game_control_layout.addWidget(self.black_score_label)
        game_control_layout.addWidget(self.white_score_label)

        game_control_layout.addStretch()  # Add stretch to push controls to the left

        main_layout.addLayout(game_control_layout)  # Add game controls to the main layout

        # Create the Go board widget
        self.go_board = GoBoard(self.game_logic, self.place_stone)
        board_layout = QHBoxLayout()  # Horizontal layout for the board and side coordinates
        board_layout.addWidget(self.go_board)  # Add the Go board to the board layout

        main_layout.addLayout(board_layout)  # Add the board layout to the main layout

        # Set the central widget with the main layout
        self.central_widget = QWidget()
        self.central_widget.setLayout(main_layout)
        self.setCentralWidget(self.central_widget)

        # Initial score update
        self.update_scores()

    # Start a new game
    def new_game(self):
        self.game_logic.reset_game()  # Reset the game logic to its initial state
        self.update_board()  # Update the board to reflect the reset
        self.update_scores()  # Update the scores to reflect the reset
        self.status_bar.showMessage("New game started! Black's turn.")  # Display message in the status bar

    # Exit the game
    def exit_game(self):
        self.close()  # Close the window

    # Update the board UI with the current game state
    def update_board(self):
        logging.info("Updating the board with current game state.")  # Log the update
        stone_size = QSize(30, 30)  # Set the stone size for icons
        # Loop through each position on the board
        for position, button in self.go_board.buttons.items():
            row, col = position
            state = self.game_logic.get_stone_color(row, col)  # Get the stone color at the position

            # Determine the icon path based on the stone color
            if state == 'black':
                icon_path = "black_stone.png"
            elif state == 'white':
                icon_path = "white_stone.png"
            else:
                icon_path = ""

            # Set the button icon if there is a stone
            if icon_path:
                pixmap = QPixmap(icon_path).scaled(stone_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                button.setIcon(QIcon(pixmap))
                button.setIconSize(stone_size)
            else:
                button.setIcon(QIcon())  # Clear the icon if there is no stone

            # Disable the button if a stone is placed or the game is over
            button.setEnabled(state is None and not self.game_logic.game_over)

        logging.info("Board update complete.")  # Log the completion of the update
        self.update_status_bar()  # Update the status bar with the current player

    # Method to update scores
    def update_scores(self):
        # Get the current scores from the game logic
        black_score, white_score, black_territory, white_territory = self.game_logic.score()

        # Update the score labels with the current scores
        self.black_score_label.setText(f'Black: {black_score} ')
        self.white_score_label.setText(f'White: {white_score} ')
        # Log the updated scores
        logging.info(f"Scores updated - Black: {black_score} (Territory: {black_territory}), White: {white_score} (Territory: {white_territory})")

    # Update the status bar with the current game state or the end game result
    def update_status_bar(self):
        if self.game_logic.game_over:  # If the game is over, no need to update
            return

        # Get the current player and display it in the status bar
        current_player = "Black" if self.game_logic.current_player == 'black' else "White"
        self.status_bar.showMessage(f"Current player: {current_player}")

    # Place a stone on the board
    def place_stone(self, row, col):
        if self.game_logic.game_over:  # If the game is over, don't allow placing stones
            return

        successful_move = self.game_logic.place_stone(row, col)  # Try to place the stone
        if successful_move:  # If the move is successful, update the board and log the move
            self.update_board()
            logging.info(f"Stone placed at row {row}, column {col}.")
        else:  # If the move is invalid, log a warning and update the status bar
            logging.warning(f"Invalid move at ({row}, {col}). Current player: {self.game_logic.current_player}")
            self.status_bar.showMessage(f"Invalid move attempted by {self.game_logic.current_player}.")

    # Display the end game results in the status bar
    def display_end_game_results(self, black_score, white_score, winner):
        # Log the end game results and display them in the status bar
        logging.info(f"Displaying end game results. Winner: {winner.capitalize()} - Black: {black_score}, White: {white_score}")
        message = f"Game ended. Winner: {winner.capitalize()} - Black: {black_score}, White: {white_score}"
        self.status_bar.showMessage(message)
