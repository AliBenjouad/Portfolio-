from flask import Flask

# Initialize the Flask application
app = Flask(__name__)

# Import routes to ensure they're attached to the app
from app import routes
