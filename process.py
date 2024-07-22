from flask import (
    Blueprint, request, jsonify, render_template, redirect, url_for,
    session, flash, current_app
)
from werkzeug.utils import secure_filename
import os
import logging
import requests
import time
import traceback
import numpy as np
from pydub import AudioSegment
from pytube import YouTube
import openai
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity
from moviepy.editor import VideoFileClip
import tempfile

# Import custom modules
from transcribe import Transcribe
from gpt_integration import GPTIntegration
from EmbeddingStorage import EmbeddingStorage

# Configure logging for debugging and tracking events within the application
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Constants
TIMEOUT = 1800  # Timeout for session data (e.g., transcription) in seconds
ALLOWED_EXTENSIONS = {"mp4", "mp3", "wav", "pdf", "docx"}  # Define file types allowed for upload

# Utility function to check if a file's extension is allowed
def allowed_file(filename):
    """Check if the file's extension is among the allowed ones."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize Blueprint for this module
bp = Blueprint("process", __name__)

@bp.route("/", methods=["GET", "POST"])
def upload():
    """
    Route to handle file uploads. It allows for file uploads and displays the upload page.
    Handles both GET (display page) and POST (process uploads) requests.
    """

    if request.method == "GET":
        # On GET, clear any previous transcription data from the session for a fresh start
        session.pop("transcript_segments", None)

    # Check and reset outdated transcription data
    if (
        "transcription_timestamp" in session
        and time.time() - session["transcription_timestamp"] > TIMEOUT
    ):
        current_app.logger.info("Transcription data is outdated, resetting...")
        session.pop("transcript_segments", None)
        session["transcription_timestamp"] = time.time()

    if request.method == "POST":
        # Handle POST request for file upload
        if "file" not in request.files:
            flash("No file part")  # Flash message for missing file part
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            flash("No selected file")  # Flash message for no file selected
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)  # Secure the file name
            file_path = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)  # Save the file
            flash("File successfully uploaded and is being processed.")
            return redirect(url_for("upload"))
        else:
            flash("Invalid file type.")  # Flash message for invalid file type
            return redirect(request.url)

    # For GET requests or initial page load, display any existing transcript data
    transcript_segments = session.get("transcript_segments", [])
    concatenated_transcript = " ".join(
        [segment["text"] for segment in transcript_segments]
    )
    conversation_history = session.get("conversation", [])

    return render_template(
        "upload.html",
        transcript=concatenated_transcript,
        conversation=conversation_history,
    )


def download_from_url(url):
    """
    Download content from a given URL with retry logic. It specifically handles YouTube URLs differently
    from direct video links, attempting to download the best available stream for YouTube videos.
    Retries up to three times before giving up if errors occur.
    """

    retries = 3  # Number of attempts to try downloading the file
    for attempt in range(retries):
        try:
            filename = None  # Initialize filename to None for each attempt
            if "youtube.com" in url or "youtu.be" in url:
                # Check if the URL is a YouTube link and process accordingly
                yt = YouTube(url)  # Create a YouTube object from the URL
                stream = yt.streams.filter(
                    file_extension="mp4", progressive=True
                ).first()  # Get the best quality progressive 'mp4' stream
                if stream:
                    # If a valid stream is found, download it
                    filename = tempfile.mktemp(prefix="download_", suffix=".mp4")  # Create a temporary file
                    stream.download(filename=filename)  # Download the stream to the file
                else:
                    # Log error and skip to the next attempt if no valid stream is found
                    logging.error("No suitable stream found for YouTube URL")
                    continue
            else:
                # Handle non-YouTube URLs assumed to be direct video links
                response = requests.get(url, stream=True)
                response.raise_for_status()  # Check for HTTP request errors
                filename = tempfile.mktemp(prefix="download_", suffix=".mp4")  # Create a temporary file
                with open(filename, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)  # Write the content to file in chunks

            if filename and os.path.exists(filename):
                # If the file is successfully downloaded and exists, return its path
                return filename
            else:
                # Log an error if the file does not exist after the download attempt
                logging.error(f"Failed to download file on attempt {attempt + 1}")
        except Exception as e:
            # Log any exceptions that occur during the download process
            logging.error(f"Attempt to download URL failed: {e}")

    return None  # Return None if all attempts fail


@bp.route("/transcribe", methods=["POST"])
def transcribe_route():
    """Handle the transcription process."""
    logging.info("Starting the transcription process.")
    response_data = {}

    try:
        file = request.files.get("file")
        video_url = request.form.get("videoUrl")
        file_path = None  # Initialize file_path variable

        if video_url:
            logging.info(f"Received video URL for transcription: {video_url}")
            file_path = download_from_url(video_url)
            if not file_path:
                logging.error("Failed to download video from provided URL.")
                raise ValueError("Failed to download video from provided URL.")
            logging.info(f"Video downloaded successfully: {file_path}")

        elif file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)
            logging.info(f"File uploaded and saved: {file_path}")

        else:
            logging.error("Invalid file type or no file provided.")
            raise ValueError("Invalid file type or no file provided.")

        if file_path:
            # Initialize Transcribe with additional parameters
            model_size_or_path = "base"
            device = "cpu"  # Consider "cuda" for GPU acceleration if available
            compute_type = (
                "default"  # Adjust based on your needs or application configuration
            )

            transcriber = Transcribe(
                file_path,
                model_size_or_path=model_size_or_path,
                device=device,
                compute_type=compute_type,
            )
            transcript_segments = transcriber.transcribe()
            logging.info(
                f"Transcription completed with {len(transcript_segments)} segments"
            )

            if transcript_segments:
                transcript_text = "\n".join(
                    [seg["text"] for seg in transcript_segments]
                )
                session["transcript_segments"] = transcript_segments
                response_data["transcript"] = transcript_text
                # Access embedding_storage_instance from current_app
                embedding_storage = current_app.embedding_storage
                embedding_storage.store_transcription(transcript_segments)
                logging.info(
                    "Embeddings for transcription segments have been generated and stored."
                )

            else:
                logging.warning("Transcription successful but no content extracted.")
                response_data["error"] = (
                    "Transcription successful but no content extracted."
                )
        else:
            logging.error("No file path available for transcription.")
            response_data["error"] = "No file path available for transcription."

        return jsonify(response_data)

    except ValueError as ve:
        logging.error(f"Error during transcription process: {ve}")
        response_data["error"] = str(ve)
        return jsonify(response_data), 400

    except Exception as e:
        logging.error(f"Unexpected error during transcription: {e}", exc_info=True)
        response_data["error"] = "An unexpected error occurred."
        return jsonify(response_data), 500

    finally:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            logging.info("Temporary file deleted.")

@bp.route("/ask", methods=["POST"])
def ask():
    """Handle queries to the GPT model."""
    logging.info("Received a request to '/ask' endpoint.")
    gpt_integration = current_app.gpt_integration
    data = request.get_json()
    query = data.get("query") if data else None

    if not query:
        logging.error("No query provided in the request.")
        return jsonify({"error": "No query provided"}), 400
    else:
        logging.info("Query received: %s", query)

    # Ensure conversation_history exists in the session
    if "conversation_history" not in session:
        logging.debug("Initializing 'conversation_history' in session.")
        session["conversation_history"] = []

    logging.debug(
        "Current conversation history before appending: %s",
        session.get("conversation_history"),
    )

    try:
        logging.debug("Attempting to enrich query context and send to GPT.")
        # It's now safe to get 'conversation_history' because we ensured its existence above
        response_text, metadata = gpt_integration.handle_query(
            session["conversation_history"], query
        )
        # Append the assistant's response to the conversation history
        session["conversation_history"].append(
            {"role": "assistant", "content": response_text}
        )
        # Ensure changes to session['conversation_history'] are saved
        session.modified = True
        logging.info("GPT response appended to conversation history.")
        return jsonify({"response": response_text})
    except Exception as e:
        logging.error("Failed to process the query: %s", e, exc_info=True)
        return jsonify({"error": "Error processing your query"}), 500

@bp.route("/reset_conversation", methods=["POST"])
def reset_conversation():
    """Reset the conversation history."""
    session.clear()
    return jsonify({"success": True})

def initialize_components(app_config):
    """Initialize components and store in the application context."""
    embedding_storage = EmbeddingStorage()
    gpt_integration = GPTIntegration(
        embedding_storage=embedding_storage,
        engine_id=app_config.get("GPT_ENGINE_ID", "gpt-3.5-turbo"),
    )
    current_app.gpt_integration = gpt_integration  # Store gpt_integration in current_app for global access
    return embedding_storage, gpt_integration
