from flask import (
    Blueprint,
    request,
    jsonify,
    render_template,
    redirect,
    url_for,
    session,
    flash,
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
from flask import current_app


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

TIMEOUT = 1800

ALLOWED_EXTENSIONS = {"mp4", "mp3", "wav", "pdf", "docx"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# Initialize Blueprint
bp = Blueprint("process", __name__)


@bp.route("/", methods=["GET", "POST"])
def upload():

    if request.method == "GET":
        # Clear previous transcription when the page is loaded
        session.pop("transcript_segments", None)
    # Check if transcription data is outdated and reset if necessary
    if (
        "transcription_timestamp" in session
        and time.time() - session["transcription_timestamp"] > TIMEOUT
    ):
        current_app.logger.info("Transcription data is outdated, resetting...")
        session.pop("transcript_segments", None)
        session["transcription_timestamp"] = time.time()

    if request.method == "POST":
        # Handle file upload
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)
            # Here you can add logic to process the file, e.g., transcribing audio
            flash("File successfully uploaded and is being processed.")
            # Redirect to the upload page again or to a different page to show the transcription
            return redirect(url_for("upload"))
        else:
            flash("Invalid file type.")
            return redirect(request.url)

    # For GET requests or initial page load
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


# Function to download content from URL
def download_from_url(url):
    # Attempt to download content with retry logic
    retries = 3
    for attempt in range(retries):
        try:
            filename = None
            if "youtube.com" in url or "youtu.be" in url:
                # Process YouTube URLs
                yt = YouTube(url)
                stream = yt.streams.filter(
                    file_extension="mp4", progressive=True
                ).first()
                if stream:
                    filename = tempfile.mktemp(prefix="download_", suffix=".mp4")
                    stream.download(filename=filename)
                else:
                    logging.error("No suitable stream found for YouTube URL")
                    continue  # Try again
            else:
                # Process other URLs (assuming direct video links)
                response = requests.get(url, stream=True)
                response.raise_for_status()  # Will throw an exception for bad responses
                filename = tempfile.mktemp(prefix="download_", suffix=".mp4")
                with open(filename, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

            if filename and os.path.exists(filename):
                return filename  # Successfully downloaded file
            else:
                logging.error(f"Failed to download file on attempt {attempt + 1}")
        except Exception as e:
            logging.error(f"Attempt to download URL failed: {e}")

    return None  # Failed to download after retries


# Route for transcription processing
@bp.route("/transcribe", methods=["POST"])
def transcribe_route():
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


# Route for handling queries with GPT
@bp.route("/ask", methods=["POST"])
def ask():
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


# Route to reset conversation history
@bp.route("/reset_conversation", methods=["POST"])
def reset_conversation():
    session.clear()
    return jsonify({"success": True})


def initialize_components(app_config):
    embedding_storage = EmbeddingStorage()
    gpt_integration = GPTIntegration(
        embedding_storage=embedding_storage,
        engine_id=app_config.get("GPT_ENGINE_ID", "gpt-3.5-turbo"),
    )
    current_app.gpt_integration = (
        gpt_integration  # Store gpt_integration in current_app for global access
    )
    return embedding_storage, gpt_integration
