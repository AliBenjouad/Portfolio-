import os
import logging
from moviepy.editor import VideoFileClip
import PyPDF2
import docx
from pathlib import Path
import tempfile
from faster_whisper import WhisperModel  # Ensure this is correctly installed and imported based on your setup

# Setup basic configuration for logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

class Transcribe:
    """
    This class handles the transcription of various media types (audio, video, PDF, DOCX)
    into text using appropriate models and techniques for each format.
    """

    def __init__(self, filepath, model_size_or_path="base", device="cuda", compute_type="default"):
        """
        Initialize the Transcribe object with the file path and settings for the transcription model.
        """
        self.filepath = filepath
        self.model_size_or_path = model_size_or_path
        self.device = device
        self.compute_type = compute_type
        self.file_type = self.determine_file_type()  # Determine the type of the file based on its extension
        self.initialize_model()  # Initialize the transcription model based on the file type

    def initialize_model(self):
        """
        Initialize the Whisper model for audio and video files.
        """
        try:
            if self.file_type in ["audio", "video"]:
                self.whisper_model = WhisperModel(
                    model_size_or_path=self.model_size_or_path, device=self.device
                )
                logging.info("Whisper model initialized successfully.")
            else:
                logging.info(f"No Whisper model required for file type: {self.file_type}")
        except Exception as e:
            logging.error(f"Failed to initialize Whisper model: {e}")

    def determine_file_type(self):
        """
        Determine the file type by checking the file extension.
        """
        extension = Path(self.filepath).suffix.lower()
        return {
            ".mp3": "audio",
            ".wav": "audio",
            ".mp4": "video",
            ".pdf": "pdf",
            ".docx": "docx",
        }.get(extension, None)

    def transcribe(self):
        """
        Main method to route the transcription process based on the file type.
        """
        if self.file_type in ["audio", "video"]:
            if not self.whisper_model:
                logging.error("Whisper model not initialized properly.")
                return None
            return self.transcribe_media()
        elif self.file_type == "pdf":
            return self.transcribe_pdf()
        elif self.file_type == "docx":
            return self.transcribe_docx()
        else:
            logging.error(f"Unsupported file type for transcription: {self.file_type}")
            return None

    def transcribe_media(self):
        """
        Transcribe audio or video files using the Whisper model.
        """
        logging.info(f"Starting transcription for {self.filepath}")
        if self.file_type == "video":
            logging.info(f"Extracting audio from video: {self.filepath}")
            audio_file_path = self.extract_audio_from_video()
            if not audio_file_path:
                logging.error("Failed to extract audio from video.")
                return None
            self.filepath = audio_file_path  # Update the filepath to the extracted audio for transcription
            logging.info(f"Audio extracted successfully: {audio_file_path}")

        logging.info(f"Transcribing {self.file_type} file: {self.filepath}")
        try:
            segments, info = self.whisper_model.transcribe(self.filepath, beam_size=5)
            transcript_segments = [
                {"text": segment.text, "start": segment.start, "end": segment.end}
                for segment in segments
            ]
            logging.info(
                f"Transcription completed with {len(transcript_segments)} segments"
            )
            logging.info(
                f"Detected language: {info.language} with probability {info.language_probability}"
            )
            return transcript_segments
        except Exception as e:
            logging.error(f"Error during {self.file_type} transcription: {e}")
            return None

    def extract_audio_from_video(self):
        """
        Extract audio from a video file to a temporary location for transcription.
        """
        try:
            output_audio_path = tempfile.mktemp(suffix=".mp3")
            video = VideoFileClip(self.filepath)
            video.audio.write_audiofile(output_audio_path)
            return output_audio_path
        except Exception as e:
            logging.error(f"Error extracting audio from video: {e}")
            return None

    def transcribe_pdf(self):
        """
        Extract text from PDF files using PyPDF2.
        """
        print(f"Extracting text from PDF: {self.filepath}")
        text_content = []
        try:
            with open(self.filepath, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text_content.append(page.extract_text())
        except Exception as e:
            print(f"Error reading PDF file: {e}")
        return [{"text": text} for text in text_content]

    def transcribe_docx(self):
        """
        Extract text from DOCX files using python-docx.
        """
        print(f"Extracting text from DOCX: {self.filepath}")
        text_content = []
        try:
            doc = docx.Document(self.filepath)
            text_content = [para.text for para in doc.paragraphs]
        except Exception as e:
            print(f"Error reading DOCX file: {e}")
        return [{"text": text} for text in text_content]
