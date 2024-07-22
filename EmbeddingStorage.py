import logging
import numpy as np
import os
from openai import OpenAI

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Instantiate the OpenAI client with the API key fetched from an environment variable
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class EmbeddingStorage:
    """
    A class to store and manage text embeddings, providing functionality to add,
    retrieve, and find relevant text segments based on embeddings.
    """

    def __init__(self):
        self.id_to_embedding = {}  # Maps unique IDs to embeddings
        self.id_to_text = {}  # Maps unique IDs to original text segments
        self.current_id = 0  # Tracks the next ID to assign

        # Check for API key presence and raise an error if it's not set
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        logging.debug("EmbeddingStorage initialized.")

    def get_text_embedding(self, text):
        """
        Fetch the embedding for a given text using OpenAI's embedding model.
        """
        logging.debug("Fetching embedding for text: '%s'", text[:30])  # Log the initial part of the text
        try:
            # Call OpenAI API to create embeddings
            response = client.embeddings.create(
                input=[text], model="text-embedding-ada-002"
            )
            embedding_vector = response.data[0].embedding
            if np.any(embedding_vector):  # Check if the embedding is not a zero vector
                logging.debug("Valid embedding vector generated.")
                return np.array(embedding_vector)
            else:
                logging.warning("Received a zero vector as embedding for text: '%s'", text[:30])
                return None
        except Exception as e:
            logging.error("Failed to get embedding from OpenAI for text: '%s', error: %s", text[:30], e)
            return None

    def store_transcription(self, transcript_segments):
        """
        Store embeddings and their corresponding text from transcription segments.
        """
        logging.debug("Storing transcriptions and embeddings.")
        for segment in transcript_segments:
            text = segment.get("text", "")
            embedding = self.get_text_embedding(text)
            if embedding is not None:
                self.id_to_text[self.current_id] = text
                self.id_to_embedding[self.current_id] = embedding
                self.current_id += 1  # Increment the ID for the next entry
            else:
                logging.warning("No valid embedding generated for segment: %s...", text[:30])
        logging.info("Stored %d segments.", len(transcript_segments))

    def find_relevant_segments(self, query, top_k=3):
        """
        Find and return the top-k most relevant text segments for a given query based on cosine similarity.
        """
        logging.debug("Finding relevant segments for query: %s", query)
        query_embedding = self.get_text_embedding(query)
        if query_embedding is None:
            logging.warning("Query embedding retrieval failed. Returning no relevant segments.")
            return []

        # Calculate cosine similarities and sort them
        scores = {
            id: np.dot(query_embedding, self.id_to_embedding[id]) / (np.linalg.norm(query_embedding) * np.linalg.norm(self.id_to_embedding[id]))
            for id in self.id_to_embedding if np.any(self.id_to_embedding[id])
        }
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # Return the most relevant segments
        relevant_segments = [{"text": self.id_to_text[id]} for id, _ in sorted_scores]
        logging.info("Found %d relevant segments for query.", len(relevant_segments))
        return relevant_segments

    def find_relevant_segments_with_metadata(self, query, top_k=3):
        """
        Similar to `find_relevant_segments` but returns metadata alongside the text.
        """
        logging.debug("Finding relevant segments with metadata for query: %s", query)
        query_embedding = self.get_text_embedding(query)
        if query_embedding is None:
            logging.warning("Query embedding retrieval failed. Returning no relevant segments.")
            return []

        # Calculate cosine similarities and sort them
        scores = {
            id: np.dot(query_embedding, self.id_to_embedding[id]) / (np.linalg.norm(query_embedding) * np.linalg.norm(self.id_to_embedding[id]))
            for id in self.id_to_embedding if np.any(self.id_to_embedding[id])
        }
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # Return the most relevant segments with additional metadata
        relevant_segments_with_metadata = [( {"text": self.id_to_text[id], "id": id}, scores[id] ) for id, _ in sorted_scores]
        logging.info("Found %d relevant segments with metadata for query.", len(relevant_segments_with_metadata))
        return relevant_segments_with_metadata
