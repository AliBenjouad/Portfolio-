import logging
import numpy as np
import os
from openai import OpenAI

# Instantiate the OpenAI client with the API key from environment variable
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class EmbeddingStorage:
    def __init__(self):
        self.id_to_embedding = {}
        self.id_to_text = {}
        self.current_id = 0
        # Directly check for the presence of the OPENAI_API_KEY in the environment
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        logging.debug("EmbeddingStorage initialized.")

    def get_text_embedding(self, text):
        logging.debug("Fetching embedding for text: '%s'", text[:30])
        try:
            response = client.embeddings.create(
                input=[text], model="text-embedding-ada-002"
            )
            embedding_vector = response.data[0].embedding
            logging.debug("Embedding vector received: %s", embedding_vector)
            if np.any(embedding_vector):
                logging.debug("Valid embedding vector generated.")
                return np.array(embedding_vector)
            else:
                logging.warning(
                    "Received a zero vector as embedding for text: '%s'", text[:30]
                )
                return None
        except Exception as e:
            logging.error(
                "Failed to get embedding from OpenAI for text: '%s', error: %s",
                text[:30],
                e,
            )
            return None

    def store_transcription(self, transcript_segments):
        logging.debug("Storing transcriptions and embeddings.")
        for segment in transcript_segments:
            text = segment.get("text", "")
            embedding = self.get_text_embedding(text)
            if embedding is not None:
                self.id_to_text[self.current_id] = text
                self.id_to_embedding[self.current_id] = embedding
                self.current_id += 1
            else:
                logging.warning(
                    f"No valid embedding generated for segment: {text[:30]}..."
                )
        logging.info(f"Stored {len(transcript_segments)} segments.")

    def find_relevant_segments(self, query, top_k=3):
        logging.debug(f"Finding relevant segments for query: {query}")
        query_embedding = self.get_text_embedding(query)
        if query_embedding is None:
            logging.warning(
                "Query embedding retrieval failed. Returning no relevant segments."
            )
            return []
        scores = {
            id: np.dot(query_embedding, self.id_to_embedding[id])
            / (
                np.linalg.norm(query_embedding)
                * np.linalg.norm(self.id_to_embedding[id])
            )
            for id in self.id_to_embedding
            if np.any(self.id_to_embedding[id])
        }
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # Wrap each segment's text in a dictionary
        relevant_segments = [{"text": self.id_to_text[id]} for id, _ in sorted_scores]

        logging.info(f"Found {len(relevant_segments)} relevant segments for query.")
        return relevant_segments

    def find_relevant_segments_with_metadata(self, query, top_k=3):
        logging.debug(f"Finding relevant segments with metadata for query: {query}")
        query_embedding = self.get_text_embedding(query)
        if query_embedding is None:
            logging.warning(
                "Query embedding retrieval failed. Returning no relevant segments."
            )
            return []
        scores = {
            id: np.dot(query_embedding, self.id_to_embedding[id])
            / (
                np.linalg.norm(query_embedding)
                * np.linalg.norm(self.id_to_embedding[id])
            )
            for id in self.id_to_embedding
            if np.any(self.id_to_embedding[id])
        }
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        relevant_segments_with_metadata = [
            ({"text": self.id_to_text[id], "id": id}, scores[id])
            for id, _ in sorted_scores
        ]
        logging.info(
            f"Found {len(relevant_segments_with_metadata)} relevant segments with metadata for query."
        )
        return relevant_segments_with_metadata
