import openai
import logging
import os
import numpy as np


class GPTIntegration:
    def __init__(self, embedding_storage, engine_id="gpt-3.5-turbo"):
        self.api_key = self.get_api_key()
        self.engine_id = engine_id
        self.embedding_storage = embedding_storage
        logging.info("GPTIntegration initialized with engine ID: %s", engine_id)

    @staticmethod
    def get_api_key():
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logging.error("OPENAI_API_KEY environment variable not set.")
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        else:
            logging.info("OPENAI_API_KEY obtained successfully.")
        return api_key

    def enrich_query_context(self, query):
        logging.info("Enriching query context for: '%s'", query)
        relevant_segments = self.embedding_storage.find_relevant_segments(query)
        if relevant_segments and all(
            isinstance(seg, dict) and "text" in seg for seg in relevant_segments
        ):
            enriched_context = "\n".join([seg["text"] for seg in relevant_segments])
            logging.info("Context enriched with %d segments.", len(relevant_segments))
            return enriched_context, relevant_segments
        else:
            logging.warning(
                "No relevant segments found or returned data is not in the expected format. Segments: %s",
                relevant_segments,
            )
            return "", []

    def handle_query(self, conversation_history, query):
        client = openai.OpenAI(api_key=self.api_key)
        logging.info("Preparing to send query to OpenAI with context.")
        enriched_context, metadata = self.enrich_query_context(query)
        if enriched_context:
            logging.info("Enriched context: %s", enriched_context)
        else:
            logging.info("No enriched context found, proceeding without it.")
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        if enriched_context:
            messages.append({"role": "system", "content": enriched_context})
        messages += [{"role": "user", "content": query}]
        try:
            response = client.chat.completions.create(
                model=self.engine_id,
                messages=messages,
                temperature=0.7,
                max_tokens=150,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
            )
            logging.info("Query sent and response received from OpenAI.")
            return response.choices[0].message.content.strip(), metadata
        except Exception as e:
            logging.error("Error fetching response from OpenAI: %s", e, exc_info=True)
            return "An error occurred while processing the request.", []

    def test_api_connection(self):
        client = openai.OpenAI(api_key=self.get_api_key())
        logging.info("Testing OpenAI API connection.")
        test_prompt = "This is a test prompt to verify the OpenAI API connection."
        try:
            response = client.completions.create(
                model="text-davinci-003",  # Use a model compatible with completions for the test
                prompt=test_prompt,
                max_tokens=5,
            )
            logging.info("OpenAI API connection test successful.")
            return True
        except Exception as e:
            logging.error("Failed to connect to OpenAI API: %s", e, exc_info=True)
            return False
