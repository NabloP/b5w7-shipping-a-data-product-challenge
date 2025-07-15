# ------------------------------------------------------------------------------
# 📄 QuestionRetriever Module for B5W6 – Intelligent Complaint Analysis (v7)
# ------------------------------------------------------------------------------
# Author: Nabil Mohamed
# Date: July 2025
# Description:
#   Retrieves top-k semantically relevant complaint chunks from ChromaDB using
#   dynamically adjustable top_k per query. Fully modular for RAG integration.
# ------------------------------------------------------------------------------

from typing import List, Dict  # For type annotations
from sentence_transformers import SentenceTransformer  # For embedding generation
from langchain_community.vectorstores import Chroma  # For vector store access


class QuestionRetriever:
    """
    Retrieves the most relevant complaint chunks from ChromaDB based on a user question.
    """

    def __init__(
        self,
        vector_store: Chroma,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        top_k: int = 5,
    ):
        """
        Initializes the QuestionRetriever.

        Args:
            vector_store (Chroma): Pre-loaded Chroma vector store instance.
            embedding_model_name (str): Embedding model name for encoding.
            top_k (int): Default number of top results to retrieve.
        """
        try:
            if not hasattr(vector_store, "similarity_search_by_vector"):
                raise ValueError(
                    "❌ Provided vector_store is invalid or not a Chroma instance."
                )

            if (
                not isinstance(embedding_model_name, str)
                or not embedding_model_name.strip()
            ):
                raise ValueError("❌ Embedding model name must be a non-empty string.")

            if not isinstance(top_k, int) or top_k <= 0:
                raise ValueError("❌ top_k must be a positive integer.")

            self.vector_store = vector_store
            self.embedding_model = SentenceTransformer(embedding_model_name)
            self.top_k = top_k

        except Exception as e:
            raise RuntimeError(f"❌ Failed to initialize QuestionRetriever: {e}") from e

    def embed_question(self, question: str) -> List[float]:
        """
        Generates an embedding vector from the user question.

        Args:
            question (str): User question.

        Returns:
            List[float]: Embedding vector.
        """
        try:
            if not question or not isinstance(question, str):
                raise ValueError("❌ Question must be a non-empty string.")

            embedding = self.embedding_model.encode(question)
            return embedding.tolist()

        except Exception as e:
            raise RuntimeError(f"❌ Failed to embed question: {e}") from e

    def retrieve(self, question: str, top_k: int = None) -> List[Dict]:
        """
        Retrieves top-k relevant complaint chunks based on the input question.

        Args:
            question (str): The question to retrieve against.
            top_k (int, optional): Number of top results to return (overrides default).

        Returns:
            List[Dict]: Retrieved complaint chunks with text and metadata.
        """
        try:
            query_embedding = self.embed_question(question)
            k = (
                top_k if top_k is not None else self.top_k
            )  # Use dynamic top_k if provided

            query_result = self.vector_store.similarity_search_by_vector(
                embedding=query_embedding, k=k
            )

            results = [
                {"document": doc.page_content, "metadata": doc.metadata}
                for doc in query_result
            ]
            return results

        except Exception as e:
            raise RuntimeError(
                f"❌ Failed to retrieve relevant complaint chunks: {e}"
            ) from e
