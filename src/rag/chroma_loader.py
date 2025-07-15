# ------------------------------------------------------------------------------
# 📄 ChromaLoader Module for B5W6 – Intelligent Complaint Analysis (v7)
# ------------------------------------------------------------------------------
# Author: Nabil Mohamed
# Date: July 2025
# Description:
#   Loads an existing ChromaDB vector store collection using LangChain's Chroma
#   wrapper for compatibility with precomputed embedding pipelines in Task 2.
#   Includes full defensive programming, diagnostic logging, and modular design.
# ------------------------------------------------------------------------------

# ---------------------------
# Standard Library Imports
# ---------------------------
import os  # For directory validation

# ---------------------------
# Third-Party Imports
# ---------------------------
from langchain_community.vectorstores import Chroma  # LangChain Chroma loader
from langchain.embeddings import HuggingFaceEmbeddings  # Embedding function for search


# ---------------------------
# ChromaLoader Class
# ---------------------------


class ChromaLoader:
    """
    Loads a persisted ChromaDB vector store collection using LangChain's Chroma class.
    Ensures full compatibility with vector stores built in Task 2 for downstream RAG applications.
    """

    def __init__(
        self, persist_directory: str, collection_name: str, embedding_model_name: str
    ):
        """
        Initializes the ChromaLoader instance.

        Args:
            persist_directory (str): Directory where the ChromaDB vector store is saved.
            collection_name (str): Logical name of the ChromaDB collection.
            embedding_model_name (str): Name of the embedding model for similarity search (e.g., 'all-MiniLM-L6-v2').
        """
        try:
            # ✅ Validate persist directory
            if not isinstance(persist_directory, str) or not os.path.isdir(
                persist_directory
            ):
                raise ValueError(
                    f"❌ Invalid or non-existent directory: '{persist_directory}'."
                )

            # ✅ Validate collection name
            if not isinstance(collection_name, str) or not collection_name.strip():
                raise ValueError("❌ Collection name must be a non-empty string.")

            # ✅ Validate embedding model name
            if (
                not isinstance(embedding_model_name, str)
                or not embedding_model_name.strip()
            ):
                raise ValueError("❌ Embedding model name must be a non-empty string.")

            # ✅ Store inputs as attributes
            self.persist_directory = persist_directory  # Store directory path
            self.collection_name = collection_name  # Store collection name
            self.embedding_model_name = (
                embedding_model_name  # Store embedding model name
            )

            # ✅ Initialize embedding model (required by LangChain Chroma)
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name
            )  # Load embedding model

        except Exception as e:
            raise RuntimeError(f"❌ Failed to initialize ChromaLoader: {e}") from e

    def load_collection(self):
        """
        Loads and returns the ChromaDB vector store using LangChain's Chroma.

        Returns:
            Chroma: Loaded Chroma vector store object ready for semantic search.
        """
        try:
            # ✅ Load the Chroma vector store from disk using LangChain's Chroma
            vector_store = Chroma(
                persist_directory=self.persist_directory,  # Path to persisted store
                collection_name=self.collection_name,  # Collection name
                embedding_function=self.embedding_model,  # Embedding function required for similarity search
            )

            # ✅ Confirm successful load and document count
            document_count = (
                vector_store._collection.count()
            )  # Access internal Chroma collection count

            print(
                f"✅ Chroma vector store '{self.collection_name}' loaded successfully with {document_count:,} documents."
            )
            print(f"📁 Storage location: {self.persist_directory}")

            return vector_store  # Return loaded vector store

        except Exception as e:
            raise RuntimeError(f"❌ Failed to load Chroma vector store: {e}") from e
