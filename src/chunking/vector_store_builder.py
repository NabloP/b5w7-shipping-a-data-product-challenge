# ------------------------------------------------------------------------------
# 📄 VectorStoreBuilder Module for B5W6 – Intelligent Complaint Analysis (v6)
# ------------------------------------------------------------------------------
# Author: Nabil Mohamed
# Date: July 2025
# Description:
#   Builds and persists a ChromaDB vector store using text chunks, optional
#   precomputed embeddings (preferred) or real-time embedding (fallback) with full
#   defensive programming, batching, and LangChain v0.1+ compatibility.
# ------------------------------------------------------------------------------

# ---------------------------
# Standard Library Imports
# ---------------------------
import os  # For safe directory management

# ---------------------------
# Third-Party Imports
# ---------------------------
from langchain_community.vectorstores import Chroma  # ChromaDB vector store
from langchain.schema import Document  # LangChain Document object

# ---------------------------
# VectorStoreBuilder Class
# ---------------------------


class VectorStoreBuilder:
    """
    A class to build and persist a ChromaDB vector store using text chunks,
    optional precomputed embeddings (preferred), or live embedding as fallback.
    """

    def __init__(
        self,
        persist_directory: str = "vector_store/chroma_db",
        collection_name: str = "complaint_chunks",
    ):
        """
        Initializes the VectorStoreBuilder instance.

        Args:
            persist_directory (str): Directory path where ChromaDB files will be stored.
            collection_name (str): Logical name of the ChromaDB collection.
        """
        try:
            self.persist_directory = persist_directory  # Directory for ChromaDB storage
            self.collection_name = (
                collection_name  # Logical grouping name for collection
            )
            os.makedirs(
                self.persist_directory, exist_ok=True
            )  # Ensure directory exists
        except Exception as e:
            raise RuntimeError(f"❌ Failed to initialize VectorStoreBuilder: {e}")

    def build_chroma_store(
        self,
        documents: list,
        embedding_model=None,
        metadatas: list = None,
        embeddings: list = None,
    ):
        """
        Builds and persists a ChromaDB vector store using precomputed embeddings (preferred) or live embedding.

        Args:
            documents (list): List of text chunks (str).
            embedding_model (object, optional): Embedding model for fallback live embedding.
            metadatas (list): List of metadata dictionaries.
            embeddings (list, optional): Precomputed dense vector embeddings.

        Returns:
            Chroma: Persisted ChromaDB vector store instance or None if failed.
        """
        try:
            # ✅ Validate documents
            if not documents or not isinstance(documents, list):
                raise ValueError("❌ Document list is missing, empty, or invalid.")

            # ✅ Validate metadatas
            if (
                metadatas is None
                or not isinstance(metadatas, list)
                or len(metadatas) != len(documents)
            ):
                raise ValueError(
                    "❌ Metadata list is missing, invalid, or mismatched in length."
                )

            # ✅ Create LangChain Document objects
            lc_documents = [
                Document(page_content=text, metadata=meta)
                for text, meta in zip(documents, metadatas)
            ]

            # ✅ Case 1: Preferred → Precomputed Embeddings with Batching
            if embeddings is not None:
                print(
                    "⚡ Using precomputed embeddings for ChromaDB creation (Option B)."
                )

                # Step 1: Create empty Chroma store (embedding_function still required by API)
                vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    collection_name=self.collection_name,
                    embedding_function=embedding_model,
                )

                # Step 2: Set safe batch size (e.g., below 5,000)
                max_batch_size = 5000
                total_docs = len(lc_documents)

                # Step 3: Add documents in batches
                for i in range(0, total_docs, max_batch_size):
                    batch_docs = lc_documents[i : i + max_batch_size]
                    batch_embeddings = embeddings[i : i + max_batch_size]

                    print(
                        f"➕ Adding batch {i // max_batch_size + 1} ({len(batch_docs):,} documents)..."
                    )

                    vector_store.add_documents(
                        documents=batch_docs,
                        embeddings=batch_embeddings,
                    )

                # Step 4: Persist after all batches
                vector_store.persist()

            # ✅ Case 2: Fallback → Live Embedding via from_documents
            else:
                if embedding_model is None:
                    raise ValueError(
                        "❌ No embedding model provided and no precomputed embeddings supplied."
                    )

                print(
                    "⚙️ No precomputed embeddings detected. Falling back to live embedding (Option A)."
                )

                vector_store = Chroma.from_documents(
                    documents=lc_documents,
                    embedding=embedding_model,
                    persist_directory=self.persist_directory,
                    collection_name=self.collection_name,
                )

                vector_store.persist()

            # ✅ Success confirmation
            print(
                f"✅ ChromaDB vector store created successfully with {len(documents):,} documents."
            )
            print(f"📁 Location: {self.persist_directory}")

            return vector_store

        except Exception as e:
            print(f"❌ Failed to build ChromaDB vector store: {e}")
            return None
