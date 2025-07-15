# ------------------------------------------------------------------------------
# 📄 EmbeddingGenerator Module for B5W6 – Intelligent Complaint Analysis
# ------------------------------------------------------------------------------
# Author: Nabil Mohamed
# Date: July 2025
# Description:
#   Generates dense vector embeddings for complaint narratives using pre-trained
#   transformer models from Sentence-Transformers. Supports GPU acceleration and
#   batch processing for scalable embedding generation in RAG pipelines.
# ------------------------------------------------------------------------------

# ---------------------------
# Standard Library Imports
# ---------------------------
import torch  # For device detection

# ---------------------------
# Third-Party Imports
# ---------------------------
from sentence_transformers import SentenceTransformer  # For pre-trained embeddings

# ---------------------------
# EmbeddingGenerator Class
# ---------------------------


class EmbeddingGenerator:
    """
    A class to generate dense vector embeddings for complaint narratives using Sentence-Transformers.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 32):
        """
        Initializes the embedding generator with specified model and batch size.

        Args:
            model_name (str): Name of the pre-trained embedding model.
            batch_size (int): Batch size for efficient encoding.
        """
        try:
            self.model_name = model_name  # Store model name
            self.batch_size = batch_size  # Store batch size

            # ✅ Detect available device (GPU if possible, else CPU)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            # ✅ Load the sentence transformer model to the correct device
            self.model = SentenceTransformer(self.model_name, device=self.device)

            print(
                f"✅ Embedding model '{self.model_name}' loaded on device: {self.device}"
            )

        except Exception as e:
            raise RuntimeError(f"❌ Failed to initialize EmbeddingGenerator: {e}")

    def generate_embeddings(self, texts: list):
        """
        Generates sentence embeddings for a list of cleaned complaint narratives.

        Args:
            texts (list): List of text strings to embed.

        Returns:
            np.ndarray: 2D NumPy array of embeddings (n_samples, embedding_dim) or None on failure.
        """
        try:
            if not texts or not isinstance(texts, list):
                raise ValueError("❌ No valid texts provided for embedding.")

            embeddings = self.model.encode(
                texts,  # Cleaned text inputs
                batch_size=self.batch_size,  # Batch size for efficiency
                show_progress_bar=True,  # Progress visibility
                convert_to_numpy=True,  # Return as NumPy array
                normalize_embeddings=True,  # Cosine distance normalization
            )

            print(
                f"✅ Embedding generation successful: {embeddings.shape[0]:,} vectors of dimension {embeddings.shape[1]}"
            )
            return embeddings  # Return the generated embeddings

        except Exception as e:
            print(f"❌ Embedding generation failed: {e}")
            return None  # Fail-safe return
