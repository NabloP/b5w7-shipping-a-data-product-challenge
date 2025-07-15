# ------------------------------------------------------------------------------
# 📄 TextChunker Module for B5W6 – Intelligent Complaint Analysis
# ------------------------------------------------------------------------------
# Author: Nabil Mohamed
# Date: July 2025
# Description:
#   Splits long complaint narratives into manageable, semantically coherent chunks
#   using LangChain's RecursiveCharacterTextSplitter. Designed for embedding and
#   semantic search pipelines (Task 2+).
# ------------------------------------------------------------------------------

# ---------------------------
# Standard Library Imports
# ---------------------------
import pandas as pd  # For structured data handling

# ---------------------------
# Third-Party Imports
# ---------------------------
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)  # For intelligent chunking

# ---------------------------
# TextChunker Class Definition
# ---------------------------


class TextChunker:
    """
    A text chunking class for splitting complaint narratives into smaller, semantically coherent chunks.
    Uses LangChain's RecursiveCharacterTextSplitter for optimal token management.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initializes the chunker with desired chunk size and overlap.

        Args:
            chunk_size (int): Maximum number of characters per chunk.
            chunk_overlap (int): Number of overlapping characters between chunks.
        """
        try:
            self.chunk_size = chunk_size  # Store chunk size
            self.chunk_overlap = chunk_overlap  # Store overlap size

            # ✅ Initialize LangChain's text splitter
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
            )
        except Exception as e:
            raise RuntimeError(f"❌ Failed to initialize TextChunker: {e}")

    def chunk_dataframe(
        self, df: pd.DataFrame, text_column: str, metadata_columns: list
    ) -> list:
        """
        Splits the narratives in a DataFrame into text chunks with attached metadata.

        Args:
            df (pd.DataFrame): Input DataFrame containing complaint narratives.
            text_column (str): Name of the text column to split.
            metadata_columns (list): List of columns to preserve as metadata.

        Returns:
            list: A list of dictionaries, each containing:
                - 'text': the chunked text
                - metadata fields (e.g., 'Complaint ID', 'Product', 'Chunk Index')
        """
        try:
            if text_column not in df.columns:
                raise ValueError(f"❌ Column '{text_column}' not found in DataFrame.")

            for col in metadata_columns:
                if col not in df.columns:
                    raise ValueError(
                        f"❌ Metadata column '{col}' not found in DataFrame."
                    )

            all_chunks = []  # To store all chunks with metadata

            for idx, row in df.iterrows():  # Iterate through each complaint
                text = str(row[text_column])  # Get the text to chunk

                # Split text into chunks using LangChain
                chunks = self.splitter.split_text(text)

                # Generate metadata for each chunk
                for i, chunk_text in enumerate(chunks):
                    chunk_data = {"text": chunk_text}  # Store chunk text
                    for meta_col in metadata_columns:
                        chunk_data[meta_col] = row[meta_col]  # Copy over metadata
                    chunk_data["chunk_index"] = i  # Add chunk index
                    all_chunks.append(chunk_data)  # Append chunk with metadata

            return all_chunks  # Return full list of chunked records

        except Exception as e:
            raise RuntimeError(f"❌ Failed to chunk narratives: {e}")
