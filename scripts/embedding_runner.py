# embedding_runner.py – Modular CLI Runner for Task 2: Embedding & Indexing (B5W6)
# ------------------------------------------------------------------------------
# Author: Nabil Mohamed
# Version: 2025-07-08 (Interim Submission Version with Full Documentation)
# Purpose: Executes full embedding and semantic indexing pipeline for
# CrediTrust Financial’s Intelligent Complaint Analysis Initiative (B5W6 – Task 2)

# ------------------------------------------------------------------------------
# 🛠 Environment Setup: Project Root and Path Configuration
# ------------------------------------------------------------------------------
import os  # For directory navigation
import sys  # For system path modification
import warnings  # To suppress warnings

# ✅ If running from notebooks/, move up to project root for consistent pathing
if os.path.basename(os.getcwd()) == "notebooks":  # Check current folder
    os.chdir("..")  # Move up one level to project root
    print("📂 Changed working directory to project root")  # Confirm directory change

# ✅ Ensure project root is in sys.path for src/ imports to work
project_root = os.getcwd()  # Get current working directory path
if project_root not in sys.path:  # If path not already added
    sys.path.insert(0, project_root)  # Add project root to sys.path
    print(f"✅ Added to sys.path: {project_root}")  # Confirm addition

# ✅ Suppress any non-critical warnings to keep logs clean
warnings.filterwarnings("ignore")  # Disable warnings globally

# ------------------------------------------------------------------------------
# 📦 Imports: Modular Pipeline Components
# ------------------------------------------------------------------------------
import pandas as pd  # For data manipulation

from src.data_loader import (
    ComplaintChunkProcessor,
)  # Data loader for cleaned complaints
from src.chunking.text_cleaner import MinimalTextPreprocessor  # Minimal text cleaner
from src.chunking.text_chunker import TextChunker  # Text chunking utility
from src.chunking.embedding_generator import (
    EmbeddingGenerator,
)  # Sentence embedding generator
from src.chunking.vector_store_building import (
    VectorStoreBuilder,
)  # ChromaDB vector store builder

# ------------------------------------------------------------------------------
# 📥 Step 1: Load Pre-Cleaned Complaint Dataset (Task 1 Output)
# ------------------------------------------------------------------------------
data_path = "data/interim/filtered_complaints.csv"  # Path to pre-cleaned dataset

# ✅ Initialize complaint processor with required file paths
loader = ComplaintChunkProcessor(
    filepath=data_path, output_path="data/interim/dummy_output.csv"
)

try:
    df = loader.load_cleaned_data()  # Attempt to load pre-cleaned data
except Exception as e:
    print(f"❌ Failed to load cleaned complaints: {e}")  # Print error on failure
    sys.exit(1)  # Exit script if loading fails

# ✅ Check for empty DataFrame before proceeding
if df.empty:  # If DataFrame has no rows
    print("⚠️ No data found. Exiting.")  # Print warning message
    sys.exit(1)  # Exit to prevent downstream failure

# ✅ Optional: Sample down to 25,000 rows for performance
df = df.sample(n=25_000, random_state=42).reset_index(drop=True)  # Random sampling
print(
    f"✅ Loaded and sampled dataset: {df.shape[0]:,} rows"
)  # Print sample confirmation

# ------------------------------------------------------------------------------
# ✨ Step 2: Apply Minimal Text Preprocessing for Embedding Readiness
# ------------------------------------------------------------------------------
preprocessor = MinimalTextPreprocessor()  # Instantiate text preprocessor

try:
    df = preprocessor.apply_to_dataframe(  # Apply minimal text cleaning
        df,  # Input DataFrame
        text_column="Consumer complaint narrative",  # Column to clean
        new_column_name="cleaned_narrative",  # New column for cleaned text
    )
except Exception as e:
    print(f"❌ Text cleaning failed: {e}")  # Print failure message
    sys.exit(1)  # Exit on error

print("✅ Minimal text cleaning applied.")  # Confirm completion

# ------------------------------------------------------------------------------
# ✂️ Step 3: Split Narratives into Chunks for Embedding
# ------------------------------------------------------------------------------
chunker = TextChunker(
    chunk_size=500, chunk_overlap=50
)  # Initialize chunker with parameters

try:
    chunked_data = chunker.chunk_dataframe(  # Apply chunking to cleaned DataFrame
        df,  # Input DataFrame
        text_column="cleaned_narrative",  # Column containing cleaned text
        metadata_columns=["Complaint ID", "Product"],  # Preserve key metadata
    )
except Exception as e:
    print(f"❌ Chunking failed: {e}")  # Print error message
    sys.exit(1)  # Exit on failure

print(
    f"✅ Text chunking complete: {len(chunked_data):,} chunks generated."
)  # Confirm success

# ------------------------------------------------------------------------------
# 🔗 Step 4: Generate Sentence Embeddings for Complaint Chunks
# ------------------------------------------------------------------------------
embedder = EmbeddingGenerator(
    model_name="all-MiniLM-L6-v2", batch_size=64
)  # Initialize embedding generator

try:
    chunk_texts = [
        chunk["text"] for chunk in chunked_data
    ]  # Extract text chunks from chunked data
    embeddings = embedder.generate_embeddings(
        chunk_texts
    )  # Generate dense vector representations
except Exception as e:
    print(f"❌ Embedding generation failed: {e}")  # Print error message
    sys.exit(1)  # Exit on error

# ✅ Check for valid embedding result
if embeddings is None:  # If no embeddings were generated
    print("⚠️ Embedding generation returned None. Exiting.")  # Print warning
    sys.exit(1)  # Exit

print(
    f"✅ Embeddings generated: {embeddings.shape[0]:,} vectors of dimension {embeddings.shape[1]}"
)  # Confirm

# ------------------------------------------------------------------------------
# 🗂️ Step 5: Build and Persist ChromaDB Vector Store
# ------------------------------------------------------------------------------
builder = VectorStoreBuilder(  # Initialize ChromaDB builder
    persist_directory="vector_store/chroma_db",  # Path for persistence
    collection_name="complaint_chunks",  # Logical collection name
)

try:
    metadatas = [  # Build metadata for each chunk
        {
            "Complaint ID": chunk.get("Complaint ID", "N/A"),  # Preserve Complaint ID
            "Product": chunk.get("Product", "N/A"),  # Preserve Product category
            "Chunk Index": chunk.get("chunk_index", 0),  # Preserve chunk index
        }
        for chunk in chunked_data  # Iterate over all chunks
    ]

    vector_store = builder.build_chroma_store(  # Build ChromaDB vector store
        documents=chunk_texts,  # Input text documents
        embeddings=embeddings,  # Precomputed embeddings
        metadatas=metadatas,  # Associated metadata
    )

except Exception as e:
    print(f"❌ ChromaDB index build failed: {e}")  # Print failure message
    sys.exit(1)  # Exit on error

# ✅ Confirm vector store creation
if vector_store is not None:  # If build was successful
    print("✅ ChromaDB vector store built and saved successfully.")  # Confirm success
else:
    print("⚠️ ChromaDB vector store build returned None.")  # Warn on failure

# ------------------------------------------------------------------------------
# 🎯 Task 2 Embedding & Indexing Completed Successfully
# ------------------------------------------------------------------------------
print(
    "🎯 Task 2 completed successfully. Ready for Task 3 (Retriever + RAG)."
)  # Final message
