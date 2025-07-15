"""
rag_interactive_runner.py

📄 Interactive Retrieval-Augmented Generation (RAG) over complaint data.
✅ Single-question interactive use with anti-hallucination safeguards.
✅ Uses ChromaDB for retrieval + Gemini API for generation.

Author: Nabil Mohamed
Date: July 2025
"""

# ------------------------------------------------------------------------------
# 📦 Standard Library Imports
# ------------------------------------------------------------------------------
import os
import sys
import time
import threading
import itertools
import pathlib

# ------------------------------------------------------------------------------
# 📦 Third-Party Imports
# ------------------------------------------------------------------------------
import pandas as pd
from tqdm import tqdm


# ------------------------------------------------------------------------------
# 🔄 Utility: Loading Spinner
# ------------------------------------------------------------------------------
def loading_animation(message, duration=2):
    spinner = itertools.cycle(["|", "/", "-", "\\"])
    end_time = time.time() + duration
    while time.time() < end_time:
        sys.stdout.write(f"\r{message} {next(spinner)}")
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write("\r" + " " * len(message) + "\r")  # Clear line


# ------------------------------------------------------------------------------
# 🏁 Step 1: Ensure Project Root
# ------------------------------------------------------------------------------
current_dir = pathlib.Path(__file__).resolve().parent
for parent in current_dir.parents:
    if (parent / "src").exists() and (parent / "vector_store").exists():
        project_root = parent
        break
else:
    raise RuntimeError("❌ Project root not found.")

os.chdir(project_root)
sys.path.insert(0, str(project_root))
print(f"✅ Running from project root: {project_root}")

# ------------------------------------------------------------------------------
# 📦 Project-Specific Imports
# ------------------------------------------------------------------------------
from src.rag.chroma_loader import ChromaLoader
from src.rag.retriever import QuestionRetriever
from src.rag.prompt_template import PromptEngineer
from src.rag.answer_generator import AnswerGenerator

# ------------------------------------------------------------------------------
# 🏁 Step 2: Load ChromaDB Vector Store (with progress)
# ------------------------------------------------------------------------------
vector_store_path = str(project_root / "vector_store" / "chroma_db")
collection_name = "complaint_chunks"
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

print("\n🚀 Loading ChromaDB Vector Store...")

try:
    loading_animation("🔄 Initializing ChromaLoader...", duration=1.5)
    chroma_loader = ChromaLoader(
        vector_store_path, collection_name, embedding_model_name
    )

    with tqdm(
        total=1,
        desc="🔗 Loading Vector Store",
        bar_format="{l_bar}{bar} [ time left: {remaining} ]",
    ) as pbar:
        vector_store = chroma_loader.load_collection()
        pbar.update(1)

    print("✅ Vector store loaded successfully.\n")

except Exception as e:
    print(f"❌ Failed to load ChromaDB: {e}")
    sys.exit(1)

# ------------------------------------------------------------------------------
# 🏁 Step 3: Initialize Components (Animated)
# ------------------------------------------------------------------------------
print("⚙️ Initializing Retrieval & Generation Components...\n")
try:
    loading_animation("🔍 Initializing QuestionRetriever...", duration=1.2)
    retriever = QuestionRetriever(vector_store, embedding_model_name, top_k=50)

    loading_animation("📝 Initializing PromptEngineer...", duration=1.2)
    prompt_engineer = PromptEngineer(max_context_length=3000)

    loading_animation("🤖 Initializing AnswerGenerator (Gemini)...", duration=1.2)
    answer_generator = AnswerGenerator(
        api_key="AIzaSyDM6YAwR1ajk6fzhGepD0q7sOtVqrnBuMs", model="gemini-2.5-flash"
    )

    print("✅ All components initialized.\n")
except Exception as e:
    print(f"❌ Initialization failed: {e}")
    sys.exit(1)

# ------------------------------------------------------------------------------
# 🏁 Step 4: Capture User Question
# ------------------------------------------------------------------------------
print("🗨️ Ready for your business question.")
user_question = input("\n❓ Enter your business question:\n> ").strip()

if not user_question:
    print("⚠️ No question provided. Exiting.")
    sys.exit(0)

# ------------------------------------------------------------------------------
# 🏁 Step 5: Retrieve Complaint Chunks (Progress)
# ------------------------------------------------------------------------------
try:
    print("\n🔎 Searching for relevant complaint data...")
    time.sleep(0.5)

    with tqdm(
        total=1,
        desc="🔍 Retrieving Chunks",
        bar_format="{l_bar}{bar} [ time left: {remaining} ]",
    ) as pbar:
        retrieved_chunks = retriever.retrieve(user_question)
        pbar.update(1)

    if not retrieved_chunks:
        print("⚠️ No complaint chunks found. Cannot generate answer.")
        sys.exit(0)

    print(f"✅ Retrieved {len(retrieved_chunks)} complaint chunks.\n")

except Exception as e:
    print(f"❌ Retrieval failed: {e}")
    sys.exit(1)

# ------------------------------------------------------------------------------
# 🏁 Step 6: Assemble Prompt (Animated)
# ------------------------------------------------------------------------------
try:
    loading_animation("✏️ Assembling Prompt...", duration=1.5)
    prompt = prompt_engineer.build_prompt(user_question, retrieved_chunks)
    print("✅ Prompt ready.\n")
except Exception as e:
    print(f"❌ Prompt assembly failed: {e}")
    sys.exit(1)

# ------------------------------------------------------------------------------
# 🏁 Step 7: Generate Answer (Animated)
# ------------------------------------------------------------------------------
try:
    loading_animation("🤖 Generating Answer...", duration=3)

    answer = answer_generator.generate_answer(prompt)

    if (
        any(
            phrase in answer.lower()
            for phrase in [
                "not enough information",
                "insufficient data",
                "cannot answer based on provided context",
            ]
        )
        or len(retrieved_chunks) == 0
    ):
        final_answer = "The available complaint data does not provide enough information to answer this question."
    else:
        final_answer = answer.strip()

    print("\n📝 Final Answer:")
    print("-" * 60)
    print(final_answer)
    print("-" * 60)

except Exception as e:
    print(f"❌ Answer generation failed: {e}")
    sys.exit(1)

# ------------------------------------------------------------------------------
# 🏁 Step 8: Display Top 2 Evidence Chunks
# ------------------------------------------------------------------------------
print("\n🔍 Evidence Preview (Top 2 Chunks):\n")
for idx, chunk in enumerate(retrieved_chunks[:2], 1):
    snippet = chunk["document"][:300].strip() + "..."
    print(f"🔹 Chunk {idx}:\n{snippet}\n")

# ------------------------------------------------------------------------------
# ✅ End of Script
# ------------------------------------------------------------------------------
