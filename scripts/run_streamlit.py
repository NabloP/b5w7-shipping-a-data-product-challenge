"""
run_streamlit.py – Entry Point for CrediTrust Complaint AI Chatbot (B5W6)
-------------------------------------------------------------------------

Launches the Streamlit-based Retrieval-Augmented Generation (RAG) chatbot
for internal use by Product, Support, and Compliance teams.

To run:
    streamlit run scripts/run_streamlit.py

Author: Nabil Mohamed
Date: July 2025
"""

# ------------------------------------------------------------------------------
# 🛠 Ensure Project Root for src/ Imports
# ------------------------------------------------------------------------------
import os
import sys

if os.path.basename(os.getcwd()) == "notebooks":
    os.chdir("..")
    print("📂 Changed to project root.")

project_root = os.getcwd()
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"✅ Project root added to sys.path: {project_root}")

# Optional diagnostic
expected_path = "vector_store/chroma_db"
print(
    "📁 Vector store path found."
    if os.path.exists(expected_path)
    else f"⚠️ Path not found: {expected_path}"
)

# ------------------------------------------------------------------------------
# 🚀 Launch Streamlit App
# ------------------------------------------------------------------------------
from ui.app import launch_app

if __name__ == "__main__":
    launch_app()
