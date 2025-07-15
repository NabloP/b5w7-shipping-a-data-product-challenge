# ------------------------------------------------------------------------------
# 📄 PromptEngineer Module for B5W6 – Intelligent Complaint Analysis (v2)
# ------------------------------------------------------------------------------
# Author: Nabil Mohamed
# Date: July 2025
# Description:
#   Builds high-quality prompts for Retrieval-Augmented Generation (RAG) by
#   combining user questions with semantically retrieved complaint chunks.
#   Includes strict grounding instructions to minimize hallucination risk.
# ------------------------------------------------------------------------------

# ---------------------------
# Standard Library Imports
# ---------------------------
from typing import List, Dict  # For type annotations


# ---------------------------
# PromptEngineer Class
# ---------------------------


class PromptEngineer:
    """
    Assembles prompt text for RAG by combining retrieved complaint chunks with user question.
    Applies strict grounding to minimize hallucination and ensure factual answers.
    """

    def __init__(self, max_context_length: int = 3000):
        """
        Initializes the PromptEngineer.

        Args:
            max_context_length (int): Maximum number of characters from retrieved chunks to include.
        """
        try:
            if not isinstance(max_context_length, int) or max_context_length <= 0:
                raise ValueError("❌ max_context_length must be a positive integer.")

            self.max_context_length = max_context_length  # Store maximum context length

        except Exception as e:
            raise RuntimeError(f"❌ Failed to initialize PromptEngineer: {e}") from e

    def build_prompt(self, question: str, retrieved_chunks: List[Dict]) -> str:
        """
        Builds the final prompt by combining retrieved complaint chunks and the user question.

        Args:
            question (str): The business question to be answered.
            retrieved_chunks (List[Dict]): List of retrieved complaint chunks with text and metadata.

        Returns:
            str: The assembled prompt ready for LLM completion.
        """
        try:
            if not isinstance(question, str) or not question.strip():
                raise ValueError("❌ Question must be a non-empty string.")

            if not isinstance(retrieved_chunks, list) or not retrieved_chunks:
                raise ValueError("❌ Retrieved chunks must be a non-empty list.")

            # ✅ Assemble context from complaint chunks up to max length
            context_parts = []
            current_length = 0

            for item in retrieved_chunks:
                chunk_text = item.get("document", "")
                if not chunk_text:
                    continue  # Skip empty

                if current_length + len(chunk_text) <= self.max_context_length:
                    context_parts.append(chunk_text.strip())
                    current_length += len(chunk_text)
                else:
                    break  # Stop adding when max context reached

            combined_context = "\n---\n".join(context_parts)  # Add separators

            # ✅ Build final prompt using strict template
            prompt = f"""
You are an impartial financial assistant for CrediTrust Financial. Your task is to answer business questions using only the information provided in the retrieved customer complaint narratives.

Instructions:
- Base your answer strictly on the retrieved context below.
- Do not add information, speculate, or make assumptions beyond the given context.
- If the context does not contain enough information to confidently answer the question, clearly say:  
  **"The available complaint data does not provide enough information to answer this question."**

Retrieved Complaint Narratives:
{combined_context}

Business Question:
{question}

Answer:
""".strip()

            return prompt

        except Exception as e:
            raise RuntimeError(f"❌ Failed to build prompt: {e}") from e
