# ------------------------------------------------------------------------------
# 📄 QualitativeEvaluator Module for B5W6 – Intelligent Complaint Analysis (v1)
# ------------------------------------------------------------------------------
# Author: Nabil Mohamed
# Date: July 2025
# Description:
#   Stores and manages qualitative evaluation of RAG pipeline outputs.
#   Enables manual scoring of relevance, accuracy, and completeness.
# ------------------------------------------------------------------------------

# ---------------------------
# Standard Library Imports
# ---------------------------
import pandas as pd  # For evaluation table storage


# ---------------------------
# QualitativeEvaluator Class
# ---------------------------


class QualitativeEvaluator:
    """
    Manages storage and display of qualitative evaluations of RAG outputs.
    """

    def __init__(self):
        """
        Initializes an empty evaluation table.
        """
        try:
            self.results = pd.DataFrame(
                columns=[
                    "Question",
                    "Retrieved Evidence (Preview)",
                    "Generated Answer",
                    "Relevance (✅/❌)",
                    "Accuracy (✅/❌)",
                    "Completeness (✅/❌)",
                    "Evaluator Comments",
                ]
            )
        except Exception as e:
            raise RuntimeError(
                f"❌ Failed to initialize QualitativeEvaluator: {e}"
            ) from e

    def add_entry(
        self,
        question,
        retrieved_chunks,
        generated_answer,
        relevance=None,
        accuracy=None,
        completeness=None,
        comments=None,
    ):
        """
        Adds a new evaluation record.

        Args:
            question (str): The original business question.
            retrieved_chunks (list): List of retrieved complaint chunk dicts.
            generated_answer (str): The generated LLM answer.
            relevance (str): Optional manual score.
            accuracy (str): Optional manual score.
            completeness (str): Optional manual score.
            comments (str): Optional evaluator comment.
        """
        try:
            evidence_preview = "\n---\n".join(
                [chunk.get("document", "")[:300] for chunk in retrieved_chunks[:3]]
            )

            new_row = {
                "Question": question,
                "Retrieved Evidence (Preview)": evidence_preview,
                "Generated Answer": generated_answer,
                "Relevance (✅/❌)": relevance,
                "Accuracy (✅/❌)": accuracy,
                "Completeness (✅/❌)": completeness,
                "Evaluator Comments": comments,
            }

            self.results = pd.concat(
                [self.results, pd.DataFrame([new_row])], ignore_index=True
            )

        except Exception as e:
            raise RuntimeError(f"❌ Failed to add evaluation entry: {e}") from e

    def display(self):
        """
        Displays the current evaluation table.
        """
        try:
            display(self.results)
        except Exception as e:
            print(f"❌ Failed to display evaluation results: {e}")
