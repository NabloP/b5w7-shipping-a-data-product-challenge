# ------------------------------------------------------------------------------
# 📄 MinimalTextPreprocessor Module for B5W6 – Intelligent Complaint Analysis
# ------------------------------------------------------------------------------
# Author: Nabil Mohamed
# Date: July 2025
# Description: Minimal text preprocessing class for embedding readiness.
# Applies lossless cleaning (lowercasing, whitespace normalization) while
# retaining semantic content for downstream RAG and semantic search tasks.
# ------------------------------------------------------------------------------

# ---------------------------
# Standard Library Imports
# ---------------------------
import re  # For regular expression operations
import pandas as pd  # For handling structured datasets

# ---------------------------
# Minimal Text Preprocessing Class
# ---------------------------


class MinimalTextPreprocessor:
    """
    A minimal text preprocessing class for intelligent complaint analysis.
    Applies light, lossless cleaning suitable for text embedding and semantic search.
    """

    def __init__(self):
        """
        Initializes the preprocessor.
        No parameters needed since operations are minimal and stateless.
        """
        pass  # No initialization logic required

    def clean_text(self, text):
        """
        Cleans a single text string by lowercasing and removing extra whitespace.

        Args:
            text (str): The text to be cleaned.

        Returns:
            str: The cleaned text string.
        """
        try:
            if pd.isnull(text):  # Check if input is missing or NaN
                return ""  # Return empty string for null inputs

            text = str(text)  # Ensure input is string type
            text = re.sub(
                r"\s+", " ", text
            )  # Replace multiple spaces/newlines with single space
            text = (
                text.strip().lower()
            )  # Trim leading/trailing spaces and lowercase text

            return text  # Return cleaned text

        except Exception as e:
            print(f"❌ Error cleaning text: {e}")  # Print error for debugging
            return ""  # Return empty string on failure

    def apply_to_dataframe(self, df, text_column, new_column_name="cleaned_narrative"):
        """
        Applies text cleaning to a specified column in a pandas DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame containing complaint narratives.
            text_column (str): Name of the column containing original text.
            new_column_name (str): Name of the new column for cleaned text.

        Returns:
            pd.DataFrame: DataFrame with an added column of cleaned text.
        """
        try:
            if text_column not in df.columns:  # Validate text column presence
                raise ValueError(
                    f"Column '{text_column}' not found in DataFrame."
                )  # Raise descriptive error

            # Apply cleaning function to each row in the target column
            df[new_column_name] = df[text_column].apply(
                self.clean_text
            )  # Apply cleaning function

            return df  # Return modified DataFrame

        except Exception as e:
            print(f"❌ Failed to apply text cleaning: {e}")  # Print error message
            return df  # Return original DataFrame without crashing
